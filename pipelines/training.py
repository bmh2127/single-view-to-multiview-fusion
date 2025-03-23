import logging
import os
from pathlib import Path
import numpy as np
import tempfile

from common import (
    PYTHON,
    packages,
    DatasetMixin,
    build_unet3d_model,
    percentile_normalization,
    compute_n_ssim,
    configure_logging,
    load_image
)
from gradient_analysis import (
    analyze_gradient_direction, 
    normalize_gradient_direction, 
    visualize_gradient_analysis
)
from gradient_features import GradientFeatureProcessor
from metaflow import (
    FlowSpec,
    Parameter,
    card,
    conda_base,
    current,
    environment,
    project,
    resources,
    step,
)

configure_logging()

# pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
@project(name="fusemycell")
@conda_base(
    python="3.12.9",
    libraries={
    #     "conda-forge::pytorch": "2.6.0",
    #     "conda-forge::torchvision": "0.21",  # Latest version
    #     # "conda-forge::torchaudio": "2.6.0",  # Latest version
    #     "conda-forge::pandas": "",
    #     "conda-forge::numpy": "",
        "conda-forge::mlflow": "",
    #     "conda-forge::tifffile": "",
    #     "conda-forge::scikit-image": "",
    #     "conda-forge::matplotlib": "",
    #     "conda-forge::boto3": "",
    #     "conda-forge::metaflow": "",
    #     "conda-forge::cellpose": "",
    },
    packages=packages(
        "scikit-learn",
        "pandas",
        "numpy",
        "keras",
        "tensorflow",
        "boto3",
        "mlflow",
        "psutil",
        "pynvml",
        "metaflow",
        "pytorch",
        "tifffile",
        "tqdm",
        "scikit-image"
    ),
)
class FuseMyCellTraining(FlowSpec, DatasetMixin):
    """FuseMyCell Training Pipeline.

    This pipeline trains, evaluates, and registers a 3D U-Net model to predict
    fused multi-view light sheet microscopy images from a single view.
    """

    mlflow_tracking_uri = Parameter(
        "mlflow-tracking-uri",
        help="Location of the MLflow tracking server.",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
    )

    training_epochs = Parameter(
        "training-epochs",
        help="Number of epochs that will be used to train the model.",
        default=10,
    )

    training_batch_size = Parameter(
        "training-batch-size",
        help="Batch size that will be used to train the model.",
        default=4,
    )

    learning_rate = Parameter(
        "learning-rate",
        help="Learning rate for model training.",
        default=0.001,
    )
    
    use_physics = Parameter(
        "use-physics",
        help="Whether to use physics-informed neural network",
        default=False,
    )
    
    n_ssim_threshold = Parameter(
        "n-ssim-threshold",
        help="Minimum normalized SSIM threshold required to register the model.",
        default=0.7,
    )

    use_patch_inference = Parameter(
        "use-patch-inference",
        help="Whether to use patch-based inference for large volumes",
        default=True,
    )
    
    test_mode = Parameter(
        "test-mode",
        help="Run in test mode with mock data",
        default=False,
    )

    normalize_gradients = Parameter(
        "normalize-gradients",
        help="Whether to normalize gradient directions during training",
        default=True,
    )

    use_gradient_features = Parameter(
        "use-gradient-features",
        help="Whether to include gradient direction features as additional input channels",
        default=False,
    )
    
    local_mode = Parameter(
        "local-mode",
        help="Run in local development mode with patch-based training",
        default=False,
    )
    use_tensorflow = Parameter(
        "use-tensorflow",
        help="Whether to use TensorFlow instead of PyTorch for the model",
        default=True,
    ),
    install_tensorflow = Parameter(
        "install-tensorflow",
        help="Whether to install TensorFlow for macOS",
        default=True,
    )

    @card
    @step
    def start(self):
        """Start and prepare the FuseMyCell training pipeline."""
        import mlflow
        import sys
        import subprocess
        
        # Install TensorFlow for macOS if requested
        if self.install_tensorflow and sys.platform == 'darwin':
            try:
                # Try importing to see if already installed
                import tensorflow as tf
                logging.info(f"TensorFlow {tf.__version__} already installed")
            except ImportError:
                logging.info("Installing TensorFlow for macOS...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "tensorflow-macos", "tensorflow-metal"
                ])
                # Import to verify installation
                import tensorflow as tf
                logging.info(f"Installed TensorFlow {tf.__version__}")

        # Try to set the MLflow tracking URI, but don't fail if it doesn't work
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        except Exception as e:
            logging.warning(f"Could not set MLflow tracking URI: {e}")

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        # Load dataset
        try:
            if self.local_mode:
                logging.info("Running in local development mode with patch-based training")
                # Use patch-based dataset if in local mode
                try:
                    from utils.patch_dataset import PatchDataset
                    from torch.utils.data import DataLoader, random_split
                    
                    metadata_file = Path(self.dataset_dir) / "patch_metadata.json"
                    if metadata_file.exists():
                        # Create dataset and data loaders from patches
                        dataset = PatchDataset(
                            metadata_file=metadata_file,
                            apply_augmentations=True
                        )
                        
                        # Split dataset
                        val_size = int(len(dataset) * 0.2)  # 20% for validation
                        train_size = len(dataset) - val_size
                        
                        train_dataset, val_dataset = random_split(
                            dataset, [train_size, val_size]
                        )
                        
                        # Create data loaders
                        self.train_loader = DataLoader(
                            train_dataset,
                            batch_size=self.training_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True
                        )
                        
                        self.val_loader = DataLoader(
                            val_dataset,
                            batch_size=self.training_batch_size,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True
                        )
                        
                        logging.info(f"Created data loaders from patch dataset with {len(dataset)} samples")
                    else:
                        logging.warning("Patch metadata not found. Run extract_patches.py first.")
                        raise FileNotFoundError(f"Patch metadata not found at {metadata_file}")
                except Exception as e:
                    logging.warning(f"Failed to load patch dataset: {e}")
                    logging.info("Falling back to regular dataset")
                    data_loaders = self.load_dataset()
                    self.train_loader = data_loaders['train']
                    self.val_loader = data_loaders['val']
            else:
                # Standard dataset loading for non-local mode
                data_loaders = self.load_dataset()
                self.train_loader = data_loaders['train']
                self.val_loader = data_loaders['val']
                
        except Exception as e:
            # Create dummy loaders if all loading methods fail
            logging.warning(f"Failed to load any dataset: {e}")
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            dummy_data = torch.randn(1, 1, 64, 128, 128)
            dummy_target = torch.randn(1, 1, 64, 128, 128)
            dummy_dataset = TensorDataset(dummy_data, dummy_target)
            self.train_loader = DataLoader(dummy_dataset, batch_size=1)
            self.val_loader = DataLoader(dummy_dataset, batch_size=1)
            logging.warning("Using dummy data for testing")

        # Try to start an MLflow run, but don't fail the flow if it doesn't work
        try:
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
            logging.info(f"MLflow run ID: {self.mlflow_run_id}")
        except Exception as e:
            logging.warning(f"Failed to connect to MLflow server: {e}")
            # Set a mock run ID for testing
            self.mlflow_run_id = "mock_run_id"
        
        # Continue to model creation
        self.next(self.create_model)
    
    @card
    # @resources(memory=8192, gpu=1)
    @environment(vars={"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                       "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),})
    @step
    def create_model(self):
        """Create the 3D U-Net model for image fusion."""
        import torch
        # Get input shape from patch size
        # Create gradient feature processor
        self.feature_processor = GradientFeatureProcessor(
            use_gradient_features=self.use_gradient_features
        )
        
        # Get input shape from patch size
        patch_size = self.get_patch_size()
        
        # Determine number of input channels based on whether we're using gradient features
        in_channels = 7 if self.use_gradient_features else 1  # 1 original + 6 gradient features
        
        # Build model
        if self.use_tensorflow:
            self.model = build_unet3d_model(
                input_shape=(in_channels, *patch_size),
                use_physics=self.use_physics,
                use_tensorflow=True,
                learning_rate=self.learning_rate
            )
            self.is_tensorflow = True
        else:
            import torch
            self.model = build_unet3d_model(
                input_shape=(in_channels, *patch_size),
                use_physics=self.use_physics,
                use_tensorflow=False
            )
            # Configure training with proper device detection
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logging.info("Using Apple Silicon GPU via MPS backend")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("Using NVIDIA GPU via CUDA")
            else:
                self.device = torch.device("cpu")
                logging.info("Using CPU (no GPU available)")
            
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            self.criterion = torch.nn.MSELoss()
            self.is_tensorflow = False
        
        # Continue to training
        self.next(self.train_model)
        
    @card
    # @resources(memory=8192, gpu=1)
    @environment(vars={"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                       "KERAS_BACKEND": os.getenv("KERAS_BACKEND", "tensorflow"),})
    @step
    def train_model(self):
        """Train the 3D U-Net model."""
        import mlflow
        import matplotlib.pyplot as plt
        
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Initialize variables to track training progress
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        # Start MLflow run
        with mlflow.start_run(run_id=self.mlflow_run_id):
            logging.info(f"Starting training for {self.training_epochs} epochs")
            
            if self.is_tensorflow:
                # TensorFlow training implementation
                import tensorflow as tf
                import numpy as np
                from tqdm import tqdm
                
                # Convert PyTorch data loaders to TensorFlow format
                train_x = []
                train_y = []
                val_x = []
                val_y = []
                file_paths = []
                
                logging.info("Preparing data for TensorFlow training...")
                
                # Process training data
                for batch in tqdm(self.train_loader, desc="Processing training data"):
                    if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
                        for i in range(len(batch['input'])):
                            # Convert from PyTorch (C,D,H,W) to TensorFlow (D,H,W,C)
                            input_np = batch['input'][i].numpy().transpose(1,2,3,0)
                            target_np = batch['target'][i].numpy().transpose(1,2,3,0)
                            train_x.append(input_np)
                            train_y.append(target_np)
                
                # Process validation data
                for batch in tqdm(self.val_loader, desc="Processing validation data"):
                    if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
                        for i in range(len(batch['input'])):
                            input_np = batch['input'][i].numpy().transpose(1,2,3,0)
                            target_np = batch['target'][i].numpy().transpose(1,2,3,0)
                            val_x.append(input_np)
                            val_y.append(target_np)
                        if 'file_path' in batch:
                            file_paths.extend(batch['file_path'])
                
                train_x = np.array(train_x)
                train_y = np.array(train_y)
                val_x = np.array(val_x)
                val_y = np.array(val_y)
                
                logging.info(f"Training data shape: {train_x.shape}, Validation data shape: {val_x.shape}")
                
                # Create TensorFlow datasets
                train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
                train_dataset = train_dataset.batch(self.training_batch_size)
                
                val_dataset = tf.data.Dataset.from_tensor_slices((val_x, val_y))
                val_dataset = val_dataset.batch(self.training_batch_size)
                
                # Create callback for saving the best model during training
                temp_dir = tempfile.mkdtemp()
                checkpoint_path = os.path.join(temp_dir, "best_model.keras")
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    save_best_only=True,
                    monitor='val_loss',
                    mode='min',
                    verbose=1
                )

                # Create custom MLflow logging callback
                class MLflowLoggingCallback(tf.keras.callbacks.Callback):
                    def __init__(self, total_epochs):
                        super().__init__()
                        self.total_epochs = total_epochs
                        
                    def on_epoch_end(self, epoch, logs=None):
                        mlflow.log_metrics({
                            "train_loss": logs['loss'],
                            "val_loss": logs['val_loss']
                        }, step=epoch)
                        
                        # Save to our tracking lists
                        train_losses.append(logs['loss'])
                        val_losses.append(logs['val_loss'])
                        
                        # Update best val loss for our tracking
                        nonlocal best_val_loss
                        if logs['val_loss'] < best_val_loss:
                            best_val_loss = logs['val_loss']
                            
                        logging.info(
                            "Epoch %d/%d - Train loss: %.4f, Val loss: %.4f",
                            epoch+1, self.total_epochs, logs['loss'], logs['val_loss']
                        )

                # Train the model with TensorFlow
                history = self.model.fit(
                    train_dataset,
                    epochs=self.training_epochs,
                    validation_data=val_dataset,
                    callbacks=[
                        checkpoint_callback,
                        MLflowLoggingCallback(total_epochs=self.training_epochs),
                        tf.keras.callbacks.TensorBoard(log_dir=f"./logs/{self.mlflow_run_id}")
                    ],
                    verbose=1
                )

                # Store the best model path (from checkpoint callback)
                self.best_model_path = checkpoint_path

                # Save the final trained model as well (optional)
                final_model_path = os.path.join(temp_dir, "final_model.keras")
                self.model.save(final_model_path)

                # Log both models to MLflow
                mlflow.log_artifact(self.best_model_path, "best_model")
                mlflow.log_artifact(final_model_path, "final_model")
                
            else:
                # Original PyTorch training implementation
                import torch
                from tqdm import tqdm
                
                self.model.train()
                
                for epoch in range(self.training_epochs):
                    # Training phase
                    self.model.train()
                    epoch_loss = 0
                    
                    with tqdm(self.train_loader, unit="batch") as train_pbar:
                        train_pbar.set_description(f"Epoch {epoch+1}/{self.training_epochs}")
                        
                        for batch in train_pbar:
                            # Process batch data
                            if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
                                # Dictionary format from your dataset
                                if self.use_gradient_features:
                                    batch = self.feature_processor.process_batch(batch)
                                    inputs = batch['input_with_features'].to(self.device)
                                else:
                                    inputs = batch['input'].to(self.device)
                                
                                targets = batch['target'].to(self.device)
                                
                            elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                # List format (likely from TensorDataset in test mode)
                                inputs = batch[0].to(self.device)
                                targets = batch[1].to(self.device)
                                
                                # Handle gradient features for list format if needed
                                if self.use_gradient_features:
                                    # Convert to appropriate format for processor
                                    temp_batch = {'input': inputs}
                                    processed_batch = self.feature_processor.process_batch(temp_batch)
                                    inputs = processed_batch['input_with_features']
                            else:
                                # Unknown format - log and skip
                                logging.warning(f"Unknown batch format: {type(batch)}")
                                continue
                            
                            # Forward pass
                            self.optimizer.zero_grad()
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            
                            # Backward pass and optimize
                            loss.backward()
                            self.optimizer.step()
                            
                            # Update loss
                            epoch_loss += loss.item()
                            train_pbar.set_postfix(loss=loss.item())
                    
                    avg_train_loss = epoch_loss / len(self.train_loader)
                    train_losses.append(avg_train_loss)
                    
                    # Validation phase
                    self.model.eval()
                    val_loss = 0
                    
                    with torch.no_grad():
                        for batch in tqdm(self.val_loader, desc="Validation"):
                            inputs = batch['input'].to(self.device)
                            targets = batch['target'].to(self.device)
                            
                            outputs = self.model(inputs)
                            loss = self.criterion(outputs, targets)
                            val_loss += loss.item()
                    
                    avg_val_loss = val_loss / len(self.val_loader)
                    val_losses.append(avg_val_loss)
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss
                    }, step=epoch)
                    
                    logging.info(
                        "Epoch %d/%d - Train loss: %.4f, Val loss: %.4f",
                        epoch+1, self.training_epochs, avg_train_loss, avg_val_loss
                    )
                    
                    # Save best model
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        # Save model checkpoint
                        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
                            torch.save(self.model.state_dict(), tmp.name)
                            self.best_model_path = tmp.name
                        
                        logging.info("Saved best model with val_loss: %.4f", best_val_loss)
                        mlflow.log_artifact(self.best_model_path, "model")
            
            # Plot and log training history for both TF and PyTorch
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Save plot to temp file and log as artifact
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                mlflow.log_artifact(tmp.name, "plots")
        
        # Load best model for evaluation
        if self.is_tensorflow:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.best_model_path)
        else:
            self.model.load_state_dict(torch.load(self.best_model_path))
        
        # Continue to evaluation
        self.next(self.evaluate_model)
    
    @card
    @resources(memory=8192, gpu=1)
    @step
    def evaluate_model(self):
        """Evaluate the trained model using the FuseMyCell metrics."""
        import mlflow
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # First, evaluate baselines
        baseline_metrics = self.evaluate_baseline_models()
        # Initialize metrics
        n_ssim_values = []
        file_paths = []
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            logging.info("Evaluating model with FuseMyCell metrics")
            
            if self.is_tensorflow:
                # TensorFlow doesn't need to set eval mode explicitly
                import tensorflow as tf
                import numpy as np
                
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    inputs = batch['input']
                    targets = batch['target']
                    
                    if 'file_path' in batch:
                        file_paths.extend(batch['file_path'])
                    
                    # Process in batches
                    for i in range(inputs.shape[0]):
                        # Convert from PyTorch to TensorFlow format
                        input_np = inputs[i].numpy().transpose(1, 2, 3, 0)  # (C,D,H,W) -> (D,H,W,C)
                        target_np = targets[i].numpy().transpose(1, 2, 3, 0)
                        
                        # Add batch dimension
                        input_tf = tf.convert_to_tensor(input_np)[tf.newaxis, ...]
                        
                        # Generate prediction
                        output_tf = self.model(input_tf, training=False)
                        
                        # Convert back to numpy
                        output_np = output_tf[0].numpy()  # Remove batch dimension
                        
                        # Calculate normalized SSIM
                        n_ssim = compute_n_ssim(output_np, target_np, input_np)
                        n_ssim_values.append(n_ssim)
                        
                        # Log individual sample metrics
                        sample_name = os.path.basename(file_paths[-1]) if file_paths else f"sample_{i}"
                        mlflow.log_metrics({
                            f"n_ssim_{sample_name}": n_ssim
                        })
            else:
                # PyTorch evaluation
                import torch
                
                # Set model to evaluation mode
                self.model.eval()
                
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc="Evaluation"):
                        inputs = batch['input'].to(self.device)
                        targets = batch['target'].to(self.device)
                        if 'file_path' in batch:
                            file_paths.extend(batch['file_path'])
                        
                        # Generate predictions
                        outputs = self.model(inputs)
                        
                        # Calculate metrics for each sample in the batch
                        for i in range(inputs.shape[0]):
                            # Convert tensors to numpy for metric calculation
                            input_np = inputs[i, 0].cpu().numpy()  # Remove channel dimension
                            target_np = targets[i, 0].cpu().numpy()
                            output_np = outputs[i, 0].cpu().numpy()
                            
                            # Calculate normalized SSIM
                            n_ssim = compute_n_ssim(output_np, target_np, input_np)
                            n_ssim_values.append(n_ssim)
                            
                            # Log individual sample metrics
                            sample_name = os.path.basename(file_paths[-1]) if file_paths else f"sample_{i}"
                            mlflow.log_metrics({
                                f"n_ssim_{sample_name}": n_ssim
                            })
            
            # Calculate and log overall metrics
            avg_n_ssim = np.mean(n_ssim_values)
            self.average_n_ssim = avg_n_ssim
            
            # Calculate improvements over baselines
            improvements = {}
            for baseline_name, baseline_value in baseline_metrics.items():
                if baseline_name.startswith('avg_'):
                    metric_name = baseline_name.replace('avg_', '')
                    improvement = avg_n_ssim - baseline_value
                    improvements[f'improvement_over_{metric_name}'] = improvement
                    mlflow.log_metric(f'improvement_over_{metric_name}', improvement)
                    logging.info(f"Improvement over {metric_name}: {improvement:.4f}")

            mlflow.log_metrics({
                "average_n_ssim": avg_n_ssim,
                "n_ssim_std": np.std(n_ssim_values)
            })
            
            logging.info("Average N-SSIM: %.4f", avg_n_ssim)
        
            # Plot distribution of N-SSIM values compared to baselines
            plt.figure(figsize=(12, 8))
            
            # Plot model N-SSIM distribution
            plt.hist(n_ssim_values, bins=20, alpha=0.7, label='Model')
            
            # Plot vertical lines for baselines
            for baseline_name, baseline_value in baseline_metrics.items():
                if baseline_name.startswith('avg_'):
                    plt.axvline(baseline_value, color='r' if 'input_as_output' in baseline_name else 'g', 
                            linestyle='dashed', linewidth=2, 
                            label=f'Baseline: {baseline_name.replace("avg_", "").replace("_n_ssim", "")}')
            
            # Plot model average
            plt.axvline(avg_n_ssim, color='b', linestyle='dashed', linewidth=2, 
                    label=f'Model average: {avg_n_ssim:.4f}')
            
            plt.xlabel('Normalized SSIM')
            plt.ylabel('Count')
            plt.title('Distribution of N-SSIM Values vs. Baselines')
            plt.legend()
            
            # Save plot to temp file and log as artifact
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                mlflow.log_artifact(tmp.name, "evaluation_plots")
            
            # Generate and log sample visualization
            self._visualize_samples(inputs, outputs, targets)
            
            # Evaluate patch-based inference if enabled
            if self.use_patch_inference:
                self._evaluate_patch_inference()
        
        # Proceed to model registration
        self.next(self.register_model)
    
    def evaluate_baseline_models(self):
        """Evaluate baseline models for comparison with our trained model."""
        import mlflow
        from tqdm import tqdm
        import numpy as np
        from scipy import ndimage
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Initialize metrics for baselines
        baseline_metrics = {
            'input_as_output': [],  # Zero-rule baseline
            'gaussian_blur': [],    # Simple enhancement baseline
            'average_views': [],    # Average views baseline
        }
        
        logging.info("Evaluating baseline models...")
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            # Run baselines on validation data
            for batch in tqdm(self.val_loader, desc="Baseline Evaluation"):
                if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
                    inputs = batch['input']
                    targets = batch['target']
                    
                    # Process each sample in the batch
                    for i in range(len(inputs)):
                        if self.is_tensorflow:
                            # For TensorFlow, convert from PyTorch format
                            input_np = inputs[i].numpy().transpose(1, 2, 3, 0)  # (C,D,H,W) -> (D,H,W,C)
                            target_np = targets[i].numpy().transpose(1, 2, 3, 0)
                            
                            # Handle channel dimension if present
                            if input_np.shape[-1] == 1:
                                input_np = input_np[..., 0]
                            if target_np.shape[-1] == 1:
                                target_np = target_np[..., 0]
                        else:
                            # For PyTorch
                            input_np = inputs[i, 0].cpu().numpy()  # Remove channel dimension
                            target_np = targets[i, 0].cpu().numpy()
                        
                        # Baseline 1: Input-as-Output (should always give N_SSIM = 0)
                        n_ssim_input = compute_n_ssim(input_np, target_np, input_np)
                        baseline_metrics['input_as_output'].append(n_ssim_input)
                        
                        # Baseline 2: Simple gaussian blur enhancement
                        enhanced_np = self._enhance_image_baseline(input_np)
                        n_ssim_enhanced = compute_n_ssim(enhanced_np, target_np, input_np)
                        baseline_metrics['gaussian_blur'].append(n_ssim_enhanced)
                        
                        # Baseline 3: Average views simulation
                        averaged_np = self._average_views_baseline(input_np)
                        n_ssim_averaged = compute_n_ssim(averaged_np, target_np, input_np)
                        baseline_metrics['average_views'].append(n_ssim_averaged)
            
            # Calculate average metrics for baselines
            avg_metrics = {}
            for baseline_name, values in baseline_metrics.items():
                avg_value = np.mean(values)
                avg_metrics[f'avg_{baseline_name}_n_ssim'] = avg_value
                
                # Log to MLflow
                mlflow.log_metric(f'baseline_{baseline_name}_n_ssim', avg_value)
                logging.info(f"Baseline {baseline_name} N-SSIM: {avg_value:.4f}")
            
            # Return metrics for comparison
            return avg_metrics

    def _enhance_image_baseline(self, image):
        """Simple image enhancement as baseline."""
        from scipy import ndimage
        
        # Apply simple gaussian blur for denoising
        enhanced = ndimage.gaussian_filter(image, sigma=1)
        
        # Simple contrast enhancement
        p2, p98 = np.percentile(enhanced, (2, 98))
        if p98 > p2:
            enhanced = (enhanced - p2) / (p98 - p2)
            enhanced = np.clip(enhanced, 0, 1)
        
        return enhanced
    
    def _average_views_baseline(self, input_image):
        """
        Simulate a fusion effect by duplicating and averaging the input image 
        with a slightly shifted version to mimic having another view.
        """
        from scipy import ndimage
        
        # Create a simulated second view by shifting slightly
        shift_amount = 2  # pixels
        shifted_view = ndimage.shift(input_image, (0, shift_amount, shift_amount), mode='nearest')
        
        # Simple average of the two views
        fused_view = (input_image + shifted_view) / 2.0
        
        return fused_view
    
    def _evaluate_patch_inference(self):
        """Evaluate patch-based inference on select validation samples."""
        import mlflow
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        import torch
        logging.info("Evaluating patch-based inference...")
        
        # Select a few samples from validation set
        max_samples = 2  # Limit the number of samples to evaluate
        sample_count = 0
        patch_n_ssim_values = []
        
        for batch in self.val_loader:
            for i in range(min(len(batch['input']), max_samples - sample_count)):
                # Get the full volume
                if self.is_tensorflow:
                    # For TensorFlow models
                    input_vol = batch['input'][i].numpy().transpose(1, 2, 3, 0)  # (C,D,H,W) -> (D,H,W,C)
                    target_vol = batch['target'][i].numpy().transpose(1, 2, 3, 0)
                else:
                    # For PyTorch models
                    input_vol = batch['input'][i, 0].cpu().numpy()  # [Z, Y, X]
                    target_vol = batch['target'][i, 0].cpu().numpy()
                
                file_path = batch['file_path'][i] if 'file_path' in batch else f"sample_{i}"
                
                patch_size = self.get_patch_size()
                patch_overlap = self.get_patch_overlap()
                
                # Extract patches
                patches = self.extract_patches_from_volume(input_vol, patch_size, patch_overlap)
                logging.info(f"Extracted {len(patches)} patches from volume of shape {input_vol.shape}")
                
                # Process each patch
                output_patches = []
                for patch, coords in patches:
                    # Normalize patch for model input
                    norm_patch = percentile_normalization(patch)
                    
                    if self.is_tensorflow:
                        # For TensorFlow models
                        # Add channel dimension if not present
                        if len(norm_patch.shape) == 3:
                            norm_patch = np.expand_dims(norm_patch, axis=-1)
                        
                        # Add batch dimension
                        patch_tensor = np.expand_dims(norm_patch, axis=0)
                        
                        # Generate prediction
                        output_patch = self.model(patch_tensor, training=False)
                        
                        # Convert to numpy and remove batch and channel dimensions
                        output_np = output_patch[0].numpy()
                        if output_np.shape[-1] == 1:  # If last dimension is channel
                            output_np = output_np[..., 0]
                    else:
                        # For PyTorch models
                        # Convert to tensor and add batch & channel dimensions
                        patch_tensor = torch.from_numpy(norm_patch).float().unsqueeze(0).unsqueeze(0)
                        patch_tensor = patch_tensor.to(self.device)
                        
                        # Generate prediction
                        with torch.no_grad():
                            output_patch = self.model(patch_tensor)
                        
                        # Save output patch and coordinates
                        output_np = output_patch[0, 0].cpu().numpy()
                    
                    # Save output patch and coordinates
                    output_patches.append((output_np, coords))
                
                # Stitch patches back together
                stitched_output = self.stitch_patches_to_volume(
                    output_patches, input_vol.shape, patch_size, patch_overlap
                )
                
                # Calculate N-SSIM for stitched output
                patch_n_ssim = compute_n_ssim(stitched_output, target_vol, input_vol)
                patch_n_ssim_values.append(patch_n_ssim)
                
                logging.info(f"Patch-based N-SSIM for sample {os.path.basename(file_path)}: {patch_n_ssim:.4f}")
                
                # Visualize stitched output vs full-volume inference
                # Run full-volume inference for comparison
                if self.is_tensorflow:
                    # For TensorFlow
                    # Add channel dimension if not present
                    if len(input_vol.shape) == 3:
                        input_tensor = np.expand_dims(input_vol, axis=-1)
                    else:
                        input_tensor = input_vol
                        
                    # Add batch dimension
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    
                    # Generate prediction
                    full_output_tensor = self.model(input_tensor, training=False)
                    
                    # Remove batch and channel dimensions
                    full_output = full_output_tensor[0].numpy()
                    if full_output.shape[-1] == 1:  # If last dimension is channel
                        full_output = full_output[..., 0]
                else:
                    # For PyTorch
                    with torch.no_grad():
                        input_tensor = torch.from_numpy(input_vol).float().unsqueeze(0).unsqueeze(0).to(self.device)
                        output_tensor = self.model(input_tensor)
                        full_output = output_tensor[0, 0].cpu().numpy()
                
                # Calculate N-SSIM for full-volume output
                full_n_ssim = compute_n_ssim(full_output, target_vol, input_vol)
                
                # Visualize middle slices
                z_mid = input_vol.shape[0] // 2
                
                # For TensorFlow models, handle possible channel dimension
                if self.is_tensorflow and len(input_vol.shape) == 4 and input_vol.shape[-1] == 1:
                    input_slice = input_vol[z_mid, :, :, 0]
                    stitched_slice = stitched_output[z_mid, :, :, 0]
                    full_output_slice = full_output[z_mid, :, :, 0]
                    target_slice = target_vol[z_mid, :, :, 0]
                else:
                    input_slice = input_vol[z_mid]
                    stitched_slice = stitched_output[z_mid]
                    full_output_slice = full_output[z_mid]
                    target_slice = target_vol[z_mid]
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                axes[0].imshow(input_slice, cmap='gray')
                axes[0].set_title('Input (Single View)')
                
                axes[1].imshow(stitched_slice, cmap='gray')
                axes[1].set_title(f'Patch-based Output (N-SSIM: {patch_n_ssim:.4f})')
                
                axes[2].imshow(full_output_slice, cmap='gray')
                axes[2].set_title(f'Full-volume Output (N-SSIM: {full_n_ssim:.4f})')
                
                axes[3].imshow(target_slice, cmap='gray')
                axes[3].set_title('Ground Truth')
                
                for ax in axes:
                    ax.axis('off')
                
                plt.tight_layout()
                
                # Save figure
                with tempfile.NamedTemporaryFile(suffix=f'_patch_inference_{sample_count}.png', delete=False) as tmp:
                    plt.savefig(tmp.name)
                    mlflow.log_artifact(tmp.name, "patch_inference")
                plt.close(fig)
                
                sample_count += 1
                if sample_count >= max_samples:
                    break
            
            if sample_count >= max_samples:
                break
        
        # Log patch-based inference metrics
        if patch_n_ssim_values:
            avg_patch_n_ssim = np.mean(patch_n_ssim_values)
            mlflow.log_metrics({
                "average_patch_n_ssim": avg_patch_n_ssim
            })
            logging.info(f"Average Patch-based N-SSIM: {avg_patch_n_ssim:.4f}")
    
    def _visualize_samples(self, inputs, outputs, targets, num_samples=3):
        """Visualize sample predictions and log them to MLflow."""
        import mlflow
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Select a few samples for visualization
        if isinstance(inputs, list) and len(inputs) > 0:
            num_available = len(inputs)
        else:
            num_available = inputs.shape[0] if hasattr(inputs, 'shape') else 0
        
        indices = np.random.choice(num_available, min(num_samples, num_available), replace=False)
        
        for idx in indices:
            if self.is_tensorflow:
                # For TensorFlow models
                # Convert from PyTorch tensor format if needed
                if hasattr(inputs, 'numpy'):
                    input_vol = inputs[idx].numpy().transpose(1, 2, 3, 0)  # (C,D,H,W) -> (D,H,W,C)
                    target_vol = targets[idx].numpy().transpose(1, 2, 3, 0)
                    
                    # Make prediction for this single sample
                    input_tf = np.expand_dims(input_vol, axis=0)  # Add batch dimension
                    output_tf = self.model(input_tf, training=False)
                    output_vol = output_tf[0].numpy()  # Remove batch dimension
                else:
                    # Assumes inputs/outputs are already in the right format
                    input_vol = inputs[idx]
                    output_vol = outputs[idx] 
                    target_vol = targets[idx]
                
                # Select middle slice for visualization (remove channel dimension if present)
                if input_vol.shape[-1] == 1:  # Check if last dimension is channel
                    z_mid = input_vol.shape[0] // 2
                    input_slice = input_vol[z_mid, :, :, 0]
                    output_slice = output_vol[z_mid, :, :, 0]
                    target_slice = target_vol[z_mid, :, :, 0]
                else:
                    z_mid = input_vol.shape[0] // 2
                    input_slice = input_vol[z_mid]
                    output_slice = output_vol[z_mid]
                    target_slice = target_vol[z_mid]
            else:
                # For PyTorch models
                input_vol = inputs[idx, 0].cpu().numpy()  # Remove channel dimension
                output_vol = outputs[idx, 0].cpu().numpy()
                target_vol = targets[idx, 0].cpu().numpy()
                
                # Select middle slice for visualization
                z_mid = input_vol.shape[0] // 2
                input_slice = input_vol[z_mid]
                output_slice = output_vol[z_mid]
                target_slice = target_vol[z_mid]
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot input slice
            im0 = axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title('Input (Single View)')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot prediction slice
            im1 = axes[1].imshow(output_slice, cmap='gray')
            axes[1].set_title('Prediction (Fused)')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot target slice
            im2 = axes[2].imshow(target_slice, cmap='gray')
            axes[2].set_title('Ground Truth (Fused)')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save plot to temp file and log as artifact
            with tempfile.NamedTemporaryFile(suffix=f'_sample_{idx}.png', delete=False) as tmp:
                plt.savefig(tmp.name)
                mlflow.log_artifact(tmp.name, "sample_visualizations")
            
            plt.close(fig)
    
    @card
    @step
    def register_model(self):
        """Register the model in MLflow Model Registry."""
        import mlflow
        import torch
        import shutil
        from pathlib import Path

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Only register if the model meets the quality threshold
        if self.average_n_ssim >= self.n_ssim_threshold:
            self.registered = True
            logging.info("Registering model with average N-SSIM: %.4f", self.average_n_ssim)
            
            # Follow the approach suggested for model registration
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                # Prepare artifacts
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                # Create inference code paths
                root = Path(__file__).parent
                
                # Define paths to existing files
                model_py_path = root / "model.py"
                backend_py_path = root / "backend.py"
                unet3d_py_path = root / "unet3d.py"
                
                # Create the inference directory structure
                inference_dir = root / "inference"
                inference_dir.mkdir(exist_ok=True)
                
                # Copy the existing files if they exist, otherwise create them
                if model_py_path.exists():
                    shutil.copy(model_py_path, inference_dir / "model.py")
                else:
                    self._create_model_file(inference_dir / "model.py")
                
                if backend_py_path.exists():
                    shutil.copy(backend_py_path, inference_dir / "backend.py")
                else:
                    self._create_backend_file(inference_dir / "backend.py")
                
                if unet3d_py_path.exists():
                    shutil.copy(unet3d_py_path, inference_dir / "unet3d.py")
                else:
                    self._create_unet3d_file(inference_dir / "unet3d.py")
                
                # Define code paths for MLflow
                self.code_paths = [
                    (inference_dir / "model.py").as_posix(),
                    (inference_dir / "unet3d.py").as_posix(),
                    (inference_dir / "backend.py").as_posix()
                ]

                # Register the model with MLflow
                mlflow.pyfunc.log_model(
                    python_model=inference_dir / "model.py",
                    artifact_path="fusemycell-model",
                    registered_model_name="fusemycell-singleview-to-multiview",
                    code_paths=self.code_paths,
                    artifacts=self.artifacts,
                    pip_requirements=self.pip_requirements
                )
                
                logging.info("Model registered successfully")
        else:
            self.registered = False
            logging.info(
                "Model not registered. Average N-SSIM (%.4f) below threshold (%.4f)",
                self.average_n_ssim,
                self.n_ssim_threshold
            )
        
        # Proceed to end
        self.next(self.end)

    def _get_model_artifacts(self, directory):
        """Return the list of artifacts that will be included with model.

        The model must preprocess the raw input data before making a prediction, so we
        need to include any required preprocessing components as artifacts.
        """
        import torch
        from pathlib import Path
        import json
        
        # Create a model directory for the artifacts
        model_dir = Path(directory) / "model_dir"
        model_dir.mkdir(exist_ok=True)
        
        artifacts_dir = model_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Save the PyTorch model inside the artifacts directory
        model_path = (artifacts_dir / "model.pth").as_posix()
        torch.save(self.model.state_dict(), model_path)
        
        # Create a code directory inside the model directory
        code_dir = model_dir / "code"
        code_dir.mkdir(exist_ok=True)
        
        # Save patch extraction configuration
        patch_config = {
            "patch_size": self.get_patch_size(),
            "patch_overlap": self.get_patch_overlap(),
            "percentile_min": 2,
            "percentile_max": 98
        }
        
        # Save model configuration
        model_config = {
            'in_channels': 7 if self.use_gradient_features else 1,
            'use_gradient_features': self.use_gradient_features,
            'use_physics': self.use_physics,
            'patch_size': self.get_patch_size()
        }
        
        config_path = (artifacts_dir / "model_config.json").as_posix()
        with open(config_path, 'w') as f:
            json.dump(model_config, f)
            
        # Save the patch config as JSON
        patch_config_path = (artifacts_dir / "patch_config.json").as_posix()
        with open(patch_config_path, 'w') as f:
            json.dump(patch_config, f)
        
        # Return the artifacts dictionary
        return {
            "model_dir": str(model_dir),
            "code": str(code_dir),
            "patch_config": patch_config_path
        }

    def _get_model_pip_requirements(self):
        """Return the list of required packages to run the model in production."""
        # Define the required packages with versions
        return [
            # "torch>=2.0.0",
            "numpy>=1.20.0",
            "tifffile>=2024.2.12",
            "scikit-image>=0.22.0",
            "pydantic>=2.0.0",
        ]

    @step
    def end(self):
        """End the pipeline and print summary."""
        if hasattr(self, 'registered') and self.registered:
            logging.info("Pipeline completed successfully. Model registered with MLflow.")
        else:
            logging.info("Pipeline completed. Model not registered.")
        
        logging.info("Check MLflow UI for experiment details: %s", self.mlflow_tracking_uri)
        logging.info("Run ID: %s", self.mlflow_run_id)


if __name__ == "__main__":
    FuseMyCellTraining()
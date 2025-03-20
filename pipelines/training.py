import logging
import os
from pathlib import Path
import numpy as np
import tempfile

from common import (
    PYTHON,
    DatasetMixin,
    build_unet3d_model,
    percentile_normalization,
    compute_n_ssim,
    configure_logging,
    load_image,
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
        "conda-forge::pytorch": "2.6.0",
        "conda-forge::torchvision": "0.21",  # Latest version
        # "conda-forge::torchaudio": "2.6.0",  # Latest version
        "conda-forge::pandas": "",
        "conda-forge::numpy": "",
        "conda-forge::mlflow": "",
        "conda-forge::tifffile": "",
        "conda-forge::scikit-image": "",
        "conda-forge::matplotlib": "",
        "conda-forge::boto3": "",
        "conda-forge::metaflow": "",
        "conda-forge::cellpose": "", 
    }
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
        default=50,
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

    @card
    @step
    def start(self):
        """Start and prepare the FuseMyCell training pipeline."""
        import mlflow

        # Try to set the MLflow tracking URI, but don't fail if it doesn't work
        try:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)
        except Exception as e:
            logging.warning(f"Could not set MLflow tracking URI: {e}")

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        # Load dataset (with error handling)
        try:
            data_loaders = self.load_dataset()
            self.train_loader = data_loaders['train']
            self.val_loader = data_loaders['val']
        except Exception as e:
            # Create dummy loaders for testing
            logging.warning(f"Failed to load dataset: {e}")
            import torch
            from torch.utils.data import DataLoader, TensorDataset
            dummy_data = torch.randn(1, 1, 64, 128, 128)
            dummy_target = torch.randn(1, 1, 64, 128, 128)
            dummy_dataset = TensorDataset(dummy_data, dummy_target)
            self.train_loader = DataLoader(dummy_dataset, batch_size=1)
            self.val_loader = DataLoader(dummy_dataset, batch_size=1)

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
    @resources(memory=8192, gpu=1)
    @environment(vars={"CUDA_VISIBLE_DEVICES": "0"})
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
        self.model = build_unet3d_model(
            input_shape=(in_channels, *patch_size),  # Update input shape
            use_physics=self.use_physics
        )
        
        # Configure training with proper device detection
        # First check for Apple Silicon MPS (Metal Performance Shaders)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            logging.info("Using Apple Silicon GPU via MPS backend")
        # Then check for CUDA
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info("Using NVIDIA GPU via CUDA")
        # Fall back to CPU
        else:
            self.device = torch.device("cpu")
            logging.info("Using CPU (no GPU available)")
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # Continue to training
        self.next(self.train_model)
        
    @card
    @resources(memory=8192, gpu=1)
    @environment(vars={"CUDA_VISIBLE_DEVICES": "0", "PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    @step
    def train_model(self):
        """Train the 3D U-Net model."""
        import mlflow
        import torch
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            logging.info(f"Starting training for {self.training_epochs} epochs")
            
            for epoch in range(self.training_epochs):
                # Training phase
                self.model.train()
                epoch_loss = 0
                
                with tqdm(self.train_loader, unit="batch") as train_pbar:
                    train_pbar.set_description(f"Epoch {epoch+1}/{self.training_epochs}")
                    
                    for batch in train_pbar:
                        # Handle different batch formats (dict or list)
                        if isinstance(batch, dict) and 'input' in batch and 'target' in batch:
                            # Dictionary format from your dataset
                            # Add gradient features if enabled
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
                    
                    # Log metrics
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
            
            # Plot and log training history
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
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # Continue to evaluation
        self.next(self.evaluate_model)
    
    @card
    @resources(memory=8192, gpu=1)
    @step
    def evaluate_model(self):
        """Evaluate the trained model using the FuseMyCell metrics."""
        import mlflow
        import torch
        from tqdm import tqdm
        import matplotlib.pyplot as plt
        
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize metrics
        n_ssim_values = []
        file_paths = []
        
        with mlflow.start_run(run_id=self.mlflow_run_id):
            logging.info("Evaluating model with FuseMyCell metrics")
            
            with torch.no_grad():
                for batch in tqdm(self.val_loader, desc="Evaluation"):
                    inputs = batch['input'].to(self.device)
                    targets = batch['target'].to(self.device)
                    file_paths.extend(batch['file_path'])
                    
                    # Generate predictions (whole volume or patch-based)
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
                        sample_name = os.path.basename(file_paths[-1])
                        mlflow.log_metrics({
                            f"n_ssim_{sample_name}": n_ssim
                        })
            
            # Calculate and log overall metrics
            avg_n_ssim = np.mean(n_ssim_values)
            self.average_n_ssim = avg_n_ssim
            mlflow.log_metrics({
                "average_n_ssim": avg_n_ssim,
                "n_ssim_std": np.std(n_ssim_values)
            })
            
            logging.info("Average N-SSIM: %.4f", avg_n_ssim)
            
            # Plot distribution of N-SSIM values
            plt.figure(figsize=(10, 6))
            plt.hist(n_ssim_values, bins=20, alpha=0.7)
            plt.axvline(avg_n_ssim, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_n_ssim:.4f}')
            plt.xlabel('Normalized SSIM')
            plt.ylabel('Count')
            plt.title('Distribution of N-SSIM Values')
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
    
    def _evaluate_patch_inference(self):
        """Evaluate patch-based inference on select validation samples."""
        import mlflow
        import torch
        import tifffile
        import matplotlib.pyplot as plt
        
        logging.info("Evaluating patch-based inference...")
        
        # Select a few samples from validation set
        max_samples = 2  # Limit the number of samples to evaluate
        sample_count = 0
        patch_n_ssim_values = []
        
        for batch in self.val_loader:
            for i in range(min(len(batch['input']), max_samples - sample_count)):
                # Get the full volume
                input_vol = batch['input'][i, 0].cpu().numpy()  # [Z, Y, X]
                target_vol = batch['target'][i, 0].cpu().numpy()
                file_path = batch['file_path'][i]
                
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
                    
                    # Convert to tensor and add batch & channel dimensions
                    patch_tensor = torch.from_numpy(norm_patch).float().unsqueeze(0).unsqueeze(0)
                    patch_tensor = patch_tensor.to(self.device)
                    
                    # Generate prediction
                    with torch.no_grad():
                        output_patch = self.model(patch_tensor)
                    
                    # Save output patch and coordinates
                    output_patches.append((output_patch[0, 0].cpu().numpy(), coords))
                
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
                with torch.no_grad():
                    input_tensor = torch.from_numpy(input_vol).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    output_tensor = self.model(input_tensor)
                    full_output = output_tensor[0, 0].cpu().numpy()
                
                # Calculate N-SSIM for full-volume output
                full_n_ssim = compute_n_ssim(full_output, target_vol, input_vol)
                
                # Visualize middle slices
                z_mid = input_vol.shape[0] // 2
                
                fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                
                axes[0].imshow(input_vol[z_mid], cmap='gray')
                axes[0].set_title('Input (Single View)')
                
                axes[1].imshow(stitched_output[z_mid], cmap='gray')
                axes[1].set_title(f'Patch-based Output (N-SSIM: {patch_n_ssim:.4f})')
                
                axes[2].imshow(full_output[z_mid], cmap='gray')
                axes[2].set_title(f'Full-volume Output (N-SSIM: {full_n_ssim:.4f})')
                
                axes[3].imshow(target_vol[z_mid], cmap='gray')
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
        
        # Select a few samples for visualization
        indices = np.random.choice(inputs.shape[0], min(num_samples, inputs.shape[0]), replace=False)
        
        for idx in indices:
            input_vol = inputs[idx, 0].cpu().numpy()
            output_vol = outputs[idx, 0].cpu().numpy()
            target_vol = targets[idx, 0].cpu().numpy()
            
            # Select middle slice for visualization
            z_mid = input_vol.shape[0] // 2
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot input slice
            im0 = axes[0].imshow(input_vol[z_mid], cmap='gray')
            axes[0].set_title('Input (Single View)')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            # Plot prediction slice
            im1 = axes[1].imshow(output_vol[z_mid], cmap='gray')
            axes[1].set_title('Prediction (Fused)')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            # Plot target slice
            im2 = axes[2].imshow(target_vol[z_mid], cmap='gray')
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
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
    packages,
    load_image,
)
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


@project(name="fusemycell")
@conda_base(
    python=PYTHON,
    packages=packages(
        "pandas",
        "numpy",
        "mlflow",
        "torch",
        "tifffile",
        "scikit-image",
        "cellpose",
        "matplotlib",
        "boto3",
        "metaflow",
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

    patch_size = Parameter(
        "patch-size",
        help="Size of 3D patches for training (z,y,x)",
        default="64,128,128",
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

    def get_patch_size(self):
        """Convert patch size parameter to tuple."""
        return tuple(map(int, self.patch_size.split(',')))

    @card
    @step
    def start(self):
        """Start and prepare the FuseMyCell training pipeline."""
        import mlflow

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        logging.info("MLflow tracking server: %s", self.mlflow_tracking_uri)

        self.mode = "production" if current.is_production else "development"
        logging.info("Running flow in %s mode.", self.mode)

        # Load dataset
        dataset_files = self.load_dataset()
        logging.info("Loaded dataset with %d single-view files and %d fused-view files",
                    len(dataset_files["single_view"]), len(dataset_files["fused_view"]))
        
        # Create train/val splits
        self.dataset_splits = self.prepare_dataset_splits(
            dataset_files["single_view"],
            dataset_files["fused_view"],
            train_ratio=0.8
        )
        
        try:
            # Start a new MLflow run
            run = mlflow.start_run(run_name=current.run_id)
            self.mlflow_run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params({
                "training_epochs": self.training_epochs,
                "training_batch_size": self.training_batch_size,
                "learning_rate": self.learning_rate,
                "patch_size": self.patch_size,
                "use_physics": self.use_physics,
            })
            
        except Exception as e:
            message = f"Failed to connect to MLflow server {self.mlflow_tracking_uri}."
            raise RuntimeError(message) from e
        
        # Continue to data preparation
        self.next(self.prepare_data)
    
    @card
    @step
    def prepare_data(self):
        """Prepare and preprocess the data for training."""
        import torch
        from torch.utils.data import Dataset, DataLoader
        
        class LightSheetDataset(Dataset):
            def __init__(self, single_view_files, fused_view_files, patch_size):
                self.single_view_files = single_view_files
                self.fused_view_files = fused_view_files
                self.patch_size = patch_size
            
            def __len__(self):
                return len(self.single_view_files)
            
            def __getitem__(self, idx):
                # Load files
                sv_path = self.single_view_files[idx]
                fv_path = self.fused_view_files[idx]
                
                # Load images
                single_view = load_image(sv_path)
                fused_view = load_image(fv_path)
                
                # Apply normalization
                single_view = percentile_normalization(single_view)
                fused_view = percentile_normalization(fused_view)
                
                # Extract a random patch if the images are larger than patch_size
                z, y, x = single_view.shape
                pz, py, px = self.patch_size
                
                if z > pz and y > py and x > px:
                    # Random starting coordinates
                    start_z = np.random.randint(0, z - pz)
                    start_y = np.random.randint(0, y - py)
                    start_x = np.random.randint(0, x - px)
                    
                    # Extract patches
                    single_view = single_view[start_z:start_z+pz, start_y:start_y+py, start_x:start_x+px]
                    fused_view = fused_view[start_z:start_z+pz, start_y:start_y+py, start_x:start_x+px]
                
                # Convert to torch tensors and add channel dimension
                single_view = torch.from_numpy(single_view).float().unsqueeze(0)
                fused_view = torch.from_numpy(fused_view).float().unsqueeze(0)
                
                return {
                    'input': single_view,
                    'target': fused_view,
                    'file_path': str(sv_path)
                }
        
        # Get patch size and create datasets
        patch_size = self.get_patch_size()
        logging.info("Using patch size: %s", patch_size)
        
        self.train_dataset = LightSheetDataset(
            self.dataset_splits['train']['single_view'],
            self.dataset_splits['train']['fused_view'],
            patch_size
        )
        
        self.val_dataset = LightSheetDataset(
            self.dataset_splits['val']['single_view'],
            self.dataset_splits['val']['fused_view'],
            patch_size
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        logging.info("Created data loaders with %d training batches and %d validation batches",
                    len(self.train_loader), len(self.val_loader))
        
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
        patch_size = self.get_patch_size()
        input_shape = (1, *patch_size)  # (C, Z, Y, X)
        
        # Build model
        self.model = build_unet3d_model(input_shape, use_physics=self.use_physics)
        
        # Configure training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info("Using device: %s", self.device)
        
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = torch.nn.MSELoss()
        
        # Continue to training
        self.next(self.train_model)
    
    @card
    @resources(memory=8192, gpu=1)
    @environment(vars={"CUDA_VISIBLE_DEVICES": "0"})
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
            logging.info("Starting training for %d epochs", self.training_epochs)
            
            for epoch in range(self.training_epochs):
                # Training phase
                self.model.train()
                epoch_loss = 0
                
                with tqdm(self.train_loader, unit="batch") as train_pbar:
                    train_pbar.set_description(f"Epoch {epoch+1}/{self.training_epochs}")
                    
                    for batch in train_pbar:
                        inputs = batch['input'].to(self.device)
                        targets = batch['target'].to(self.device)
                        
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
        
        # Proceed to model registration
        self.next(self.register_model)
    
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
        import joblib

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Only register if the model meets the quality threshold
        if self.average_n_ssim >= self.n_ssim_threshold:
            self.registered = True
            logging.info("Registering model with average N-SSIM: %.4f", self.average_n_ssim)
            
            # We'll use the approach you suggested for model registration
            with (
                mlflow.start_run(run_id=self.mlflow_run_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                self.artifacts = self._get_model_artifacts(directory)
                self.pip_requirements = self._get_model_pip_requirements()

                # Create inference code paths
                root = Path(__file__).parent
                inference_dir = root / "inference"
                inference_dir.mkdir(exist_ok=True)
                
                # Create backend file for inference
                self._create_inference_backend(inference_dir)
                self.code_paths = [(inference_dir / "backend.py").as_posix()]

                # Register the model with MLflow
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
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


if __name__ == "__main__":
    FuseMyCellTraining()
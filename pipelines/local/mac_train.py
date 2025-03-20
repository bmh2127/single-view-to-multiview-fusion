#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import json
import tifffile
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import time
import random
import psutil
import threading

# Import from your original code
from pipelines.common import (
    build_unet3d_model,
    compute_n_ssim,
    apply_gradient_augmentations
)


class PatchDataset(Dataset):
    """Dataset for loading pre-extracted patches for FuseMyCell training."""
    
    def __init__(self, metadata_file, transform=None, apply_augmentations=True, cache_size=0, float32=False):
        """
        Initialize the dataset from a metadata file.
        
        Args:
            metadata_file: Path to JSON metadata file containing patch information
            transform: Optional transform to apply to the images
            apply_augmentations: Whether to apply random augmentations
            cache_size: Number of patches to cache in memory (0 = no caching)
            float32: Convert patches to float32 when loading
        """
        self.base_dir = Path(metadata_file).parent
        self.transform = transform
        self.apply_augmentations = apply_augmentations
        self.cache_size = cache_size
        self.float32 = float32
        self.cache = {}  # Cache for loaded patches
        
        # Load metadata
        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)
        
        print(f"Loaded dataset with {len(self.metadata)} patch pairs")
        
        # Fill cache if requested
        if self.cache_size > 0:
            print(f"Pre-caching {min(self.cache_size, len(self.metadata))} samples...")
            for i in range(min(self.cache_size, len(self.metadata))):
                self._load_and_cache(i)
    
    def _load_and_cache(self, idx):
        """Load a patch pair and add to cache."""
        if idx in self.cache:
            return self.cache[idx]
            
        patch_info = self.metadata[idx]
        
        # Get file paths
        angle_path = self.base_dir / patch_info["angle_path"]
        fused_path = self.base_dir / patch_info["fused_path"]
        
        # Load patches
        try:
            angle_patch = tifffile.imread(angle_path)
            fused_patch = tifffile.imread(fused_path)
            
            # Ensure 3D format (handle 2D images)
            if len(angle_patch.shape) == 2:
                angle_patch = angle_patch[np.newaxis, ...]
            if len(fused_patch.shape) == 2:
                fused_patch = fused_patch[np.newaxis, ...]
            
            # Convert to float32 if requested
            if self.float32:
                # Normalize uint16 to [0, 1] range
                if angle_patch.dtype == np.uint16:
                    angle_patch = angle_patch.astype(np.float32) / 65535.0
                    fused_patch = fused_patch.astype(np.float32) / 65535.0
                else:
                    angle_patch = angle_patch.astype(np.float32)
                    fused_patch = fused_patch.astype(np.float32)
            
            # Convert to tensors
            angle_tensor = torch.from_numpy(angle_patch).float().unsqueeze(0)  # Add channel dimension
            fused_tensor = torch.from_numpy(fused_patch).float().unsqueeze(0)  # Add channel dimension
            
            result = {
                'input': angle_tensor,
                'target': fused_tensor,
                'metadata': patch_info
            }
            
            # Add to cache if using caching
            if self.cache_size > 0:
                self.cache[idx] = result
                
            return result
            
        except Exception as e:
            print(f"Error loading patch {patch_info['patch_id']}: {e}")
            # Return a different sample on error
            return self.__getitem__((idx + 1) % len(self))
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        result = self._load_and_cache(idx)
        
        # Apply augmentations if enabled
        if self.apply_augmentations:
            result['input'], result['target'] = apply_gradient_augmentations(
                result['input'], result['target'], apply_augs=True
            )
        
        # Apply transforms if specified
        if self.transform:
            result['input'] = self.transform(result['input'])
            result['target'] = self.transform(result['target'])
        
        return result

def configure_logging():
    """Configure logging for training."""
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/training_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    return log_file

def parse_args():
    parser = argparse.ArgumentParser(description="Train FuseMyCell model from pre-extracted patches")
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Directory containing the extracted patches and metadata")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--batch-size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--use-amp", action="store_true",
                        help="Use automatic mixed precision")
    parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation set ratio")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of data loading workers")
    parser.add_argument("--use-physics", action="store_true",
                        help="Use physics-informed neural network")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--accumulation-steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--cache-size", type=int, default=0,
                        help="Number of samples to cache in memory (0 = no caching)")
    parser.add_argument("--float32", action="store_true",
                        help="Convert patches to float32 when loading")
    parser.add_argument("--monitor-interval", type=int, default=0,
                        help="Memory monitoring interval in seconds (0 = disabled)")
    return parser.parse_args()

def visualize_predictions(model, val_loader, device, epoch, output_dir, max_samples=3):
    """Visualize model predictions on validation data."""
    model.eval()
    os.makedirs(output_dir / "visualizations", exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_samples:
                break
                
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            # Generate predictions
            outputs = model(inputs)
            
            # Take the first sample in the batch
            input_slice = inputs[0, 0, inputs.shape[2]//2].cpu().numpy()
            target_slice = targets[0, 0, targets.shape[2]//2].cpu().numpy()
            output_slice = outputs[0, 0, outputs.shape[2]//2].cpu().numpy()
            
            # Create figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot slices
            im0 = axes[0].imshow(input_slice, cmap='gray')
            axes[0].set_title('Input (Single View)')
            plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
            
            im1 = axes[1].imshow(output_slice, cmap='gray')
            axes[1].set_title('Prediction')
            plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
            
            im2 = axes[2].imshow(target_slice, cmap='gray')
            axes[2].set_title('Target (Fused View)')
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_dir / "visualizations" / f"epoch_{epoch}_sample_{i}.png")
            plt.close(fig)

def train_model():
    args = parse_args()
    log_file = configure_logging()
    
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Start memory monitor if requested
    memory_monitor = None
    if args.monitor_interval > 0:
        memory_monitor = MemoryMonitor(interval=args.monitor_interval)
        memory_monitor.start()
    
    # Save command line arguments
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    # Determine device - use MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using MPS (Metal Performance Shaders) acceleration")
    else:
        device = torch.device("cpu")
        logging.info("MPS not available, using CPU")
    
    # Load metadata and create dataset
    metadata_file = Path(args.data_dir) / "patch_metadata.json"
    if not metadata_file.exists():
        logging.error(f"Metadata file not found: {metadata_file}")
        return
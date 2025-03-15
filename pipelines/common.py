import os
import logging
import logging.config
import sys
import time
from io import StringIO
from pathlib import Path

import numpy as np
import tifffile
from metaflow import IncludeFile, Parameter, current, S3

PYTHON = "3.12.8"

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.2",
    "tensorflow": "2.17.0",
    "torch": "2.3.0",
    "tifffile": "2024.2.12",
    "scikit-image": "0.22.0",
    "cellpose": "2.2.4",
}


class DatasetMixin:
    """A mixin for loading and preparing light sheet microscopy datasets.

    This mixin is designed to be combined with any pipeline that requires accessing
    light sheet microscopy datasets for the FuseMyCell challenge.
    """

    dataset_dir = Parameter(
        "dataset-dir",
        help="Directory containing the light sheet microscopy dataset.",
        default="data/raw",
    )

    single_view_pattern = Parameter(
        "single-view-pattern",
        help="Pattern to match single view files.",
        default="**/view_0/**/*.tif",
    )

    fused_view_pattern = Parameter(
        "fused-view-pattern",
        help="Pattern to match fused view files.",
        default="**/fused/**/*.tif",
    )

    s3_bucket = Parameter(
        "s3-bucket",
        help="S3 bucket containing the dataset files (used in production mode).",
        default="",
    )

    def load_dataset(self):
        """Load and prepare the light sheet microscopy dataset.
        
        Returns:
            dict: Dictionary containing lists of single-view and fused-view file paths.
        """
        if current.is_production and self.s3_bucket:
            with S3(s3root=self.s3_bucket) as s3:
                files = s3.get_all()
                logging.info("Found %d file(s) in remote location", len(files))
                # In a real implementation, filter and process the files appropriately
                # For now, we'll just return a placeholder
                return {"single_view": [], "fused_view": []}
        else:
            logging.info("Running in development mode. Loading data from local filesystem.")
            dataset_path = Path(self.dataset_dir)
            
            # Find single-view and fused-view files
            single_view_files = sorted(list(dataset_path.glob(self.single_view_pattern)))
            fused_view_files = sorted(list(dataset_path.glob(self.fused_view_pattern)))
            
            logging.info("Found %d single-view files and %d fused-view files", 
                         len(single_view_files), len(fused_view_files))
            
            return {
                "single_view": single_view_files,
                "fused_view": fused_view_files,
            }

    def prepare_dataset_splits(self, single_view_files, fused_view_files, train_ratio=0.8):
        """Split the dataset into training and validation sets.
        
        Args:
            single_view_files: List of single-view file paths
            fused_view_files: List of fused-view file paths
            train_ratio: Ratio of training data (default: 0.8)
            
        Returns:
            dict: Dictionary containing training and validation splits
        """
        # We need to ensure that corresponding single and fused views stay together
        # For this example, we'll assume the files are already sorted and paired
        
        # Generate paired indices for the dataset
        n_samples = min(len(single_view_files), len(fused_view_files))
        indices = np.arange(n_samples)
        
        # Shuffle indices for random split
        seed = int(time.time() * 1000) if current.is_production else 42
        generator = np.random.default_rng(seed=seed)
        generator.shuffle(indices)
        
        # Split into training and validation sets
        n_train = int(n_samples * train_ratio)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Create dataset splits
        train_single_view = [single_view_files[i] for i in train_indices]
        train_fused_view = [fused_view_files[i] for i in train_indices]
        val_single_view = [single_view_files[i] for i in val_indices]
        val_fused_view = [fused_view_files[i] for i in val_indices]
        
        logging.info("Created dataset splits: %d training pairs, %d validation pairs",
                    len(train_single_view), len(val_single_view))
        
        return {
            "train": {"single_view": train_single_view, "fused_view": train_fused_view},
            "val": {"single_view": val_single_view, "fused_view": val_fused_view}
        }


def packages(*names: str):
    """Return a dictionary of the specified packages and their corresponding version.

    This function is useful to set up the different pipelines while keeping the
    package versions consistent and centralized in a single location.

    Any packages that should be locked to a specific version will be part of the
    `PACKAGES` dictionary. If a package is not present in the dictionary, it will be
    installed using the latest version available.
    """
    return {name: PACKAGES.get(name, "") for name in names}


def configure_logging():
    """Configure logging handlers and return a logger instance."""
    if Path("logging.conf").exists():
        logging.config.fileConfig("logging.conf")
    else:
        logging.basicConfig(
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
            level=logging.INFO,
        )


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    """
    Compute a percentile normalization for the given image, as specified by the FuseMyCell challenge.
    
    Parameters:
    - image (array): array (2D or 3D) of the image file.
    - pmin  (int or float): the minimal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - pmax  (int or float): the maximal percentage for the percentiles to compute.
                            Values must be between 0 and 100 inclusive.
    - axis : Axis or axes along which the percentiles are computed.
             The default (=None) is to compute it along a flattened version of the array.

    Returns:
    Normalized image (np.ndarray): An array containing the normalized image.
    """
    if not (np.isscalar(pmin) and np.isscalar(pmax) and 0 <= pmin < pmax <= 100):
        raise ValueError("Invalid values for pmin and pmax")

    low_percentile = np.percentile(image, pmin, axis=axis, keepdims=True)
    high_percentile = np.percentile(image, pmax, axis=axis, keepdims=True)

    if low_percentile == high_percentile:
        logging.warning(f"Same min {low_percentile} and high {high_percentile}, image may be empty")
        return image

    return (image - low_percentile) / (high_percentile - low_percentile)


def load_image(image_path):
    """Load a 3D TIFF image.
    
    Args:
        image_path: Path to the TIFF image
        
    Returns:
        numpy.ndarray: 3D image array
    """
    try:
        # Load the image
        image = tifffile.imread(str(image_path))
        
        # Handle different dimensions
        if len(image.shape) == 2:  # Single 2D image
            image = image[np.newaxis, ...]
        elif len(image.shape) == 4:  # Multiple channels
            # For simplicity, we'll just take the first channel
            image = image[..., 0]
            
        return image
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None


def build_unet3d_model(input_shape, use_physics=False):
    """Build a 3D U-Net model for single-view to multiview fusion.
    
    Args:
        input_shape: Shape of the input volume (z, y, x)
        use_physics: Whether to use physics-informed neural network
        
    Returns:
        model: PyTorch model for 3D image fusion
    """
    import torch
    import torch.nn as nn
    
    class DoubleConv3D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(DoubleConv3D, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        
        def forward(self, x):
            return self.conv(x)
    
    class Down3D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(Down3D, self).__init__()
            self.mpconv = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv3D(in_channels, out_channels)
            )
        
        def forward(self, x):
            return self.mpconv(x)
    
    class Up3D(nn.Module):
        def __init__(self, in_channels, out_channels, bilinear=False):
            super(Up3D, self).__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            else:
                self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            
            self.conv = DoubleConv3D(in_channels, out_channels)
        
        def forward(self, x1, x2):
            x1 = self.up(x1)
            
            # Pad x1 if needed for concatenation
            diffZ = x2.size()[2] - x1.size()[2]
            diffY = x2.size()[3] - x1.size()[3]
            diffX = x2.size()[4] - x1.size()[4]
            
            x1 = torch.nn.functional.pad(
                x1, (diffX // 2, diffX - diffX // 2,
                     diffY // 2, diffY - diffY // 2,
                     diffZ // 2, diffZ - diffZ // 2)
            )
            
            # Concatenate along the channel dimension
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
    
    class UNet3D(nn.Module):
        def __init__(self, in_channels=1, out_channels=1, init_features=64):
            super(UNet3D, self).__init__()
            
            # Initial convolution
            self.inc = DoubleConv3D(in_channels, init_features)
            
            # Downsampling path
            self.down1 = Down3D(init_features, init_features * 2)
            self.down2 = Down3D(init_features * 2, init_features * 4)
            self.down3 = Down3D(init_features * 4, init_features * 8)
            self.down4 = Down3D(init_features * 8, init_features * 16)
            
            # Upsampling path
            self.up1 = Up3D(init_features * 16 + init_features * 8, init_features * 8)
            self.up2 = Up3D(init_features * 8 + init_features * 4, init_features * 4)
            self.up3 = Up3D(init_features * 4 + init_features * 2, init_features * 2)
            self.up4 = Up3D(init_features * 2 + init_features, init_features)
            
            # Output convolution
            self.outc = nn.Conv3d(init_features, out_channels, kernel_size=1)
        
        def forward(self, x):
            # Downsampling
            x1 = self.inc(x)
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            
            # Upsampling with skip connections
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
            
            # Final output
            x = self.outc(x)
            return x
            
    # Create the basic UNet3D model
    if use_physics:
        # For a physics-informed model, we would add additional constraints
        # This is a simplified example - in a real implementation, you would add
        # physics-based loss terms and potentially modify the architecture
        logging.info("Creating physics-informed UNet3D model")
        return UNet3D(in_channels=1, out_channels=1, init_features=64)
    else:
        logging.info("Creating standard UNet3D model")
        return UNet3D(in_channels=1, out_channels=1, init_features=64)


def compute_3d_ssim(img1, img2):
    """
    Compute SSIM between two 3D volumes.
    
    Args:
        img1 (numpy.ndarray): First 3D volume
        img2 (numpy.ndarray): Second 3D volume
        
    Returns:
        float: SSIM score
    """
    from skimage.metrics import structural_similarity as ssim
    
    # Ensure both images are normalized
    img1 = percentile_normalization(img1)
    img2 = percentile_normalization(img2)
    
    # Compute SSIM for each slice and average
    ssim_values = []
    for z in range(img1.shape[0]):
        ssim_slice = ssim(img1[z], img2[z], data_range=1.0)
        ssim_values.append(ssim_slice)
    
    return np.mean(ssim_values)


def compute_n_ssim(prediction, ground_truth, input_image):
    """
    Compute Normalized SSIM as defined in the challenge.
    
    Args:
        prediction (numpy.ndarray): Predicted 3D volume
        ground_truth (numpy.ndarray): Ground truth 3D volume
        input_image (numpy.ndarray): Input 3D volume (single view)
        
    Returns:
        float: Normalized SSIM score
    """
    # Compute SSIM between prediction and ground truth
    prediction_ssim = compute_3d_ssim(prediction, ground_truth)
    
    # Compute reference SSIM between input and ground truth
    reference_ssim = compute_3d_ssim(input_image, ground_truth)
    
    # Compute normalized SSIM
    n_ssim = (prediction_ssim - reference_ssim) / (1 - reference_ssim)
    
    return n_ssim
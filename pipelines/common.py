import os
import logging
import logging.config
import sys
import time
from io import StringIO
from pathlib import Path
import re

import numpy as np
import torch
import tifffile
from torch.utils.data import Dataset, DataLoader
from metaflow import IncludeFile, Parameter, current, S3
from unet3d import UNet3D, PhysicsInformedUNet3D
from gradient_analysis import analyze_gradient_direction, normalize_gradient_direction, visualize_gradient_analysis
PYTHON = "3.12.8"

PACKAGES = {
    "keras": "3.8.0",
    "scikit-learn": "1.6.1",
    "mlflow": "2.20.2",
    "tensorflow": "2.17.0",
    "torch": "2.3.1",
    "torchvision": "0.18.1",
    "torchaudio": "2.3.1",
    "tifffile": "2024.2.12",
    "scikit-image": "0.22.0",
    "cellpose": "2.2.4",
}

def apply_gradient_augmentations(angle_tensor, fused_tensor=None, apply_augs=True, seed=None):
    """
    Apply augmentations to handle variable gradient directions in both Z and XY planes.
    
    Args:
        angle_tensor: PyTorch tensor of shape [C, Z, Y, X] for single-view input
        fused_tensor: PyTorch tensor of shape [C, Z, Y, X] for fused-view target (optional)
        apply_augs: Whether to apply augmentations (set to False during validation/inference)
        seed: Random seed for reproducible augmentations
        
    Returns:
        Tuple of augmented (angle_tensor, fused_tensor) or just angle_tensor if fused_tensor is None
    """
    import random
    
    # Only apply augmentations when requested
    if not apply_augs:
        if fused_tensor is not None:
            return angle_tensor, fused_tensor
        return angle_tensor
    
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Generate flip decisions
    flip_z = random.random() > 0.5
    flip_x = random.random() > 0.5
    flip_y = random.random() > 0.5
    
    # Flip in Z direction (50% chance)
    if flip_z:
        angle_tensor = torch.flip(angle_tensor, dims=[1])  # Flip Z dimension
        if fused_tensor is not None:
            fused_tensor = torch.flip(fused_tensor, dims=[1])
        
    # Flip in X direction (50% chance)
    if flip_x:
        angle_tensor = torch.flip(angle_tensor, dims=[3])  # Flip X dimension
        if fused_tensor is not None:
            fused_tensor = torch.flip(fused_tensor, dims=[3])
        
    # Flip in Y direction (50% chance)
    if flip_y:
        angle_tensor = torch.flip(angle_tensor, dims=[2])  # Flip Y dimension
        if fused_tensor is not None:
            fused_tensor = torch.flip(fused_tensor, dims=[2])
    
    # Reset seed if it was set
    if seed is not None:
        random.seed()
    
    if fused_tensor is not None:
        return angle_tensor, fused_tensor
    return angle_tensor

class FuseMyCellDataset(Dataset):
    """
    Dataset for loading single-view and fused-view light sheet microscopy data.
    
    This dataset handles the specific file naming pattern from the FuseMyCell challenge.
    """
    def __init__(
        self,
        root_dir,
        study_ids=None,
        transform=None,
        patch_size=(64, 128, 128),
        random_crop=True,
        max_samples=None,
        file_pattern=None,
        apply_augmentations=True,
        normalize_gradients=False
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Directory containing the dataset files
            study_ids: List of study IDs to include (default: all studies)
            transform: Optional transforms to apply to the images
            patch_size: Size of patches to extract (z, y, x)
            random_crop: Whether to use random crops during training
            max_samples: Maximum number of samples to use (for testing)
            file_pattern: Optional custom file pattern
            apply_augmentations: Whether to apply gradient direction augmentations
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.apply_augmentations = apply_augmentations
        self.normalize_gradients = normalize_gradients 
        
        # Define study ranges
        self.studies = {
            1: list(range(0, 114)) + list(range(146, 169)),
            2: list(range(114, 146)),
            3: list(range(169, 213)),
            4: list(range(213, 389)),
            5: list(range(389, 401))
        }
        
        # Filter by study if specified
        if study_ids:
            study_image_numbers = []
            for study_id in study_ids:
                if study_id in self.studies:
                    study_image_numbers.extend(self.studies[study_id])
                else:
                    logging.warning(f"Study ID {study_id} not found")
            image_numbers = study_image_numbers
        else:
            # Use all image numbers
            image_numbers = []
            for nums in self.studies.values():
                image_numbers.extend(nums)
        
        # Find all pairs of angle and fused files
        self.file_pairs = self._find_file_pairs(image_numbers, file_pattern)
        
        # Limit the number of samples if needed
        if max_samples and max_samples < len(self.file_pairs):
            self.file_pairs = self.file_pairs[:max_samples]
            
        logging.info(f"Found {len(self.file_pairs)} paired images")
    
    def _find_file_pairs(self, image_numbers, file_pattern=None):
        """
        Find all pairs of angle and fused files.
        
        Args:
            image_numbers: List of image numbers to include
            file_pattern: Optional custom file pattern
            
        Returns:
            List of tuples (angle_file, fused_file)
        """
        # Default patterns
        if file_pattern is None:
            angle_pattern = r"image_(\d+)_nucleus_angle\.tif"
            fused_pattern = r"image_(\d+)_nucleus_fused\.tif"
        else:
            angle_pattern, fused_pattern = file_pattern
        
        # Find all TIFF files in the directory
        all_files = list(self.root_dir.glob("*.tif"))
        
        # Separate angle and fused files
        angle_files = {}
        fused_files = {}
        
        for file_path in all_files:
            # Check for angle files
            angle_match = re.match(angle_pattern, file_path.name)
            if angle_match:
                image_num = int(angle_match.group(1))
                angle_files[image_num] = file_path
                continue
                
            # Check for fused files
            fused_match = re.match(fused_pattern, file_path.name)
            if fused_match:
                image_num = int(fused_match.group(1))
                fused_files[image_num] = file_path
        
        # Create pairs
        file_pairs = []
        for image_num in image_numbers:
            if image_num in angle_files and image_num in fused_files:
                file_pairs.append((angle_files[image_num], fused_files[image_num]))
        
        return file_pairs
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        """
        Get a pair of single-view and fused-view images.
        
        Returns:
            dict: Dictionary containing:
                - 'input': Single-view image tensor
                - 'target': Fused-view image tensor
                - 'file_path': Path to the input file
        """
        def __getitem__(self, idx):
            angle_file, fused_file = self.file_pairs[idx]
            
            # Load the TIFF files
            try:
                angle_img = tifffile.imread(angle_file)
                fused_img = tifffile.imread(fused_file)
            except Exception as e:
                logging.error(f"Error loading files {angle_file} and {fused_file}: {e}")
                # Return a placeholder if loading fails
                return self.__getitem__((idx + 1) % len(self))
            
            # Ensure 3D format (handle 2D images)
            if len(angle_img.shape) == 2:
                angle_img = angle_img[np.newaxis, ...]
            if len(fused_img.shape) == 2:
                fused_img = fused_img[np.newaxis, ...]
                
            # Handle 4D images (with channels)
            if len(angle_img.shape) == 4:
                angle_img = angle_img[..., 0]
            if len(fused_img.shape) == 4:
                fused_img = fused_img[..., 0]
            
            # Apply gradient direction normalization if requested
            if self.normalize_gradients:
                angle_img, gradient_info = normalize_gradient_direction(angle_img)
                # Use the same direction for the fused image
                target_directions = {
                    'z_direction': gradient_info['z_direction'],
                    'y_direction': gradient_info['y_direction'],
                    'x_direction': gradient_info['x_direction']
                }
                fused_img, _ = normalize_gradient_direction(fused_img, target_directions)
                
                # Log gradient info occasionally
                if idx % 100 == 0:
                    logging.info(f"Gradient info for sample {idx}: {gradient_info}")
        
        # Apply normalization
        angle_img = self._normalize(angle_img)
        fused_img = self._normalize(fused_img)
        
        # Extract patch
        angle_patch, fused_patch = self._extract_patch(angle_img, fused_img)
        
        # Convert to tensors
        angle_tensor = torch.from_numpy(angle_patch).float().unsqueeze(0)  # Add channel dimension
        fused_tensor = torch.from_numpy(fused_patch).float().unsqueeze(0)  # Add channel dimension
        
        # Apply gradient direction augmentations
        if self.apply_augmentations:
            angle_tensor, fused_tensor = apply_gradient_augmentations(
                angle_tensor, 
                fused_tensor,
                apply_augs=self.random_crop  # Only apply when in training mode
            )
        
        # Apply transforms if specified
        if self.transform:
            angle_tensor = self.transform(angle_tensor)
            fused_tensor = self.transform(fused_tensor)
        
        return {
            'input': angle_tensor,
            'target': fused_tensor,
            'file_path': str(angle_file)
        }
    
    def _normalize(self, image, pmin=2, pmax=99.8):
        """
        Apply percentile normalization to the image.
        
        Args:
            image: 3D numpy array
            pmin: Minimum percentile
            pmax: Maximum percentile
            
        Returns:
            Normalized image
        """
        low = np.percentile(image, pmin)
        high = np.percentile(image, pmax)
        
        if high > low:
            image = (image - low) / (high - low)
            # Clip values to [0, 1] range
            image = np.clip(image, 0, 1)
        
        return image
    
    def _extract_patch(self, angle_img, fused_img):
        """
        Extract a patch from the angle and fused images.
        
        Args:
            angle_img: 3D numpy array for the angle image
            fused_img: 3D numpy array for the fused image
            
        Returns:
            Tuple of (angle_patch, fused_patch)
        """
        z, y, x = angle_img.shape
        pz, py, px = self.patch_size
        
        # If images are smaller than patch size, pad them
        if z < pz or y < py or x < px:
            # Calculate padding
            pad_z = max(0, pz - z)
            pad_y = max(0, py - y)
            pad_x = max(0, px - x)
            
            # Pad images
            angle_img = np.pad(angle_img, 
                               ((0, pad_z), (0, pad_y), (0, pad_x)), 
                               mode='constant')
            fused_img = np.pad(fused_img, 
                              ((0, pad_z), (0, pad_y), (0, pad_x)), 
                              mode='constant')
            
            # Update dimensions
            z, y, x = angle_img.shape
        
        # Extract random or centered patch
        if self.random_crop and z >= pz and y >= py and x >= px:
            # Random crop
            sz = np.random.randint(0, z - pz + 1)
            sy = np.random.randint(0, y - py + 1)
            sx = np.random.randint(0, x - px + 1)
        else:
            # Center crop
            sz = (z - pz) // 2
            sy = (y - py) // 2
            sx = (x - px) // 2
        
        # Extract patches
        angle_patch = angle_img[sz:sz+pz, sy:sy+py, sx:sx+px]
        fused_patch = fused_img[sz:sz+pz, sy:sy+py, sx:sx+px]
        
        return angle_patch, fused_patch


def prepare_data_loaders(dataset_dir, study_ids=None, batch_size=4, patch_size=(64, 128, 128), 
                        num_workers=4, train_ratio=0.8, max_samples=None):
    """
    Prepare data loaders for training and validation.
    
    Args:
        dataset_dir: Directory containing the dataset
        study_ids: List of study IDs to include (default: all studies)
        batch_size: Batch size for training
        patch_size: Size of patches to extract (z, y, x)
        num_workers: Number of worker threads for data loading
        train_ratio: Ratio of training data to total data
        max_samples: Maximum number of samples to use (for testing)
        
    Returns:
        dict: Dictionary containing training and validation data loaders
    """
    # Create dataset
    dataset = FuseMyCellDataset(
        root_dir=dataset_dir,
        study_ids=study_ids,
        patch_size=patch_size,
        random_crop=True,
        max_samples=max_samples,
        apply_augmentations=True,
        normalize_gradients=False
    )
    
    # Split into training and validation sets
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    
    train_size = int(train_ratio * dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logging.info(f"Created data loaders with {len(train_loader)} training batches and {len(val_loader)} validation batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'dataset': dataset
    }


class DatasetMixin:
    """A mixin for loading and preparing light sheet microscopy datasets.

    This mixin is designed to be combined with any pipeline that requires accessing
    light sheet microscopy datasets for the FuseMyCell challenge.
    """

    dataset_dir = Parameter(
        "dataset-dir",
        help="Directory containing the light sheet microscopy dataset.",
        default="data",
    )

    study_ids = Parameter(
        "study-ids",
        help="Comma-separated list of study IDs to include (e.g., '1,3,4'). Leave empty to use all studies.",
        default="",
    )

    patch_size = Parameter(
        "patch-size",
        help="Size of 3D patches for training (z,y,x)",
        default="32,64,64",  # Updated to a more suitable default
    )

    patch_overlap = Parameter(
        "patch-overlap",
        help="Overlap between adjacent patches during inference (z,y,x)",
        default="8,16,16",  # 25% overlap for each dimension
    )

    s3_bucket = Parameter(
        "s3-bucket",
        help="S3 bucket containing the dataset files (used in production mode).",
        default="",
    )

    def get_patch_size(self):
        """Convert patch size parameter to tuple."""
        return tuple(map(int, self.patch_size.split(',')))
    
    def get_patch_overlap(self):
        """Convert patch overlap parameter to tuple."""
        return tuple(map(int, self.patch_overlap.split(',')))

    def load_dataset(self):
        """Load and prepare the light sheet microscopy dataset."""
        if current.is_production and self.s3_bucket:
            with S3(s3root=self.s3_bucket) as s3:
                files = s3.get_all()
                logging.info("Found %d file(s) in remote location", len(files))
                # In a real implementation, filter and process the files appropriately
                # For now, we'll just return a placeholder
                return {"single_view": [], "fused_view": []}
        else:
            logging.info("Loading data from local filesystem.")
            
            # Parse study IDs if provided
            if self.study_ids:
                study_ids = [int(s.strip()) for s in self.study_ids.split(',')]
                logging.info(f"Using studies: {study_ids}")
            else:
                study_ids = None
                logging.info("Using all available studies")
                
            # Create dataset using the helper function with updated patch size
            patch_size = self.get_patch_size()
            logging.info(f"Using patch size: {patch_size}")
            
            try:
                data_loaders = prepare_data_loaders(
                    dataset_dir=self.dataset_dir,
                    study_ids=study_ids,
                    batch_size=self.training_batch_size,
                    patch_size=patch_size,
                    num_workers=4,
                    normalize_gradients=self.normalize_gradients  # Pass the parameter
                )
                
                self.train_loader = data_loaders['train']
                self.val_loader = data_loaders['val']
                
                logging.info(f"Loaded dataset with {len(data_loaders['dataset'])} sample pairs")
                
                return data_loaders
            except Exception as e:
                logging.error(f"Error loading dataset: {str(e)}")
                raise
    
    def extract_patches_from_volume(self, volume, patch_size=None, overlap=None):
        """
        Extract overlapping patches from a volume for inference.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
            patch_size: Tuple of (z, y, x) patch size
            overlap: Tuple of (z, y, x) overlap between patches
            
        Returns:
            List of (patch, coords) where coords is (z_start, y_start, x_start)
        """
        if patch_size is None:
            patch_size = self.get_patch_size()
        
        if overlap is None:
            overlap = self.get_patch_overlap()
        
        pz, py, px = patch_size
        oz, oy, ox = overlap
        z, y, x = volume.shape
        
        # Calculate step sizes with overlap
        step_z = pz - oz
        step_y = py - oy
        step_x = px - ox
        
        # Calculate number of patches in each dimension
        n_z = max(1, (z - pz) // step_z + 1 + (1 if (z - pz) % step_z > 0 else 0))
        n_y = max(1, (y - py) // step_y + 1 + (1 if (y - py) % step_y > 0 else 0))
        n_x = max(1, (x - px) // step_x + 1 + (1 if (x - px) % step_x > 0 else 0))
        
        # If the volume is smaller than patch size, use a single patch
        if z < pz or y < py or x < px:
            # Pad volume to patch size
            pad_z = max(0, pz - z)
            pad_y = max(0, py - y)
            pad_x = max(0, px - x)
            
            padded_volume = np.pad(volume, 
                                  ((0, pad_z), (0, pad_y), (0, pad_x)), 
                                  mode='constant')
            return [(padded_volume[:pz, :py, :px], (0, 0, 0))]
        
        patches = []
        for iz in range(n_z):
            for iy in range(n_y):
                for ix in range(n_x):
                    # Calculate start position of the patch
                    z_start = min(iz * step_z, z - pz)
                    y_start = min(iy * step_y, y - py)
                    x_start = min(ix * step_x, x - px)
                    
                    # Extract the patch
                    patch = volume[z_start:z_start+pz, y_start:y_start+py, x_start:x_start+px]
                    patches.append((patch, (z_start, y_start, x_start)))
        
        return patches
    
    def stitch_patches_to_volume(self, patches, original_shape, patch_size=None, overlap=None):
        """
        Stitch a list of patches back into a volume using weighted averaging in overlapping regions.
        
        Args:
            patches: List of (patch, coords) where coords is (z_start, y_start, x_start)
            original_shape: Shape of the original volume (Z, Y, X)
            patch_size: Tuple of (z, y, x) patch size
            overlap: Tuple of (z, y, x) overlap between patches
            
        Returns:
            Reconstructed volume as numpy array
        """
        if patch_size is None:
            patch_size = self.get_patch_size()
        
        if overlap is None:
            overlap = self.get_patch_overlap()
        
        z, y, x = original_shape
        pz, py, px = patch_size
        
        # Initialize output volume and weight volume
        output = np.zeros((z, y, x), dtype=np.float32)
        weights = np.zeros((z, y, x), dtype=np.float32)
        
        # Create weight map for patches (higher weights in center, lower at edges)
        weight_map = self._create_weight_map(patch_size)
        
        # Place each patch in the output volume with weighted averaging
        for patch, (z_start, y_start, x_start) in patches:
            # Calculate the actual patch size (might be smaller at boundaries)
            z_end = min(z_start + pz, z)
            y_end = min(y_start + py, y)
            x_end = min(x_start + px, x)
            
            # Handle potentially smaller patch
            p_z = z_end - z_start
            p_y = y_end - y_start
            p_x = x_end - x_start
            
            # Get actual patch and weight map
            actual_patch = patch[:p_z, :p_y, :p_x]
            actual_weights = weight_map[:p_z, :p_y, :p_x]
            
            # Add weighted patch to output
            output[z_start:z_end, y_start:y_end, x_start:x_end] += actual_patch * actual_weights
            weights[z_start:z_end, y_start:y_end, x_start:x_end] += actual_weights
        
        # Normalize output by weights
        # Avoid division by zero
        mask = weights > 0
        output[mask] /= weights[mask]
        
        return output
    
    def _create_weight_map(self, patch_size):
        """
        Create a weight map for blending patches, with higher weights in the center
        and lower weights at the edges.
        
        Args:
            patch_size: Tuple of (z, y, x) patch size
            
        Returns:
            3D numpy array with weights
        """
        pz, py, px = patch_size
        
        # Create 1D weight arrays using a cosine window
        z_weights = np.cos((np.linspace(-np.pi/2, np.pi/2, pz)) ** 2)
        y_weights = np.cos((np.linspace(-np.pi/2, np.pi/2, py)) ** 2)
        x_weights = np.cos((np.linspace(-np.pi/2, np.pi/2, px)) ** 2)
        
        # Create 3D weight map using outer product
        zz, yy, xx = np.meshgrid(z_weights, y_weights, x_weights, indexing='ij')
        weight_map = zz * yy * xx
        
        return weight_map


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
    """
    Build a 3D U-Net model for single-view to multiview fusion.
    
    Args:
        input_shape: Shape of the input volume (z, y, x) or (c, z, y, x)
        use_physics: Whether to use physics-informed neural network
        
    Returns:
        model: PyTorch model for 3D image fusion
    """
    import logging
    
    # Determine whether input_shape includes channel dimension
    if len(input_shape) == 3:  # If input_shape is (z, y, x)
        in_channels = 1
    elif len(input_shape) == 4:  # If input_shape is (c, z, y, x)
        in_channels = input_shape[0]
    else:
        raise ValueError(f"Unexpected input shape: {input_shape}. Should be (z, y, x) or (c, z, y, x)")
    
    # Create appropriate model based on use_physics flag
    if use_physics:
        logging.info("Creating physics-informed UNet3D model")
        model = PhysicsInformedUNet3D(in_channels=in_channels, out_channels=1, init_features=64)
    else:
        logging.info("Creating standard UNet3D model")
        model = UNet3D(in_channels=in_channels, out_channels=1, init_features=64)
    
    return model


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
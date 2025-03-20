"""
Patch-based dataset for memory-efficient training on FuseMyCell data.

This module provides a PyTorch Dataset implementation that loads pre-extracted
patches from disk, with optional caching for frequently used samples.
"""

import os
import torch
import numpy as np
import json
import tifffile
from pathlib import Path
from torch.utils.data import Dataset
import logging

class PatchDataset(Dataset):
    """Dataset for loading pre-extracted patches for FuseMyCell training."""
    
    def __init__(self, metadata_file, transform=None, apply_augmentations=True, 
                 cache_size=0, float32=False):
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
        
        # Set up logging
        self.logger = logging.getLogger("PatchDataset")
        
        # Load metadata
        try:
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Loaded dataset with {len(self.metadata)} patch pairs")
            
            # Fill cache if requested
            if self.cache_size > 0:
                self.logger.info(f"Pre-caching {min(self.cache_size, len(self.metadata))} samples...")
                for i in range(min(self.cache_size, len(self.metadata))):
                    self._load_and_cache(i)
                    
        except Exception as e:
            self.logger.error(f"Error loading metadata file {metadata_file}: {e}")
            # Initialize with empty metadata
            self.metadata = []
    
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
                # If cache is full, remove a random item
                if len(self.cache) >= self.cache_size:
                    # Remove oldest item (not random to avoid thrashing)
                    old_key = next(iter(self.cache))
                    del self.cache[old_key]
                
                self.cache[idx] = result
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading patch {patch_info.get('patch_id', idx)}: {e}")
            # Return a different sample on error
            alt_idx = (idx + 1) % len(self.metadata)
            return self.__getitem__(alt_idx)
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # Load and get the sample
        result = self._load_and_cache(idx)
        
        # Apply augmentations if enabled
        if self.apply_augmentations:
            # Import here to avoid circular imports
            try:
                from ...common import apply_gradient_augmentations
                
                result['input'], result['target'] = apply_gradient_augmentations(
                    result['input'], result['target'], apply_augs=True
                )
            except ImportError:
                try:
                    # Try direct import if module structure is different
                    from common import apply_gradient_augmentations
                    
                    result['input'], result['target'] = apply_gradient_augmentations(
                        result['input'], result['target'], apply_augs=True
                    )
                except ImportError:
                    self.logger.warning("Could not import apply_gradient_augmentations function")
        
        # Apply transforms if specified
        if self.transform:
            result['input'] = self.transform(result['input'])
            result['target'] = self.transform(result['target'])
        
        return result
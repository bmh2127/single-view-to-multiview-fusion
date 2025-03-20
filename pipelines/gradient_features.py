# gradient_features.py
import numpy as np
import torch
import logging
from typing import Tuple
from scipy.ndimage import gaussian_filter

def compute_gradient_features(volume: np.ndarray) -> np.ndarray:
    """
    Compute gradient direction features for a 3D volume.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        
    Returns:
        np.ndarray: Gradient features with shape (6, Z, Y, X)
    """
    # Ensure we have a 3D volume
    if len(volume.shape) < 3:
        volume = volume[np.newaxis, ...]
    elif len(volume.shape) > 3:
        volume = volume[..., 0]
        
    # Compute gradients along each axis
    grad_z, grad_y, grad_x = np.gradient(volume)
    
    # Compute gradient magnitude
    grad_magnitude = np.sqrt(grad_z**2 + grad_y**2 + grad_x**2)
    
    # Avoid division by zero
    epsilon = 1e-8
    nonzero_mask = grad_magnitude > epsilon
    
    # Initialize normalized gradient components
    norm_grad_z = np.zeros_like(grad_z)
    
    # Normalize only where magnitude is non-zero
    norm_grad_z[nonzero_mask] = grad_z[nonzero_mask] / (grad_magnitude[nonzero_mask] + epsilon)
    
    # Compute directional consistency
    smooth_grad_z = gaussian_filter(norm_grad_z, sigma=2.0)
    smooth_grad_y = gaussian_filter(grad_y / (grad_magnitude + epsilon), sigma=2.0)
    smooth_grad_x = gaussian_filter(grad_x / (grad_magnitude + epsilon), sigma=2.0)
    
    directional_consistency = np.sqrt(smooth_grad_z**2 + smooth_grad_y**2 + smooth_grad_x**2)
    
    # Stack all features
    gradient_features = np.stack([
        grad_z,              # Z gradient
        grad_y,              # Y gradient
        grad_x,              # X gradient
        grad_magnitude,      # Gradient magnitude
        norm_grad_z,         # Normalized Z component
        directional_consistency  # Local consistency
    ])
    
    return gradient_features


class GradientFeatureProcessor:
    """
    A class for extracting and processing gradient features.
    """
    
    def __init__(self, use_gradient_features=True):
        self.use_gradient_features = use_gradient_features
    
    def process_batch(self, batch_data):
        """
        Add gradient features to a batch of data.
        
        Args:
            batch_data: Dictionary with 'input' tensor of shape [B, C, Z, Y, X]
            
        Returns:
            Updated batch with gradient features concatenated to input
        """
        if not self.use_gradient_features:
            return batch_data
            
        # Get input tensor
        input_tensor = batch_data['input']
        batch_size = input_tensor.shape[0]
        device = input_tensor.device
        
        # Process each item in batch
        feature_tensors = []
        for b in range(batch_size):
            volume = input_tensor[b, 0].cpu().numpy()  # Get single channel volume
            gradient_features = compute_gradient_features(volume)
            feature_tensor = torch.from_numpy(gradient_features).float().to(device)
            feature_tensors.append(feature_tensor)
        
        # Stack features into batch
        features_batch = torch.stack(feature_tensors)
        
        # Concatenate with original input along channel dimension
        # Original: [B, 1, Z, Y, X]
        # Features: [B, 6, Z, Y, X]
        # Result: [B, 7, Z, Y, X]
        enhanced_input = torch.cat([input_tensor, features_batch], dim=1)
        
        # Update batch data
        batch_data['input_with_features'] = enhanced_input
        
        return batch_data
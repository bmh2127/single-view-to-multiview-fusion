import importlib
import json
import logging
import os
import uuid
from pathlib import Path
import tempfile

import numpy as np
import torch
import tifffile


def percentile_normalization(image, pmin=2, pmax=99.8, axis=None):
    """
    Compute a percentile normalization for the given image.
    
    Parameters:
    - image (array): array (2D or 3D) of the image file.
    - pmin (int or float): the minimal percentage for the percentiles to compute.
                           Values must be between 0 and 100 inclusive.
    - pmax (int or float): the maximal percentage for the percentiles to compute.
                           Values must be between 0 and 100 inclusive.
    - axis: Axis or axes along which the percentiles are computed.
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


def predict_fused_view(model, single_view):
    """
    Generate a fused multi-view prediction from a single view input.
    
    Args:
        model: PyTorch model
        single_view: 3D numpy array (Z, Y, X)
        
    Returns:
        fused_view: 3D numpy array (Z, Y, X)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Normalize input
    normalized_input = percentile_normalization(single_view)
    
    # Convert to tensor and add batch and channel dimensions
    input_tensor = torch.from_numpy(normalized_input).float().unsqueeze(0).unsqueeze(0)
    
    # Move to the same device as the model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Generate prediction
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert back to numpy
    fused_view = output_tensor.cpu().squeeze().numpy()
    
    return fused_view


class _FuseMyCellWrapper:
    """
    A wrapper class that loads artifacts and implements prediction logic for the FuseMyCell model.
    This class is used by MLflow's pyfunc model flavor.
    """
    
    def __init__(self, model_path):
        """
        Load model and artifacts from the given path.
        
        Args:
            model_path: Path to the directory containing model artifacts
        """
        import torch
        import importlib.util
        from pathlib import Path
        
        logging.info(f"Loading FuseMyCell model from {model_path}")
        self.model_path = model_path
        
        # Load model architecture
        model_file = Path(model_path) / "model.pth"
        artifacts_dir = Path(model_path) / "artifacts"
        
        # Import the model definition
        spec = importlib.util.spec_from_file_location(
            "unet3d", 
            Path(model_path) / "code" / "unet3d.py"
        )
        unet_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(unet_module)
        
        # Determine device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Create model instance
        self.model = unet_module.UNet3D(in_channels=1, out_channels=1, init_features=64)
        
        # Load model weights
        self.model.load_state_dict(torch.load(
            artifacts_dir / "model.pth", 
            map_location=self.device
        ))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logging.info("FuseMyCell model loaded successfully")
    
    def predict(self, context, model_input):
        """
        Generate fused view predictions from single view inputs.
        
        Args:
            context: MLflow model context
            model_input: Dictionary containing the input data
                {'single_view': numpy array or path to TIFF file}
                
        Returns:
            Dictionary containing the prediction results
        """
        # Process input (which could be a file path or numpy array)
        if isinstance(model_input, dict) and 'single_view' in model_input:
            input_data = model_input['single_view']
        else:
            input_data = model_input
            
        # Handle different input types
        if isinstance(input_data, str):
            # Input is a file path
            logging.info(f"Loading input from file: {input_data}")
            try:
                single_view = tifffile.imread(input_data)
                
                # Handle different dimensions
                if len(single_view.shape) == 2:  # Single 2D image
                    single_view = single_view[np.newaxis, ...]
                elif len(single_view.shape) == 4:  # Multiple channels
                    # For simplicity, just take the first channel
                    single_view = single_view[..., 0]
                
            except Exception as e:
                raise ValueError(f"Error loading input file: {str(e)}")
        elif isinstance(input_data, np.ndarray):
            # Input is a numpy array
            single_view = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        # Generate prediction
        fused_view = predict_fused_view(self.model, single_view)
        
        # Save prediction to a temporary file if needed
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            output_path = tmp.name
            tifffile.imwrite(output_path, fused_view)
        
        return {
            'fused_view': fused_view,
            'output_path': output_path
        }
import importlib
import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import random

import numpy as np
import pydantic
import torch
import tifffile
import mlflow
from mlflow.models import set_model
from mlflow.pyfunc.model import PythonModelContext


class Input(pydantic.BaseModel):
    """Prediction input that will be received from the client.

    This class defines the structure of the input data that the
    model will receive from the client. The input data will be automatically validated
    by MLflow against this schema before making a prediction.
    """
    # The input can be either a path to a TIFF file or a numpy array
    file_path: Optional[str] = None
    patch_size: Optional[List[int]] = None  # [z, y, x] dimensions for patch extraction
    ground_truth_path: Optional[str] = None  # Path to ground truth image for evaluation
    use_patches: Optional[bool] = False  # Whether to use patch-based inference
    patch_overlap: Optional[List[int]] = None  # [z, y, x] dimensions for patch overlap
    apply_augmentations: Optional[bool] = False  # Whether to apply gradient direction augmentations


class Output(pydantic.BaseModel):
    """Prediction output that will be returned to the client.

    This class defines the structure of the output data that the
    model will return to the client.
    """
    n_ssim: Optional[float] = None      # Normalized SSIM score if ground truth is provided
    output_path: Optional[str] = None   # Path to the saved output TIFF file


def apply_gradient_augmentations(tensor, apply_augs=True, seed=None):
    """
    Apply augmentations to handle variable gradient directions in both Z and XY planes.
    
    Args:
        tensor: PyTorch tensor of shape [C, Z, Y, X] for input
        apply_augs: Whether to apply augmentations (set to False during validation/inference)
        seed: Random seed for reproducible augmentations
        
    Returns:
        Augmented tensor
    """
    # Only apply augmentations when requested
    if not apply_augs:
        return tensor
    
    # Set seed for reproducibility if provided
    if seed is not None:
        random.seed(seed)
    
    # Generate flip decisions
    flip_z = random.random() > 0.5
    flip_x = random.random() > 0.5
    flip_y = random.random() > 0.5
    
    # Flip in Z direction (50% chance)
    if flip_z:
        tensor = torch.flip(tensor, dims=[1])  # Flip Z dimension
    
    # Flip in X direction (50% chance)
    if flip_x:
        tensor = torch.flip(tensor, dims=[3])  # Flip X dimension
    
    # Flip in Y direction (50% chance)
    if flip_y:
        tensor = torch.flip(tensor, dims=[2])  # Flip Y dimension
    
    # Reset seed if it was set
    if seed is not None:
        random.seed()
    
    return tensor

class Model(mlflow.pyfunc.PythonModel):
    """A custom model implementing an inference pipeline for single-view to multiview fusion.

    This inference pipeline processes single-view light sheet microscopy images and 
    predicts their corresponding fused multi-view representation using a 3D U-Net model.
    """

    def __init__(self) -> None:
        """Initialize the model."""
        self.backend = None
        self.model = None
        self.device = None

    def load_context(self, context: PythonModelContext) -> None:
        """Load and prepare the model context to make predictions.

        This function is called only once when the model is loaded. It loads
        the 3D U-Net model and prepares it for inference.
        """
        self._configure_logging()
        self._initialize_backend()
        self._load_artifacts(context)

        logging.info("FuseMyCell model is ready to receive requests")

    def predict(
        self,
        context,  # noqa: ARG002
        model_input: Union[Dict[str, Any], List[Dict[str, Any]]],
        params: Optional[Dict[str, Any]] = None,  # noqa: ARG002
    ) -> List[Dict[str, Any]]:
        """Handle the request received from the client.

        This method processes the input data, makes a prediction using the model,
        and returns the results to the client.

        Args:
            context: MLflow model context
            model_input: Input data as a dictionary or list of dictionaries
            params: Additional parameters for prediction (not used)

        Returns:
            List of dictionaries containing prediction results
        """
        # Convert input to list if it's a single dictionary
        if isinstance(model_input, dict):
            model_input = [model_input]

        if not model_input:
            logging.warning("Received an empty request.")
            return []

        logging.info(
            "Received prediction request with %d %s",
            len(model_input),
            "samples" if len(model_input) > 1 else "sample",
        )

        # Process each sample and collect results
        model_output = []
        for i, sample in enumerate(model_input):
            try:
                # Check if we should use patch-based inference
                use_patches = sample.get('use_patches', False)
                
                if use_patches:
                    # Process using patch-based approach
                    result = self._process_with_patches(sample)
                else:
                    # Load and preprocess the input for whole-volume inference
                    processed_input, input_volume = self.process_input(sample)
                    if processed_input is None:
                        logging.warning(f"Sample at index {i} could not be processed")
                        continue

                    # Generate prediction
                    with torch.no_grad():
                        prediction = self.model(processed_input)

                    # Process the output
                    result = self.process_output(prediction, input_volume, sample)
                
                model_output.append(result)

            except Exception as e:
                logging.exception(f"Error processing sample at index {i}: {str(e)}")
                model_output.append({"error": str(e)})

        # Store the inputs and outputs if a backend is configured
        if self.backend is not None:
            self.backend.save(model_input, model_output)

        logging.info("Returning prediction to the client")
        logging.debug("%s", model_output)

        return model_output

    def _process_with_patches(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a sample using patch-based inference.
        
        Args:
            sample: Input data dictionary
            
        Returns:
            Dictionary containing prediction results
        """
        logging.info("Using patch-based inference...")
        
        # Get the input volume
        input_volume = self._load_input_volume(sample)
        if input_volume is None:
            return {"error": "Failed to load input volume"}
        
        # Save the original input for metrics calculation
        original_input = input_volume.copy()
            
        # Handle different dimensions
        if len(input_volume.shape) == 2:  # Single 2D image
            input_volume = input_volume[np.newaxis, ...]
        elif len(input_volume.shape) == 4:  # Multiple channels
            input_volume = input_volume[..., 0]  # Take first channel
        
        # Apply normalization to the entire volume
        normalized_volume = self._percentile_normalization(input_volume)
        
        # Extract patches - default or specified patch size
        patch_size = sample.get('patch_size', [64, 128, 128])
        if not isinstance(patch_size, list) or len(patch_size) != 3:
            patch_size = [64, 128, 128]
        
        # Extract patches - default or specified patch overlap
        patch_overlap = sample.get('patch_overlap', [16, 32, 32])
        if not isinstance(patch_overlap, list) or len(patch_overlap) != 3:
            patch_overlap = [16, 32, 32]
        
        # Check if we should apply augmentations
        apply_augs = sample.get('apply_augmentations', False)
        if apply_augs:
            logging.info("Using gradient direction augmentations for patch-based inference")
        
        # Extract patches
        patches = self._extract_patches_from_volume(normalized_volume, patch_size, patch_overlap)
        logging.info(f"Extracted {len(patches)} patches from volume of shape {normalized_volume.shape}")
        
        # Process each patch
        output_patches = []
        with torch.no_grad():
            for patch, coords in patches:
                # Convert to tensor and add batch & channel dimensions
                patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
                
                # Apply gradient direction augmentations if requested
                if apply_augs:
                    # Use patch coordinates as part of the seed for reproducible results
                    seed = sum(coords)
                    patch_tensor = apply_gradient_augmentations(
                        patch_tensor, 
                        apply_augs=True,
                        seed=seed
                    )
                
                patch_tensor = patch_tensor.to(self.device)
                
                # Generate prediction
                output_patch = self.model(patch_tensor)
                
                # If augmentations were applied, apply inverse augmentations to the output
                if apply_augs:
                    # Use same seed for reproducibility
                    random.seed(seed)
                    
                    # Recompute the same flip decisions
                    flip_z = random.random() > 0.5
                    flip_x = random.random() > 0.5
                    flip_y = random.random() > 0.5
                    
                    # Apply inverse flips in reverse order
                    if flip_y:
                        output_patch = torch.flip(output_patch, dims=[2])
                    if flip_x:
                        output_patch = torch.flip(output_patch, dims=[3])
                    if flip_z:
                        output_patch = torch.flip(output_patch, dims=[1])
                    
                    # Reset seed
                    random.seed()
                
                # Save output patch and coordinates
                output_patches.append((output_patch[0, 0].cpu().numpy(), coords))
        
        # Stitch patches back together
        stitched_output = self._stitch_patches_to_volume(
            output_patches, normalized_volume.shape, patch_size, patch_overlap
        )
        
        # Process the stitched output
        return self._save_output_and_compute_metrics(stitched_output, original_input, sample)

    def _load_input_volume(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Load input volume from file or data field.
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Numpy array containing input volume
        """
        try:
            # Get the file path from the input
            file_path = input_data.get('file_path')
            
            if file_path:
                # Load the TIFF file
                logging.info(f"Loading input from file: {file_path}")
                input_volume = tifffile.imread(file_path)
            elif 'data' in input_data:
                # Use the provided data
                input_volume = input_data['data']
                if not isinstance(input_volume, np.ndarray):
                    raise ValueError("Input data must be a numpy array")
            else:
                raise ValueError("Input must contain either 'file_path' or 'data'")
                
            return input_volume
            
        except Exception as e:
            logging.exception(f"Error loading input volume: {str(e)}")
            return None

    def _save_output_and_compute_metrics(self, output_volume: np.ndarray, 
                                        input_volume: np.ndarray,
                                        input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Save output volume to file and compute metrics if ground truth is available.
        
        Args:
            output_volume: Output volume as numpy array
            input_volume: Original input volume
            input_data: Original input data dictionary
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Save the output to a temporary file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                output_path = tmp.name
                tifffile.imwrite(output_path, output_volume)
            
            result = {
                "output_path": output_path
            }
            
            # Calculate N_SSIM if ground truth is provided
            if 'ground_truth_path' in input_data and input_data['ground_truth_path']:
                try:
                    # Load ground truth
                    gt_path = input_data['ground_truth_path']
                    ground_truth = tifffile.imread(gt_path)
                    
                    # Ensure appropriate dimensions
                    if len(ground_truth.shape) == 2:
                        ground_truth = ground_truth[np.newaxis, ...]
                    elif len(ground_truth.shape) == 4:
                        ground_truth = ground_truth[..., 0]
                        
                    # Calculate N_SSIM
                    result["n_ssim"] = self._compute_n_ssim(
                        output_volume, ground_truth, input_volume
                    )
                    logging.info(f"Calculated N_SSIM: {result['n_ssim']}")
                except Exception as e:
                    logging.warning(f"Could not calculate N_SSIM: {str(e)}")
            
            return result
            
        except Exception as e:
            logging.exception(f"Error saving output and computing metrics: {str(e)}")
            return {"error": str(e)}
                
    def process_input(self, input_data: Dict[str, Any]) -> tuple:
        """Process the input data received from the client.

        Args:
            input_data: Input data dictionary with file_path or data

        Returns:
            Tuple of (processed input tensor, original input volume)
        """
        logging.info("Processing input data...")

        try:
            # Load the input volume
            input_volume = self._load_input_volume(input_data)
            if input_volume is None:
                return None, None
            
            # Save original input before any modifications
            original_input = input_volume.copy()
            
            # Handle different dimensions
            if len(input_volume.shape) == 2:  # Single 2D image
                input_volume = input_volume[np.newaxis, ...]
            elif len(input_volume.shape) == 4:  # Multiple channels
                # Take the first channel
                input_volume = input_volume[..., 0]
            
            # Apply normalization
            normalized_volume = self._percentile_normalization(input_volume)
            
            # Extract patch if requested
            patch_size = input_data.get('patch_size')
            if patch_size and len(patch_size) == 3:
                z, y, x = normalized_volume.shape
                pz, py, px = patch_size
                
                if z >= pz and y >= py and x >= px:
                    # Extract centered patch
                    sz = (z - pz) // 2
                    sy = (y - py) // 2
                    sx = (x - px) // 2
                    normalized_volume = normalized_volume[sz:sz+pz, sy:sy+py, sx:sx+px]
            
            # Convert to tensor and add batch and channel dimensions
            input_tensor = torch.from_numpy(normalized_volume).float().unsqueeze(0).unsqueeze(0)
            
            # Apply gradient direction augmentations if requested
            apply_augs = input_data.get('apply_augmentations', False)
            if apply_augs:
                input_tensor = apply_gradient_augmentations(
                    input_tensor,
                    apply_augs=True,
                    # Use deterministic augmentation for reproducibility
                    seed=42
                )
                logging.info("Applied gradient direction augmentations")
            
            # Move to the appropriate device
            input_tensor = input_tensor.to(self.device)
            
            return input_tensor, original_input
            
        except Exception as e:
            logging.exception(f"Error processing input: {str(e)}")
            return None, None

    def process_output(self, output: torch.Tensor, input_volume: np.ndarray, 
                      input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the prediction generated by the model.

        Args:
            output: Raw model output tensor
            input_volume: Original input volume
            input_data: Original input data dictionary

        Returns:
            Processed output as a dictionary
        """
        logging.info("Processing model output...")

        # Convert output tensor to numpy array
        output_volume = output.cpu().squeeze().numpy()
        
        # Save output and compute metrics
        return self._save_output_and_compute_metrics(output_volume, input_volume, input_data)

    def _initialize_backend(self):
        """Initialize the model backend that the pipeline will use to store the data.

        The backend is responsible for storing the input requests and the predictions.
        """
        logging.info("Initializing model backend...")
        backend_class = os.getenv("MODEL_BACKEND") or None

        if backend_class is not None:
            # We can optionally load a JSON configuration file for the backend
            backend_config = os.getenv("MODEL_BACKEND_CONFIG", None)

            try:
                if backend_config is not None:
                    backend_config = Path(backend_config)
                    backend_config = (
                        json.loads(backend_config.read_text())
                        if backend_config.exists()
                        else None
                    )

                module, cls = backend_class.rsplit(".", 1)
                module = importlib.import_module(module)
                self.backend = getattr(module, cls)(config=backend_config)
                
            except Exception:
                logging.exception(
                    'There was an error initializing backend "%s".',
                    backend_class,
                )

        logging.info("Backend: %s", backend_class if self.backend else None)

    def _load_artifacts(self, context: PythonModelContext):
        """Load model artifacts from the MLflow context.

        Args:
            context: MLflow model context containing model artifacts
        """
        if context is None:
            logging.warning("No model context was provided.")
            return

        try:
            logging.info("Loading model artifacts...")
            
            # Import the model architecture
            model_dir = context.artifacts.get("model_dir")
            if not model_dir:
                raise ValueError("Model directory not found in artifacts")
                
            # Add the code directory to the Python path so we can import from it
            code_path = Path(context.artifacts.get("code", ""))
            if code_path.exists():
                import sys
                sys.path.append(str(code_path.parent))
            
            # Import the UNet3D class
            try:
                from unet3d import UNet3D
            except ImportError:
                # Try another location
                sys.path.append(str(code_path))
                try:
                    from unet3d import UNet3D
                except ImportError:
                    raise ImportError("Could not import UNet3D model definition")
            
            # Determine device (CPU or GPU)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            
            # Create model instance
            self.model = UNet3D(in_channels=1, out_channels=1, init_features=64)
            
            # Load model weights
            artifacts_dir = os.path.join(model_dir, "artifacts")
            model_path = os.path.join(artifacts_dir, "model.pth")
            
            self.model.load_state_dict(torch.load(
                model_path, 
                map_location=self.device
            ))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logging.info("Model artifacts loaded successfully")
            
        except Exception as e:
            logging.exception(f"Error loading model artifacts: {str(e)}")
            raise

    def _configure_logging(self):
        """Configure how the logging system will behave."""
        import sys

        if Path("logging.conf").exists():
            logging.config.fileConfig("logging.conf")
        else:
            logging.basicConfig(
                format="%(asctime)s [%(levelname)s] %(message)s",
                handlers=[logging.StreamHandler(sys.stdout)],
                level=logging.INFO,
            )
    
    def _percentile_normalization(self, image, pmin=2, pmax=99.8, axis=None):
        """Compute a percentile normalization for the given image."""
        low_percentile = np.percentile(image, pmin, axis=axis, keepdims=True)
        high_percentile = np.percentile(image, pmax, axis=axis, keepdims=True)
        
        if low_percentile == high_percentile:
            logging.warning(f"Same min {low_percentile} and high {high_percentile}, image may be empty")
            return image
        
        return (image - low_percentile) / (high_percentile - low_percentile)
    
    def _compute_3d_ssim(self, img1, img2):
        """Compute SSIM between two 3D volumes."""
        from skimage.metrics import structural_similarity as ssim
        
        # Ensure both images are normalized
        img1 = self._percentile_normalization(img1)
        img2 = self._percentile_normalization(img2)
        
        # Ensure both images have the same shape
        if img1.shape != img2.shape:
            logging.warning(f"Image shapes don't match: {img1.shape} vs {img2.shape}")
            # Resize to the smaller dimensions
            min_shape = [min(d1, d2) for d1, d2 in zip(img1.shape, img2.shape)]
            img1 = img1[:min_shape[0], :min_shape[1], :min_shape[2]]
            img2 = img2[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        # Compute SSIM for each slice and average
        ssim_values = []
        for z in range(img1.shape[0]):
            try:
                ssim_slice = ssim(img1[z], img2[z], data_range=1.0)
                ssim_values.append(ssim_slice)
            except Exception as e:
                logging.warning(f"Error computing SSIM for slice {z}: {e}")
        
        if not ssim_values:
            return 0.0
        
        return np.mean(ssim_values)
    
    def _compute_n_ssim(self, prediction, ground_truth, input_image):
        """Compute Normalized SSIM as defined in the challenge."""
        # Compute SSIM between prediction and ground truth
        prediction_ssim = self._compute_3d_ssim(prediction, ground_truth)
        
        # Compute reference SSIM between input and ground truth
        reference_ssim = self._compute_3d_ssim(input_image, ground_truth)
        
        # Ensure reference_ssim is not 1 to avoid division by zero
        if reference_ssim >= 0.9999:
            logging.warning("Reference SSIM is very close to 1, using fixed denominator")
            return prediction_ssim - reference_ssim  # Just return the improvement
        
        # Compute normalized SSIM
        n_ssim = (prediction_ssim - reference_ssim) / (1.0 - reference_ssim)
        
        return n_ssim
        
    def _extract_patches_from_volume(self, volume, patch_size, overlap):
        """Extract overlapping patches from a volume for inference.
        
        Args:
            volume: 3D numpy array (Z, Y, X)
            patch_size: List or tuple of [z, y, x] patch size
            overlap: List or tuple of [z, y, x] overlap between patches
            
        Returns:
            List of (patch, coords) where coords is (z_start, y_start, x_start)
        """
        # Ensure inputs are tuples/lists of length 3
        if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
            patch_size = (64, 128, 128)  # Default patch size
        else:
            patch_size = tuple(patch_size)
            
        if not isinstance(overlap, (list, tuple)) or len(overlap) != 3:
            overlap = (16, 32, 32)  # Default 25% overlap
        else:
            overlap = tuple(overlap)
        
        pz, py, px = patch_size
        oz, oy, ox = overlap
        z, y, x = volume.shape
        
        # Calculate step sizes with overlap
        step_z = max(1, pz - oz)
        step_y = max(1, py - oy)
        step_x = max(1, px - ox)
        
        # Calculate number of patches in each dimension
        n_z = max(1, math.ceil((z - pz) / step_z) + 1) if z > pz else 1
        n_y = max(1, math.ceil((y - py) / step_y) + 1) if y > py else 1
        n_x = max(1, math.ceil((x - px) / step_x) + 1) if x > px else 1
        
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
    
    def _stitch_patches_to_volume(self, patches, original_shape, patch_size, overlap):
        """Stitch a list of patches back into a volume using weighted averaging in overlapping regions.
        
        Args:
            patches: List of (patch, coords) where coords is (z_start, y_start, x_start)
            original_shape: Shape of the original volume (Z, Y, X)
            patch_size: List or tuple of [z, y, x] patch size
            overlap: List or tuple of [z, y, x] overlap between patches
            
        Returns:
            Reconstructed volume as numpy array
        """
        # Ensure inputs are tuples/lists of length 3
        if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
            patch_size = (64, 128, 128)  # Default patch size
        else:
            patch_size = tuple(patch_size)
            
        if not isinstance(overlap, (list, tuple)) or len(overlap) != 3:
            overlap = (16, 32, 32)  # Default 25% overlap
        else:
            overlap = tuple(overlap)
        
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
        """Create a weight map for blending patches, with higher weights in the center
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


# Set the model instance for MLflow
set_model(Model())
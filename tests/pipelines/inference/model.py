import importlib
import json
import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pydantic
import torch
import tifffile
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


class Output(pydantic.BaseModel):
    """Prediction output that will be returned to the client.

    This class defines the structure of the output data that the
    model will return to the client.
    """
    n_ssim: Optional[float] = None      # Normalized SSIM score if ground truth is provided
    output_path: Optional[str] = None   # Path to the saved output TIFF file


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
                # Load and preprocess the input
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

    def process_input(self, input_data: Dict[str, Any]) -> tuple:
        """Process the input data received from the client.

        Args:
            input_data: Input data dictionary with file_path or data

        Returns:
            Tuple of (processed input tensor, original input volume)
        """
        logging.info("Processing input data...")

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

        try:
            # Convert output tensor to numpy array
            output_volume = output.cpu().squeeze().numpy()
            
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
            logging.exception(f"Error processing output: {str(e)}")
            return {"error": str(e)}

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


# Set the model instance for MLflow
set_model(Model())
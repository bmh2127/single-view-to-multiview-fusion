import importlib
import json
import logging
import os
import random
import re
import sqlite3
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
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


def extract_patches_from_volume(volume, patch_size=(32, 64, 64), overlap=(8, 16, 16)):
    """
    Extract overlapping patches from a volume for inference.
    
    Args:
        volume: 3D numpy array (Z, Y, X)
        patch_size: Tuple of (z, y, x) patch size
        overlap: Tuple of (z, y, x) overlap between patches
        
    Returns:
        List of (patch, coords) where coords is (z_start, y_start, x_start)
    """
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


def create_weight_map(patch_size):
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


def stitch_patches_to_volume(patches, original_shape, patch_size=(32, 64, 64), overlap=(8, 16, 16)):
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
    z, y, x = original_shape
    pz, py, px = patch_size
    
    # Initialize output volume and weight volume
    output = np.zeros((z, y, x), dtype=np.float32)
    weights = np.zeros((z, y, x), dtype=np.float32)
    
    # Create weight map for patches (higher weights in center, lower at edges)
    weight_map = create_weight_map(patch_size)
    
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


def predict_fused_view(model, single_view, patch_size=(32, 64, 64), overlap=(8, 16, 16)):
    """
    Generate a fused multi-view prediction from a single view input.
    
    This function handles large volumes by:
    1. Breaking the input into patches
    2. Processing each patch through the model
    3. Stitching the predictions back together
    
    Args:
        model: PyTorch model
        single_view: 3D numpy array (Z, Y, X)
        patch_size: Size of patches to process
        overlap: Overlap between adjacent patches
        
    Returns:
        fused_view: 3D numpy array (Z, Y, X)
    """
    # Ensure model is in evaluation mode
    model.eval()
    
    # Get original shape for reconstruction
    original_shape = single_view.shape
    
    # Handle small volumes directly
    if all(dim <= max_dim for dim, max_dim in zip(original_shape, patch_size)):
        # If volume is smaller than patch size, process directly
        normalized_input = percentile_normalization(single_view)
        input_tensor = torch.from_numpy(normalized_input).float().unsqueeze(0).unsqueeze(0)
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        input_tensor = input_tensor.to(device)
        
        # Generate prediction
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Convert back to numpy
        return output_tensor.cpu().squeeze().numpy()
    
    # For large volumes, process in patches
    logging.info(f"Processing large volume of shape {original_shape} using patches of size {patch_size}")
    
    # Extract patches
    patches = extract_patches_from_volume(single_view, patch_size, overlap)
    logging.info(f"Extracted {len(patches)} patches")
    
    # Process each patch
    processed_patches = []
    for i, (patch, coords) in enumerate(patches):
        # Normalize the patch
        normalized_patch = percentile_normalization(patch)
        
        # Convert to tensor
        patch_tensor = torch.from_numpy(normalized_patch).float().unsqueeze(0).unsqueeze(0)
        
        # Move to the same device as the model
        device = next(model.parameters()).device
        patch_tensor = patch_tensor.to(device)
        
        # Generate prediction
        with torch.no_grad():
            output_tensor = model(patch_tensor)
        
        # Convert back to numpy
        output_patch = output_tensor.cpu().squeeze().numpy()
        
        # Store processed patch with its coordinates
        processed_patches.append((output_patch, coords))
        
        if (i + 1) % 10 == 0:
            logging.info(f"Processed {i + 1}/{len(patches)} patches")
    
    # Stitch patches back together
    logging.info("Stitching patches together")
    fused_view = stitch_patches_to_volume(processed_patches, original_shape, patch_size, overlap)
    
    return fused_view


class Backend(ABC):
    """Abstract class defining the interface of a backend."""

    @abstractmethod
    def load(self, limit: int) -> pd.DataFrame | None:
        """Load production data from the backend database.

        Args:
            limit: The maximum number of samples to load from the database.

        """

    @abstractmethod
    def save(self, model_input: pd.DataFrame, model_output: list) -> None:
        """Save production data and model outputs to the database.

        Args:
            model_input: The input data received by the model
            model_output: The output data generated by the model.

        """

    @abstractmethod
    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Label every unlabeled sample stored in the backend database.

        This function will generate fake ground truth data for any unlabeled samples
        stored in the backend database.

        Args:
            ground_truth_quality: The quality of the ground truth labels to generate.
                A value of 1.0 will generate labels that match the model predictions. A
                value less than 1.0 will introduce noise in the labels to simulate
                inaccurate model predictions

        """

    @abstractmethod
    def invoke(self, payload: list | dict) -> dict | None:
        """Make a prediction request to the hosted model.

        Args:
            payload: The data to send to the model for prediction.

        """

    @abstractmethod
    def deploy(self, model_uri: str, model_version: str) -> None:
        """Deploy the supplied model.

        Args:
            model_uri: The path where the model artifacts are located.
            model_version: The version of the model that will be deployed.

        """

    def get_fake_label(self, prediction, ground_truth_quality):
        """Generate a fake ground truth label for a sample.

        This function will randomly return a ground truth label taking into account the
        prediction quality we want to achieve.

        Args:
            prediction: The model prediction for the sample.
            ground_truth_quality: The quality of the ground truth labels to generate.

        """
        # For FuseMyCell, we need to create a fake ground truth 3D volume
        # This is a simplified implementation - in a real scenario, you would
        # want to generate more realistic synthetic ground truth
        
        # Get the shape of the prediction
        if hasattr(prediction, 'shape'):
            shape = prediction.shape
        else:
            # If prediction is not a numpy array, use a default shape
            shape = (64, 128, 128)  # Default patch size
            
        # Create random noise with similar statistics as the prediction
        noise = np.random.normal(0.5, 0.2, shape)
        
        # Blend prediction with noise based on quality
        if ground_truth_quality >= 1.0:
            # Return exact prediction (perfect quality)
            return prediction
        else:
            # Blend prediction with noise
            blend = ground_truth_quality * prediction + (1 - ground_truth_quality) * noise
            
            # Normalize to [0, 1] range
            return percentile_normalization(blend)


class Local(Backend):
    """Local backend implementation.

    A model with this backend will be deployed using `mlflow model serve` and will use
    a SQLite database to store production data.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize backend using the supplied configuration.

        If the configuration is not provided, the class will attempt to read the
        configuration from environment variables.
        """
        self.target = (
            config.get("target", "http://127.0.0.1:8080/invocations")
            if config
            else "http://127.0.0.1:8080/invocations"
        )
        self.database = "fusemycell.db"

        if config:
            self.database = config.get("database", self.database)
        else:
            self.database = os.getenv("MODEL_BACKEND_DATABASE", self.database)
            
        # Data directory for storing TIFF files
        self.data_dir = os.path.join(os.path.dirname(self.database), "tiff_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configure patch settings
        if config and 'patch_config' in config:
            patch_config = config['patch_config']
            self.patch_size = patch_config.get('patch_size', (32, 64, 64))
            self.patch_overlap = patch_config.get('patch_overlap', (8, 16, 16))
        else:
            self.patch_size = (32, 64, 64)
            self.patch_overlap = (8, 16, 16)

        logging.info("Backend database: %s", self.database)
        logging.info("TIFF data directory: %s", self.data_dir)
        logging.info("Using patch size: %s", self.patch_size)
        logging.info("Using patch overlap: %s", self.patch_overlap)

    def load(self, limit: int = 100) -> pd.DataFrame | None:
        """Load production data from a SQLite database."""
        if not Path(self.database).exists():
            logging.error("Database %s does not exist.", self.database)
            return None

        connection = sqlite3.connect(self.database)

        query = (
            "SELECT uuid, input_path, output_path, n_ssim, ground_truth_path, prediction_date "
            "FROM predictions "
            "ORDER BY prediction_date DESC LIMIT ?;"
        )

        data = pd.read_sql_query(query, connection, params=(limit,))
        connection.close()

        return data

    def save(self, model_input: list | dict, model_output: list) -> None:
        """Save production data to a SQLite database.

        If the database doesn't exist, this function will create it.
        """
        logging.info("Storing production data in the database...")

        # Ensure model_input is a list
        if isinstance(model_input, dict):
            model_input = [model_input]

        connection = None
        try:
            connection = sqlite3.connect(self.database)
            
            # Create the table if it doesn't exist
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    uuid TEXT PRIMARY KEY,
                    input_path TEXT,
                    output_path TEXT,
                    n_ssim REAL,
                    ground_truth_path TEXT,
                    prediction_date TIMESTAMP
                )
                """
            )

            # Process and save each prediction
            current_time = datetime.now(timezone.utc).isoformat()
            
            for i, (input_data, output_data) in enumerate(zip(model_input, model_output)):
                record_uuid = str(uuid.uuid4())
                
                # Handle input data
                input_path = input_data.get('file_path')
                if not input_path:
                    # If no file path was provided, save the input data to a file
                    if 'data' in input_data and isinstance(input_data['data'], np.ndarray):
                        input_path = os.path.join(self.data_dir, f"{record_uuid}_input.tif")
                        tifffile.imwrite(input_path, input_data['data'])
                
                # Handle output data
                output_path = output_data.get('output_path')
                if output_path:
                    # Copy the output file to our data directory for long-term storage
                    new_output_path = os.path.join(self.data_dir, f"{record_uuid}_output.tif")
                    try:
                        # Read and then write the file (to avoid file system issues)
                        output_data_array = tifffile.imread(output_path)
                        tifffile.imwrite(new_output_path, output_data_array)
                        output_path = new_output_path
                    except Exception as e:
                        logging.error(f"Failed to copy output file: {e}")
                
                # Extract N-SSIM if available
                n_ssim = output_data.get('n_ssim')
                
                # Get ground truth path if available
                ground_truth_path = input_data.get('ground_truth_path')
                
                # Insert record into the database
                connection.execute(
                    """
                    INSERT INTO predictions 
                    (uuid, input_path, output_path, n_ssim, ground_truth_path, prediction_date)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (record_uuid, input_path, output_path, n_ssim, ground_truth_path, current_time)
                )
            
            connection.commit()
            logging.info(f"Saved {len(model_output)} prediction records to database")

        except sqlite3.Error:
            logging.exception(
                "There was an error saving production data to the database.",
            )
        finally:
            if connection:
                connection.close()

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Label unlabeled samples by generating synthetic ground truth."""
        if not Path(self.database).exists():
            logging.error("Database %s does not exist.", self.database)
            return 0

        connection = None
        try:
            connection = sqlite3.connect(self.database)

            # Get all records without ground truth
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE ground_truth_path IS NULL",
                connection,
            )
            logging.info("Loaded %s unlabeled samples.", len(df))

            # If there are no unlabeled samples, we don't need to do anything else.
            if df.empty:
                return 0

            labeled_count = 0
            for _, row in df.iterrows():
                uuid_val = row["uuid"]
                output_path = row["output_path"]
                
                if not output_path or not os.path.exists(output_path):
                    continue
                
                try:
                    # Load the prediction
                    prediction = tifffile.imread(output_path)
                    
                    # Generate synthetic ground truth
                    ground_truth = self.get_fake_label(prediction, ground_truth_quality)
                    
                    # Save the synthetic ground truth
                    ground_truth_path = os.path.join(self.data_dir, f"{uuid_val}_ground_truth.tif")
                    tifffile.imwrite(ground_truth_path, ground_truth)
                    
                    # Update the database
                    connection.execute(
                        "UPDATE predictions SET ground_truth_path = ? WHERE uuid = ?",
                        (ground_truth_path, uuid_val)
                    )
                    
                    labeled_count += 1
                except Exception as e:
                    logging.error(f"Error labeling sample {uuid_val}: {e}")

            connection.commit()
            logging.info(f"Labeled {labeled_count} samples")
            return labeled_count
            
        except Exception:
            logging.exception("There was an error labeling production data")
            return 0
        finally:
            if connection:
                connection.close()

    def invoke(self, payload: list | dict) -> dict | None:
        """Make a prediction request to the hosted model.
        
        For large volumes, this method automatically handles patch-based processing
        to avoid memory issues.
        """
        import requests

        logging.info('Running prediction on "%s"...', self.target)

        try:
            # Check if we need to process in patches
            need_patching = False
            if isinstance(payload, dict) and 'file_path' in payload:
                # Check file size or dimensions
                try:
                    # Load only the header to check dimensions
                    with tifffile.TiffFile(payload['file_path']) as tif:
                        shape = tif.series[0].shape
                        # If any dimension is larger than 512, use patching
                        if any(dim > 512 for dim in shape):
                            need_patching = True
                            logging.info(f"Large input detected {shape}, will use patch-based processing")
                except Exception as e:
                    logging.warning(f"Could not check file dimensions: {e}")
            
            if need_patching:
                return self._invoke_with_patches(payload)
            else:
                # Standard invocation for smaller files
                predictions = requests.post(
                    url=self.target,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "inputs": payload,
                        },
                    ),
                    timeout=15,  # Longer timeout for image processing
                )
                return predictions.json()
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")
            return None
    
    def _invoke_with_patches(self, payload):
        """Process large volumes by breaking them into patches."""
        import requests
        
        try:
            # Load the full image
            if 'file_path' in payload:
                input_image = tifffile.imread(payload['file_path'])
            elif 'data' in payload and isinstance(payload['data'], np.ndarray):
                input_image = payload['data']
            else:
                raise ValueError("Input must contain either 'file_path' or 'data'")
            
            # Extract patches
            patches = extract_patches_from_volume(input_image, 
                                                 patch_size=self.patch_size, 
                                                 overlap=self.patch_overlap)
            logging.info(f"Processing large input in {len(patches)} patches")
            
            # Process each patch
            processed_patches = []
            for i, (patch, coords) in enumerate(patches):
                # Create payload for this patch
                patch_payload = {
                    'data': patch.tolist() if isinstance(patch, np.ndarray) else patch,
                }
                
                # Make request for this patch
                response = requests.post(
                    url=self.target,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"inputs": patch_payload}),
                    timeout=10,
                )
                
                # Parse response
                patch_result = response.json()
                
                if isinstance(patch_result, list):
                    patch_result = patch_result[0]  # Take first result if it's a list
                
                # Extract the prediction data
                if 'fused_view' in patch_result:
                    output_patch = np.array(patch_result['fused_view'])
                elif 'output_path' in patch_result:
                    output_patch = tifffile.imread(patch_result['output_path'])
                else:
                    raise ValueError(f"Unexpected response format: {patch_result}")
                
                # Store processed patch with coordinates
                processed_patches.append((output_patch, coords))
                
                if (i + 1) % 5 == 0:
                    logging.info(f"Processed {i+1}/{len(patches)} patches")
            
            # Stitch patches back together
            output_volume = stitch_patches_to_volume(processed_patches, 
                                                   input_image.shape,
                                                   patch_size=self.patch_size,
                                                   overlap=self.patch_overlap)
            
            # Save the stitched result to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                output_path = tmp.name
                tifffile.imwrite(output_path, output_volume)
            
            # Compute N-SSIM if ground truth is available
            n_ssim = None
            if 'ground_truth_path' in payload and payload['ground_truth_path']:
                try:
                    ground_truth = tifffile.imread(payload['ground_truth_path'])
                    n_ssim = compute_n_ssim(output_volume, ground_truth, input_image)
                except Exception as e:
                    logging.warning(f"Error computing N-SSIM: {e}")
            
            # Return the final result
            return {
                'predictions': [{
                    'output_path': output_path,
                    'n_ssim': n_ssim
                }]
            }
            
        except Exception as e:
            logging.exception(f"Error in patch-based processing: {e}")
            return None

    def deploy(self, model_uri: str, model_version: str) -> None:
        """Not Implemented.

        Deploying a model is not applicable when serving the model directly.
        """
        logging.info(
            "Deploy not implemented for Local backend. "
            "Use 'mlflow models serve' to serve the model."
        )


class JsonLines(Backend):
    """JsonLines backend implementation.

    A model with this backend will store production data in a JSON Lines formatted file.
    Each line in the file is a valid JSON object representing an input request and its 
    corresponding prediction.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize backend using the supplied configuration.

        If the configuration is not provided, the class will attempt to read the
        configuration from environment variables.
        """
        self.target = (
            config.get("target", "http://127.0.0.1:8080/invocations")
            if config
            else "http://127.0.0.1:8080/invocations"
        )
        
        # File path for the JSON Lines storage
        self.storage_file = "fusemycell.jsonl"

        if config:
            self.storage_file = config.get("storage_file", self.storage_file)
        else:
            self.storage_file = os.getenv("MODEL_BACKEND_STORAGE_FILE", self.storage_file)

        # Data directory for storing TIFF files
        self.data_dir = os.path.join(os.path.dirname(self.storage_file), "tiff_data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Make sure the directory exists
        storage_dir = os.path.dirname(self.storage_file)
        if storage_dir and not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
            
        # Configure patch settings
        if config and 'patch_config' in config:
            patch_config = config['patch_config']
            self.patch_size = patch_config.get('patch_size', (32, 64, 64))
            self.patch_overlap = patch_config.get('patch_overlap', (8, 16, 16))
        else:
            self.patch_size = (32, 64, 64)
            self.patch_overlap = (8, 16, 16)
        logging.info("Backend storage file: %s", self.storage_file)
        logging.info("TIFF data directory: %s", self.data_dir)
        logging.info("Using patch size: %s", self.patch_size)
        logging.info("Using patch overlap: %s", self.patch_overlap)
    def load(self, limit: int = 100) -> pd.DataFrame | None:
        """Load production data from the JSON Lines file."""
        if not Path(self.storage_file).exists():
            logging.error("Storage file %s does not exist.", self.storage_file)
            return None

        try:
            # Read the JSON Lines file into a DataFrame
            data = []
            with open(self.storage_file, "r") as f:
                for line in f:
                    data.append(json.loads(line))
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # If there are no records, return an empty DataFrame
            if df.empty:
                return df
                
            # Filter for records that have ground truth
            df_with_gt = df[df["ground_truth_path"].notna()]
            
            # Return the most recent 'limit' records
            return df_with_gt.sort_values("prediction_date", ascending=False).head(limit)
            
        except Exception:
            logging.exception("Error loading data from the JSONL file.")
            return None

    def save(self, model_input: list | dict, model_output: list) -> None:
        """Save production data to a JSON Lines file.

        If the file doesn't exist, this function will create it.
        """
        logging.info("Storing production data in the JSONL file...")

        # Ensure model_input is a list
        if isinstance(model_input, dict):
            model_input = [model_input]

        try:
            # Process and save each prediction
            current_time = datetime.now(timezone.utc).isoformat()
            
            records = []
            for i, (input_data, output_data) in enumerate(zip(model_input, model_output)):
                record_uuid = str(uuid.uuid4())
                
                # Handle input data
                input_path = input_data.get('file_path')
                if not input_path:
                    # If no file path was provided, save the input data to a file
                    if 'data' in input_data and isinstance(input_data['data'], np.ndarray):
                        input_path = os.path.join(self.data_dir, f"{record_uuid}_input.tif")
                        tifffile.imwrite(input_path, input_data['data'])
                
                # Handle output data
                output_path = output_data.get('output_path')
                if output_path:
                    # Copy the output file to our data directory for long-term storage
                    new_output_path = os.path.join(self.data_dir, f"{record_uuid}_output.tif")
                    try:
                        # Read and then write the file (to avoid file system issues)
                        output_data_array = tifffile.imread(output_path)
                        tifffile.imwrite(new_output_path, output_data_array)
                        output_path = new_output_path
                    except Exception as e:
                        logging.error(f"Failed to copy output file: {e}")
                
                # Extract N-SSIM if available
                n_ssim = output_data.get('n_ssim')
                
                # Get ground truth path if available
                ground_truth_path = input_data.get('ground_truth_path')
                
                # Create record
                record = {
                    "uuid": record_uuid,
                    "input_path": input_path,
                    "output_path": output_path,
                    "n_ssim": n_ssim,
                    "ground_truth_path": ground_truth_path,
                    "prediction_date": current_time
                }
                
                records.append(record)
            
            # Append to the JSON Lines file
            with open(self.storage_file, "a") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")
                    
            logging.info(f"Saved {len(records)} prediction records to JSONL file")

        except Exception:
            logging.exception("There was an error saving production data to the JSONL file.")

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Label unlabeled samples by generating synthetic ground truth."""
        if not Path(self.storage_file).exists():
            logging.error("Storage file %s does not exist.", self.storage_file)
            return 0

        try:
            # Read the existing data
            with open(self.storage_file, "r") as f:
                lines = f.readlines()
            
            records = [json.loads(line) for line in lines]
            
            # Count how many records need labeling
            labeled_count = 0
            
            # Process each record
            for i, record in enumerate(records):
                if record.get("ground_truth_path") is None and record.get("output_path") is not None:
                    uuid_val = record["uuid"]
                    output_path = record["output_path"]
                    
                    if not os.path.exists(output_path):
                        continue
                    
                    try:
                        # Load the prediction
                        prediction = tifffile.imread(output_path)
                        
                        # Generate synthetic ground truth
                        ground_truth = self.get_fake_label(prediction, ground_truth_quality)
                        
                        # Save the synthetic ground truth
                        ground_truth_path = os.path.join(self.data_dir, f"{uuid_val}_ground_truth.tif")
                        tifffile.imwrite(ground_truth_path, ground_truth)
                        
                        # Update the record
                        record["ground_truth_path"] = ground_truth_path
                        records[i] = record
                        
                        labeled_count += 1
                    except Exception as e:
                        logging.error(f"Error labeling sample {uuid_val}: {e}")
            
            # Write back all records if we labeled anything
            if labeled_count > 0:
                with open(self.storage_file, "w") as f:
                    for record in records:
                        f.write(json.dumps(record) + "\n")
            
            logging.info(f"Labeled {labeled_count} samples")
            return labeled_count
            
        except Exception:
            logging.exception("There was an error labeling production data")
            return 0

    def invoke(self, payload: list | dict) -> dict | None:
        """Make a prediction request to the hosted model.
        
        For large volumes, this method automatically handles patch-based processing
        to avoid memory issues.
        """
        import requests

        logging.info('Running prediction on "%s"...', self.target)

        try:
            # Check if we need to process in patches
            need_patching = False
            if isinstance(payload, dict) and 'file_path' in payload:
                # Check file size or dimensions
                try:
                    # Load only the header to check dimensions
                    with tifffile.TiffFile(payload['file_path']) as tif:
                        shape = tif.series[0].shape
                        # If any dimension is larger than 512, use patching
                        if any(dim > 512 for dim in shape):
                            need_patching = True
                            logging.info(f"Large input detected {shape}, will use patch-based processing")
                except Exception as e:
                    logging.warning(f"Could not check file dimensions: {e}")
            
            if need_patching:
                return self._invoke_with_patches(payload)
            else:
                # Standard invocation for smaller files
                predictions = requests.post(
                    url=self.target,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(
                        {
                            "inputs": payload,
                        },
                    ),
                    timeout=15,  # Longer timeout for image processing
                )
                return predictions.json()
        except Exception:
            logging.exception("There was an error sending traffic to the endpoint.")
            return None
    
    def _invoke_with_patches(self, payload):
        """Process large volumes by breaking them into patches."""
        import requests
        
        try:
            # Load the full image
            if 'file_path' in payload:
                input_image = tifffile.imread(payload['file_path'])
            elif 'data' in payload and isinstance(payload['data'], np.ndarray):
                input_image = payload['data']
            else:
                raise ValueError("Input must contain either 'file_path' or 'data'")
            
            # Extract patches
            patches = extract_patches_from_volume(input_image, 
                                                 patch_size=self.patch_size, 
                                                 overlap=self.patch_overlap)
            logging.info(f"Processing large input in {len(patches)} patches")
            
            # Process each patch
            processed_patches = []
            for i, (patch, coords) in enumerate(patches):
                # Create payload for this patch
                patch_payload = {
                    'data': patch.tolist() if isinstance(patch, np.ndarray) else patch,
                }
                
                # Make request for this patch
                response = requests.post(
                    url=self.target,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"inputs": patch_payload}),
                    timeout=10,
                )
                
                # Parse response
                patch_result = response.json()
                
                if isinstance(patch_result, list):
                    patch_result = patch_result[0]  # Take first result if it's a list
                
                # Extract the prediction data
                if 'fused_view' in patch_result:
                    output_patch = np.array(patch_result['fused_view'])
                elif 'output_path' in patch_result:
                    output_patch = tifffile.imread(patch_result['output_path'])
                else:
                    raise ValueError(f"Unexpected response format: {patch_result}")
                
                # Store processed patch with coordinates
                processed_patches.append((output_patch, coords))
                
                if (i + 1) % 5 == 0:
                    logging.info(f"Processed {i+1}/{len(patches)} patches")
            
            # Stitch patches back together
            output_volume = stitch_patches_to_volume(processed_patches, 
                                                   input_image.shape,
                                                   patch_size=self.patch_size,
                                                   overlap=self.patch_overlap)
            
            # Save the stitched result to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                output_path = tmp.name
                tifffile.imwrite(output_path, output_volume)
            
            # Compute N-SSIM if ground truth is available
            n_ssim = None
            if 'ground_truth_path' in payload and payload['ground_truth_path']:
                try:
                    ground_truth = tifffile.imread(payload['ground_truth_path'])
                    n_ssim = compute_n_ssim(output_volume, ground_truth, input_image)
                except Exception as e:
                    logging.warning(f"Error computing N-SSIM: {e}")
            
            # Return the final result
            return {
                'predictions': [{
                    'output_path': output_path,
                    'n_ssim': n_ssim
                }]
            }
            
        except Exception as e:
            logging.exception(f"Error in patch-based processing: {e}")
            return None

    def deploy(self, model_uri: str, model_version: str) -> None:
        """Not Implemented.

        Deploying a model is not applicable when serving the model directly.
        """
        logging.info(
            "Deploy not implemented for JsonLines backend. "
            "Use 'mlflow models serve' to serve the model."
        )

class Sagemaker(Backend):
    """Sagemaker backend implementation.

    A model with this backend will be deployed to Sagemaker and will use S3
    to store production data.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize backend using the supplied configuration."""
        from mlflow.deployments import get_deploy_client

        self.target = config.get("target", "fusemycell") if config else "fusemycell"
        self.data_capture_uri = config.get("data-capture-uri", None) if config else None
        self.ground_truth_uri = config.get("ground-truth-uri", None) if config else None

        # Let's make sure the ground truth uri ends with a '/'
        self.ground_truth_uri = (
            self.ground_truth_uri.rstrip("/") + "/" if self.ground_truth_uri else None
        )

        self.assume_role = config.get("assume-role", None) if config else None
        self.region = config.get("region", "us-east-1") if config else "us-east-1"

        self.deployment_target_uri = (
            f"sagemaker:/{self.region}/{self.assume_role}"
            if self.assume_role
            else f"sagemaker:/{self.region}"
        )

        self.deployment_client = get_deploy_client(self.deployment_target_uri)

        # Data directory for storing temporary TIFF files
        self.temp_dir = tempfile.mkdtemp(prefix="sagemaker_fusemycell_")
        
        # Configure patch settings
        if config and 'patch_config' in config:
            patch_config = config['patch_config']
            self.patch_size = patch_config.get('patch_size', (32, 64, 64))
            self.patch_overlap = patch_config.get('patch_overlap', (8, 16, 16))
        else:
            self.patch_size = (32, 64, 64)
            self.patch_overlap = (8, 16, 16)
        
        logging.info("Target: %s", self.target)
        logging.info("Data capture URI: %s", self.data_capture_uri)
        logging.info("Ground truth URI: %s", self.ground_truth_uri)
        logging.info("Assume role: %s", self.assume_role)
        logging.info("Region: %s", self.region)
        logging.info("Deployment target URI: %s", self.deployment_target_uri)
        logging.info("Temporary directory: %s", self.temp_dir)
        logging.info("Using patch size: %s", self.patch_size)
        logging.info("Using patch overlap: %s", self.patch_overlap)

    def load(self, limit: int = 100) -> pd.DataFrame:
        """Load production data from an S3 bucket."""
        import boto3

        s3 = boto3.client("s3")
        data = self._load_collected_data(s3)

        if data.empty:
            return data

        # We want to return samples that have a ground truth label.
        data = data[data["ground_truth_path"].notna()]

        # We need to remove a few columns that are not needed for the monitoring tests
        # and return `limit` number of samples.
        if "date" in data.columns:
            data = data.drop(columns=["date"])
        if "event_id" in data.columns:
            data = data.drop(columns=["event_id"])

        # We want to return `limit` number of samples.
        return data.head(limit)

    def label(self, ground_truth_quality=0.8):
        """Label every unlabeled sample stored in S3.

        This function loads any unlabeled data from the location where Sagemaker stores
        the data captured by the endpoint and generates fake ground truth labels.
        """
        import json
        from datetime import datetime, timezone

        import boto3

        if self.ground_truth_uri is None:
            logging.error("Ground truth URI is not defined.")
            return 0

        s3 = boto3.client("s3")
        data = self._load_unlabeled_data(s3)

        logging.info("Loaded %s unlabeled samples from S3.", len(data))

        # If there are no unlabeled samples, we don't need to do anything else.
        if data.empty:
            return 0

        records = []
        for event_id, group in data.groupby("event_id"):
            # Generate synthetic ground truth for each sample
            for _, row in group.iterrows():
                output_path = row.get("output_path")
                
                if not output_path:
                    continue
                
                try:
                    # Download the output file locally
                    local_output_path = os.path.join(self.temp_dir, f"{event_id}_output.tif")
                    output_bucket, output_key = self._parse_s3_uri(output_path)
                    s3.download_file(output_bucket, output_key, local_output_path)
                    
                    # Load the prediction
                    prediction = tifffile.imread(local_output_path)
                    
                    # Generate synthetic ground truth
                    ground_truth = self.get_fake_label(prediction, ground_truth_quality)
                    
                    # Save the synthetic ground truth
                    local_gt_path = os.path.join(self.temp_dir, f"{event_id}_ground_truth.tif")
                    tifffile.imwrite(local_gt_path, ground_truth)
                    
                    # Upload to S3
                    gt_bucket = self.ground_truth_uri.split("/")[2]
                    upload_time = datetime.now(tz=timezone.utc)
                    gt_key = f"ground_truth/{event_id}/{upload_time:%Y/%m/%d/%H/%M%S}.tif"
                    s3.upload_file(local_gt_path, gt_bucket, gt_key)
                    
                    # Create record for the ground truth data
                    record = {
                        "groundTruthData": {
                            "path": f"s3://{gt_bucket}/{gt_key}",
                            "format": "TIFF",
                        },
                        "eventMetadata": {
                            "eventId": event_id,
                        },
                        "eventVersion": "0",
                    }
                    
                    records.append(json.dumps(record))
                except Exception as e:
                    logging.exception(f"Error generating ground truth for event {event_id}: {e}")

        # If we have records, upload them to S3
        if records:
            ground_truth_payload = "\n".join(records)
            upload_time = datetime.now(tz=timezone.utc)
            uri = (
                "/".join(self.ground_truth_uri.split("/")[3:])
                + f"{upload_time:%Y/%m/%d/%H/%M%S}.jsonl"
            )

            s3.put_object(
                Body=ground_truth_payload,
                Bucket=self.ground_truth_uri.split("/")[2],
                Key=uri,
            )

        return len(data)
    
    def save(self, model_input: list | dict, model_output: list) -> None:
        """Not implemented for Sagemaker.

        Models deployed on Sagemaker will use data capture to automatically store
        inference data. No need to manually save data.
        """
        logging.info("Save method not implemented for Sagemaker backend. Data capture is automatic.")
        
    def invoke(self, payload: list | dict) -> dict | None:
        """Make a prediction request to the Sagemaker endpoint."""
        logging.info('Running prediction on "%s"...', self.target)

        # Check if we need to process in patches
        need_patching = False
        if isinstance(payload, dict) and 'file_path' in payload:
            # Check file size or dimensions
            try:
                # Load only the header to check dimensions
                with tifffile.TiffFile(payload['file_path']) as tif:
                    shape = tif.series[0].shape
                    # If any dimension is larger than 512, use patching
                    if any(dim > 512 for dim in shape):
                        need_patching = True
                        logging.info(f"Large input detected {shape}, will use patch-based processing")
            except Exception as e:
                logging.warning(f"Could not check file dimensions: {e}")
        
        if need_patching:
            return self._invoke_with_patches(payload)
        else:
            # Standard invocation for smaller files
            try:
                response = self.deployment_client.predict(
                    self.target,
                    json.dumps(
                        {
                            "inputs": payload,
                        },
                    ),
                )
                
                # For FuseMyCell, the response structure will be different from penguin classifier
                # We return the full JSON response 
                logging.info("Received prediction response from endpoint")
                
                return response
            except Exception as e:
                logging.exception(f"Error invoking Sagemaker endpoint: {e}")
                return None
        
    def _invoke_with_patches(self, payload):
        """Process large volumes by breaking them into patches."""
        try:
            # Load the full image
            if 'file_path' in payload:
                input_image = tifffile.imread(payload['file_path'])
            elif 'data' in payload and isinstance(payload['data'], np.ndarray):
                input_image = payload['data']
            else:
                raise ValueError("Input must contain either 'file_path' or 'data'")
            
            # Extract patches
            patches = extract_patches_from_volume(input_image, 
                                                 patch_size=self.patch_size, 
                                                 overlap=self.patch_overlap)
            logging.info(f"Processing large input in {len(patches)} patches")
            
            # Process each patch
            processed_patches = []
            for i, (patch, coords) in enumerate(patches):
                # Create payload for this patch
                patch_payload = {
                    'data': patch.tolist() if isinstance(patch, np.ndarray) else patch,
                }
                
                # Make request for this patch
                response = self.deployment_client.predict(
                    self.target,
                    json.dumps({"inputs": patch_payload}),
                )
                
                # Parse response
                patch_result = response
                
                if isinstance(patch_result, list):
                    patch_result = patch_result[0]  # Take first result if it's a list
                
                # Extract the prediction data
                if 'fused_view' in patch_result:
                    output_patch = np.array(patch_result['fused_view'])
                elif 'output_path' in patch_result:
                    output_patch = tifffile.imread(patch_result['output_path'])
                else:
                    raise ValueError(f"Unexpected response format: {patch_result}")
                
                # Store processed patch with coordinates
                processed_patches.append((output_patch, coords))
                
                if (i + 1) % 5 == 0:
                    logging.info(f"Processed {i+1}/{len(patches)} patches")
            
            # Stitch patches back together
            output_volume = stitch_patches_to_volume(processed_patches, 
                                                   input_image.shape,
                                                   patch_size=self.patch_size,
                                                   overlap=self.patch_overlap)
            
            # Save the stitched result to a temporary file
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                output_path = tmp.name
                tifffile.imwrite(output_path, output_volume)
            
            # Compute N-SSIM if ground truth is available
            n_ssim = None
            if 'ground_truth_path' in payload and payload['ground_truth_path']:
                try:
                    ground_truth = tifffile.imread(payload['ground_truth_path'])
                    n_ssim = compute_n_ssim(output_volume, ground_truth, input_image)
                except Exception as e:
                    logging.warning(f"Error computing N-SSIM: {e}")
            
            # Return the final result
            return {
                'predictions': [{
                    'output_path': output_path,
                    'n_ssim': n_ssim
                }]
            }
            
        except Exception as e:
            logging.exception(f"Error in patch-based processing: {e}")
            return None
            
    def deploy(self, model_uri: str, model_version: str) -> None:
        """Deploy the model to Sagemaker.

        This function creates a new Sagemaker Model, Sagemaker Endpoint Configuration,
        and Sagemaker Endpoint to serve the latest version of the model.

        If the endpoint already exists, this function will update it with the latest
        version of the model.
        """
        from mlflow.exceptions import MlflowException

        deployment_configuration = {
            "instance_type": "ml.m4.xlarge",
            "instance_count": 1,
            "synchronous": True,
            # We want to archive resources associated with the endpoint that become
            # inactive as the result of updating an existing deployment.
            "archive": True,
            # Notice how we are storing the version number as a tag.
            "tags": {"version": model_version},
        }

        # If the data capture destination is defined, we can configure the Sagemaker
        # endpoint to capture data.
        if self.data_capture_uri is not None:
            deployment_configuration["data_capture_config"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": 100,
                "DestinationS3Uri": self.data_capture_uri,
                "CaptureOptions": [
                    {"CaptureMode": "Input"},
                    {"CaptureMode": "Output"},
                ],
                "CaptureContentTypeHeader": {
                    "CsvContentTypes": ["text/csv", "application/octect-stream"],
                    "JsonContentTypes": [
                        "application/json",
                        "application/octect-stream",
                    ],
                },
            }

        if self.assume_role:
            deployment_configuration["execution_role_arn"] = self.assume_role

        try:
            # Let's return the deployment with the name of the endpoint we want to
            # create. If the endpoint doesn't exist, this function will raise an
            # exception.
            deployment = self.deployment_client.get_deployment(self.target)

            # We now need to check whether the model we want to deploy is already
            # associated with the endpoint.
            if self._is_sagemaker_model_running(deployment, model_version):
                logging.info(
                    'Endpoint "%s" is already running model "%s".',
                    self.target,
                    model_version,
                )
            else:
                # If the model we want to deploy is not associated with the endpoint,
                # we need to update the current deployment to replace the previous model
                # with the new one.
                self._update_sagemaker_deployment(
                    deployment_configuration,
                    model_uri,
                    model_version,
                )
        except MlflowException:
            # If the endpoint doesn't exist, we can create a new deployment.
            self._create_sagemaker_deployment(
                deployment_configuration,
                model_uri,
                model_version,
            )
            
    def _get_boto3_client(self, service):
        """Return a boto3 client for the specified service.

        If the `assume_role` parameter is provided, this function will assume the role
        and return a new client with temporary credentials.
        """
        import boto3

        if not self.assume_role:
            return boto3.client(service)

        # If we have to assume a role, we need to create a new
        # Security Token Service (STS)
        sts_client = boto3.client("sts")

        # Let's use the STS client to assume the role and return
        # temporary credentials
        credentials = sts_client.assume_role(
            RoleArn=self.assume_role,
            RoleSessionName="fusemycell-session",
        )["Credentials"]

        # We can use the temporary credentials to create a new session
        # from where to create the client for the target service.
        session = boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

        return session.client(service)
        
    def _is_sagemaker_model_running(self, deployment, version):
        """Check if the model is already running in Sagemaker.

        This function will check if the current model is already associated with a
        running Sagemaker endpoint.
        """
        sagemaker_client = self._get_boto3_client(service="sagemaker")

        # Here, we're assuming there's only one production variant associated with
        # the endpoint. This code will need to be updated if an endpoint could have
        # multiple variants.
        variant = deployment.get("ProductionVariants", [])[0]

        # From the variant, we can get the ARN of the model associated with the
        # endpoint.
        model_arn = sagemaker_client.describe_model(
            ModelName=variant.get("VariantName"),
        ).get("ModelArn")

        # With the model ARN, we can get the tags associated with the model.
        tags = sagemaker_client.list_tags(ResourceArn=model_arn).get("Tags", [])

        # Finally, we can check whether the model has a `version` tag that matches
        # the model version we're trying to deploy.
        model = next(
            (
                tag["Value"]
                for tag in tags
                if (tag["Key"] == "version" and tag["Value"] == version)
            ),
            None,
        )

        return model is not None
        
    def _create_sagemaker_deployment(
        self,
        deployment_configuration,
        model_uri,
        model_version,
    ):
        """Create a new Sagemaker deployment using the supplied configuration."""
        logging.info(
            'Creating endpoint "%s" with model "%s"...',
            self.target,
            model_version,
        )

        self.deployment_client.create_deployment(
            name=self.target,
            model_uri=model_uri,
            flavor="python_function",
            config=deployment_configuration,
        )

    def _update_sagemaker_deployment(
        self,
        deployment_configuration,
        model_uri,
        model_version,
    ):
        """Update an existing Sagemaker deployment using the supplied configuration."""
        logging.info(
            'Updating endpoint "%s" with model "%s"...',
            self.target,
            model_version,
        )

        # If you wanted to implement a staged rollout, you could extend the deployment
        # configuration with a `mode` parameter with the value
        # `mlflow.sagemaker.DEPLOYMENT_MODE_ADD` to create a new production variant. You
        # can then route some of the traffic to the new variant using the Sagemaker SDK.
        self.deployment_client.update_deployment(
            name=self.target,
            model_uri=model_uri,
            flavor="python_function",
            config=deployment_configuration,
        )

    def _parse_s3_uri(self, uri):
        """Parse an S3 URI into bucket and key components."""
        if not uri.startswith('s3://'):
            raise ValueError(f"Invalid S3 URI: {uri}")
            
        parts = uri[5:].split('/', 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {uri}")
            
        return parts[0], parts[1]
        
    def _load_unlabeled_data(self, s3):
        """Load any unlabeled data from the specified S3 location.

        This function will load the data captured from the endpoint during inference
        that does not have a corresponding ground truth information.
        """
        data = self._load_collected_data(s3)
        return data if data.empty else data[data["ground_truth_path"].isna()]
        
    def _load_collected_data(self, s3):
        """Load data from the endpoint and merge it with its ground truth."""
        data = self._load_collected_data_files(s3)
        ground_truth = self._load_ground_truth_files(s3)

        if len(data) == 0:
            return pd.DataFrame()

        if len(ground_truth) > 0:
            # Merge based on event_id
            data = data.merge(
                ground_truth,
                on=["event_id"],
                how="left",
            )

        return data
        
    def _load_ground_truth_files(self, s3):
        """Load the ground truth data from the specified S3 location."""
        if not self.ground_truth_uri:
            return pd.DataFrame()

        df = self._load_files(s3, self.ground_truth_uri)
        if df is None or df.empty:
            return pd.DataFrame()

        processed_data = []
        for _, row in df.iterrows():
            try:
                gt_path = row["groundTruthData"]["path"]
                event_id = row["eventMetadata"]["eventId"]
                
                processed_data.append({
                    "event_id": event_id,
                    "ground_truth_path": gt_path
                })
            except (KeyError, TypeError) as e:
                logging.warning(f"Error processing ground truth record: {e}")
                
        return pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
        
    def _load_collected_data_files(self, s3):
        """Load the data captured from the endpoint during inference."""
        if not self.data_capture_uri:
            return pd.DataFrame()
            
        df = self._load_files(s3, self.data_capture_uri)
        if df is None:
            return pd.DataFrame()

        processed_data = []
        for _, row in df.iterrows():
            try:
                date = row["eventMetadata"]["inferenceTime"]
                event_id = row["eventMetadata"]["eventId"]
                
                input_data = json.loads(row["captureData"]["endpointInput"]["data"])
                output_data = json.loads(row["captureData"]["endpointOutput"]["data"])
                
                # Extract file paths if available
                input_path = None
                output_path = None
                n_ssim = None
                
                if isinstance(input_data, dict) and "file_path" in input_data:
                    input_path = input_data["file_path"]
                
                if isinstance(output_data, dict):
                    if "output_path" in output_data:
                        output_path = output_data["output_path"]
                    if "n_ssim" in output_data:
                        n_ssim = output_data["n_ssim"]
                
                processed_data.append({
                    "event_id": event_id,
                    "date": date,
                    "input_path": input_path,
                    "output_path": output_path,
                    "n_ssim": n_ssim
                })
            except (KeyError, json.JSONDecodeError) as e:
                logging.warning(f"Error processing capture data record: {e}")
        
        if not processed_data:
            return pd.DataFrame()
            
        result_df = pd.DataFrame(processed_data)
        return result_df.sort_values(by="date", ascending=False).reset_index(drop=True)
        
    def _load_files(self, s3, s3_uri):
        """Load every file stored in the supplied S3 location.

        This function will recursively return the contents of every file stored under
        the specified location. The function assumes that the files are stored in JSON
        Lines format.
        """
        if not s3_uri:
            return None
            
        bucket = s3_uri.split("/")[2]
        prefix = "/".join(s3_uri.split("/")[3:])

        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        files = [
            obj["Key"]
            for page in pages
            if "Contents" in page
            for obj in page["Contents"]
        ]

        if len(files) == 0:
            return None

        dfs = []
        for file in files:
            try:
                obj = s3.get_object(Bucket=bucket, Key=file)
                data = obj["Body"].read().decode("utf-8")

                json_lines = data.splitlines()

                # Parse each line as a JSON object and collect into a list
                records = []
                for line in json_lines:
                    if line.strip():  # Skip empty lines
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            logging.warning(f"Invalid JSON in file {file}")
                
                if records:
                    dfs.append(pd.DataFrame(records))
            except Exception as e:
                logging.warning(f"Error reading file {file}: {e}")

        # Concatenate all DataFrames into a single DataFrame
        return pd.concat(dfs, ignore_index=True) if dfs else None
    
class FuseMyCellWrapper:
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
        artifacts_dir = Path(model_path) / "artifacts"
        
        # Load patch configuration if available
        try:
            patch_config_path = artifacts_dir / "patch_config.json"
            if patch_config_path.exists():
                with open(patch_config_path, 'r') as f:
                    patch_config = json.load(f)
                self.patch_size = tuple(patch_config.get("patch_size", (32, 64, 64)))
                self.patch_overlap = tuple(patch_config.get("patch_overlap", (8, 16, 16)))
                self.percentile_min = patch_config.get("percentile_min", 2)
                self.percentile_max = patch_config.get("percentile_max", 98)
            else:
                # Default values
                self.patch_size = (32, 64, 64)
                self.patch_overlap = (8, 16, 16)
                self.percentile_min = 2
                self.percentile_max = 98
        except Exception as e:
            logging.warning(f"Error loading patch configuration: {e}")
            # Default values
            self.patch_size = (32, 64, 64)
            self.patch_overlap = (8, 16, 16)
            self.percentile_min = 2
            self.percentile_max = 98
        
        logging.info(f"Using patch size: {self.patch_size}, overlap: {self.patch_overlap}")
        
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
        
        This method handles large volumes by breaking them into patches.
        
        Args:
            context: MLflow model context
            model_input: List of dictionaries or a single dictionary containing input data
                Keys can include:
                - 'file_path': Path to input TIFF file
                - 'data': Input numpy array
                - 'ground_truth_path': Optional path to ground truth image
                
        Returns:
            List of dictionaries containing prediction results
        """
        # Ensure model_input is a list
        if not isinstance(model_input, list):
            model_input = [model_input]
        
        predictions = []
        for i, input_data in enumerate(model_input):
            try:
                # Process input (which could be a file path or numpy array)
                single_view = self._load_input(input_data)
                
                # Determine if we need patch-based processing
                large_volume = False
                for dim in single_view.shape:
                    if dim > 512:  # Arbitrary threshold for "large"
                        large_volume = True
                        break
                
                if large_volume:
                    logging.info(f"Processing large volume of shape {single_view.shape} using patches")
                    fused_view = self._process_volume_with_patches(single_view)
                else:
                    # For smaller volumes, process directly
                    fused_view = predict_fused_view(self.model, single_view)
                
                # Save prediction to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                    output_path = tmp.name
                    tifffile.imwrite(output_path, fused_view)
                
                # Compute N-SSIM if ground truth is available
                n_ssim = None
                if 'ground_truth_path' in input_data and input_data['ground_truth_path']:
                    try:
                        ground_truth = tifffile.imread(input_data['ground_truth_path'])
                        # Ensure ground_truth has correct dimensions
                        if len(ground_truth.shape) == 2:
                            ground_truth = ground_truth[np.newaxis, ...]
                        elif len(ground_truth.shape) == 4:
                            ground_truth = ground_truth[..., 0]
                        
                        n_ssim = compute_n_ssim(fused_view, ground_truth, single_view)
                    except Exception as e:
                        logging.warning(f"Error computing N-SSIM: {e}")
                
                predictions.append({
                    'output_path': output_path,
                    'n_ssim': n_ssim
                })
                
            except Exception as e:
                logging.error(f"Error processing input {i}: {e}")
                predictions.append({'error': str(e)})
        
        return predictions
    
    def _load_input(self, input_data):
        """Load and preprocess input data."""
        # Handle input from file path
        if 'file_path' in input_data:
            file_path = input_data['file_path']
            logging.info(f"Loading input from file: {file_path}")
            try:
                single_view = tifffile.imread(file_path)
                
                # Handle different dimensions
                if len(single_view.shape) == 2:  # Single 2D image
                    single_view = single_view[np.newaxis, ...]
                elif len(single_view.shape) == 4:  # Multiple channels
                    # For simplicity, just take the first channel
                    single_view = single_view[..., 0]
                
            except Exception as e:
                raise ValueError(f"Error loading input file: {str(e)}")
        elif 'data' in input_data:
            # Input is a numpy array
            single_view = input_data['data']
            if not isinstance(single_view, np.ndarray):
                raise ValueError(f"Input data must be a numpy array, got {type(single_view)}")
        else:
            raise ValueError("Input must contain either 'file_path' or 'data'")
        
        return single_view
    
    def _process_volume_with_patches(self, volume):
        """
        Process a large volume by breaking it into overlapping patches.
        
        Args:
            volume: 3D numpy array
            
        Returns:
            3D numpy array of predictions
        """
        # Extract patches
        patches = extract_patches_from_volume(volume, 
                                             patch_size=self.patch_size, 
                                             overlap=self.patch_overlap)
        logging.info(f"Extracted {len(patches)} patches")
        
        # Process each patch
        processed_patches = []
        for i, (patch, coords) in enumerate(patches):
            # Normalize the patch
            normalized_patch = percentile_normalization(patch, 
                                                      pmin=self.percentile_min,
                                                      pmax=self.percentile_max)
            
            # Convert to tensor and add batch and channel dimensions
            patch_tensor = torch.from_numpy(normalized_patch).float().unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                output_tensor = self.model(patch_tensor)
            
            # Convert back to numpy
            output_patch = output_tensor.cpu().squeeze().numpy()
            
            # Store processed patch with its coordinates
            processed_patches.append((output_patch, coords))
            
            if (i + 1) % 10 == 0:
                logging.info(f"Processed {i + 1}/{len(patches)} patches")
        
        # Stitch patches back together
        logging.info("Stitching patches together")
        output_volume = stitch_patches_to_volume(processed_patches,
                                               volume.shape,
                                               patch_size=self.patch_size,
                                               overlap=self.patch_overlap)
        
        return output_volume
    
class Mock(Backend):
    """Mock implementation of the Backend abstract class.

    This class is helpful for testing purposes to simulate access to
    a production backend with 3D image data.
    """

    def __init__(self, config: dict | None = None) -> None:
        """Initialize the mock backend."""
        self.temp_dir = tempfile.mkdtemp(prefix="mock_fusemycell_")
        logging.info(f"Mock backend initialized with temp directory: {self.temp_dir}")
        
        # Create a simple in-memory storage to simulate database
        self.storage = []
        
        # Track ground truth data
        self.ground_truth = {}
        
        # Configure patch settings
        if config and 'patch_config' in config:
            patch_config = config['patch_config']
            self.patch_size = patch_config.get('patch_size', (32, 64, 64))
            self.patch_overlap = patch_config.get('patch_overlap', (8, 16, 16))
        else:
            self.patch_size = (32, 64, 64)
            self.patch_overlap = (8, 16, 16)
        
    def load(self, limit: int = 100) -> pd.DataFrame | None:
        """Return fake data for testing purposes."""
        if not self.storage:
            # Return empty if no data
            return pd.DataFrame()
            
        # Create DataFrame from storage and filter for labeled data
        df = pd.DataFrame(self.storage)
        df_labeled = df[df["ground_truth_path"].notna()]
        
        # Return most recent entries up to limit
        return df_labeled.sort_values("prediction_date", ascending=False).head(limit)

    def save(self, model_input: list | dict, model_output: list) -> None:
        """Mock saving data to storage."""
        logging.info("Mock backend: Storing production data...")
        
        # Ensure model_input is a list
        if isinstance(model_input, dict):
            model_input = [model_input]
            
        # Current time for all records
        current_time = datetime.now(timezone.utc).isoformat()
        
        for i, (input_data, output_data) in enumerate(zip(model_input, model_output)):
            record_uuid = str(uuid.uuid4())
            
            # Handle input data
            input_path = input_data.get('file_path')
            if not input_path and 'data' in input_data and isinstance(input_data['data'], np.ndarray):
                input_path = os.path.join(self.temp_dir, f"{record_uuid}_input.tif")
                # Just pretend to save it, don't actually write the file for speed in testing
                
            # Handle output data (simulate paths without writing files)
            output_path = output_data.get('output_path')
            if not output_path:
                output_path = os.path.join(self.temp_dir, f"{record_uuid}_output.tif")
            
            # Extract N-SSIM if available
            n_ssim = output_data.get('n_ssim')
            
            # Create and store record
            record = {
                "uuid": record_uuid,
                "input_path": input_path,
                "output_path": output_path,
                "n_ssim": n_ssim,
                "ground_truth_path": None,  # Initially unlabeled
                "prediction_date": current_time
            }
            
            self.storage.append(record)
            
        logging.info(f"Mock backend: Saved {len(model_output)} records")

    def label(self, ground_truth_quality: float = 0.8) -> int:
        """Simulate labeling of unlabeled samples."""
        unlabeled = [record for record in self.storage if record["ground_truth_path"] is None]
        logging.info(f"Mock backend: Found {len(unlabeled)} unlabeled samples")
        
        if not unlabeled:
            return 0
            
        labeled_count = 0
        for record in unlabeled:
            uuid_val = record["uuid"]
            
            # Simulate creating ground truth
            ground_truth_path = os.path.join(self.temp_dir, f"{uuid_val}_ground_truth.tif")
            record["ground_truth_path"] = ground_truth_path
            
            # Remember the quality for this record
            self.ground_truth[uuid_val] = ground_truth_quality
            
            labeled_count += 1
            
        logging.info(f"Mock backend: Labeled {labeled_count} samples")
        return labeled_count

    def invoke(self, payload: list | dict) -> dict | None:
        """Simulate model prediction."""
        logging.info("Mock backend: Simulating model prediction...")
        
        # Create a generic mock response for any input
        mock_responses = []
        
        # Ensure payload is a list
        if isinstance(payload, dict):
            payload = [payload]
            
        for item in payload:
            # Create a fake output path
            output_path = os.path.join(self.temp_dir, f"{uuid.uuid4()}_prediction.tif")
            
            # Generate a random N-SSIM score between 0.5 and 0.9
            n_ssim = round(0.5 + 0.4 * random.random(), 4)
            
            mock_responses.append({
                "output_path": output_path,
                "n_ssim": n_ssim
            })
        
        return {"predictions": mock_responses}

    def deploy(self, model_uri: str, model_version: str) -> None:
        """Mock deployment."""
        logging.info(f"Mock backend: Simulating deployment of model {model_version} from {model_uri}")
        # Nothing to actually do for mock
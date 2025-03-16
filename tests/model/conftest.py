from unittest.mock import Mock, patch
import numpy as np
import pytest
import torch
from pathlib import Path
import sys
import importlib.util

# Import your model class
from tests import Model, Input, Output


@pytest.fixture
def mock_unet3d_module():
    """Create a mock UNet3D module for testing."""
    # Create a mock UNet3D class
    mock_unet = Mock()
    mock_unet.return_value = Mock()
    mock_unet.return_value.eval = Mock()
    
    # Create prediction output tensor
    output_tensor = torch.zeros((1, 1, 64, 128, 128))
    mock_unet.return_value.forward = Mock(return_value=output_tensor)
    
    return mock_unet


@pytest.fixture
def mock_torch_device():
    """Mock torch.device to always return CPU."""
    with patch('torch.device', return_value='cpu'):
        yield


@pytest.fixture
def mock_tifffile():
    """Mock tifffile for testing."""
    with patch('tifffile.imread') as mock_imread, patch('tifffile.imwrite') as mock_imwrite:
        # Create a mock 3D volume when reading a TIFF file
        mock_imread.return_value = np.zeros((64, 128, 128), dtype=np.float32)
        yield mock_imread, mock_imwrite


@pytest.fixture
def mock_temp_file():
    """Mock NamedTemporaryFile for testing."""
    mock_tempfile = Mock()
    mock_tempfile.name = '/tmp/test_output.tif'
    
    with patch('tempfile.NamedTemporaryFile') as mock_named_temp:
        mock_named_temp.return_value.__enter__.return_value = mock_tempfile
        yield mock_tempfile


@pytest.fixture
def model(monkeypatch, mock_unet3d_module, mock_torch_device, mock_tifffile, mock_temp_file):
    """Return a model instance with mocked dependencies."""
    model = Model()
    
    # Create a mock context with artifacts
    context = Mock()
    context.artifacts = {
        "model_dir": "/mock/model_dir",
        "code": "/mock/code_dir"
    }
    
    # Mock Path.exists to return True
    monkeypatch.setattr(Path, "exists", lambda _: True)
    
    # Mock sys.path operations
    original_path = sys.path.copy()
    monkeypatch.setattr(sys.path, "append", lambda _: None)
    
    # Mock importlib operations for UNet3D
    def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'unet3d':
            mock_module = Mock()
            mock_module.UNet3D = mock_unet3d_module
            return mock_module
        else:
            return importlib.__import__(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr("importlib.__import__", mock_import)
    
    # Mock torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    
    # Mock compute_n_ssim
    monkeypatch.setattr(model, "_compute_n_ssim", lambda *args: 0.85)
    
    # Load model context
    model.load_context(context)
    
    return model


@pytest.fixture
def sample_input_dict():
    """Return a sample input dictionary for testing predict method."""
    return {
        'file_path': '/tmp/test_input.tif', 
        'ground_truth_path': '/tmp/test_ground_truth.tif'
    }


@pytest.fixture
def sample_input_list():
    """Return a sample input list for testing predict method."""
    return [
        {'file_path': '/tmp/test_input1.tif'},
        {'file_path': '/tmp/test_input2.tif', 'ground_truth_path': '/tmp/test_ground_truth2.tif'}
    ]
import os
from unittest.mock import Mock, patch
import sys
import importlib
import torch

import pytest

from pipelines.inference.model import Model


@pytest.fixture
def context():
    """Return a mock context."""
    mock_context = Mock()
    mock_context.artifacts = {
        "model_dir": "/mock/model_dir",
        "code": "/mock/code_dir",
    }
    return mock_context


@pytest.fixture
def model(monkeypatch):
    """Return a model instance."""
    model = Model()
    
    # Mock UNet3D class
    mock_unet = Mock()
    
    # Mock imports and sys.path
    def mock_import(name, *args, **kwargs):
        if name == 'unet3d':
            mock_module = Mock()
            mock_module.UNet3D = mock_unet
            return mock_module
        return importlib.__import__(name, *args, **kwargs)
    
    monkeypatch.setattr(importlib, "import_module", mock_import)
    monkeypatch.setattr(sys.path, "append", Mock())
    
    # Mock Path.exists
    monkeypatch.setattr("pathlib.Path.exists", lambda _: True)
    
    # Mock torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {})
    
    # Mock torch.device
    monkeypatch.setattr(torch, "device", lambda x: "cpu")
    
    return model


def test_load_artifacts_loads_unet_model(model, context):
    model.load_context(context)
    assert model.model is not None


def test_device_is_set_correctly(model, context):
    with patch('torch.cuda.is_available', return_value=True):
        model.load_context(context)
        assert model.device == "cpu"  # Since we mocked torch.device to return "cpu"


def test_model_is_set_to_eval_mode(model, context):
    model.load_context(context)
    assert model.model.eval.called


def test_load_artifacts_handles_missing_context(model):
    result = model._load_artifacts(None)
    assert result is None


def test_code_path_added_to_sys_path(model, context):
    model.load_context(context)
    sys.path.append.assert_called_with(str(context.artifacts["code"].parent))


def test_load_artifacts_raises_error_if_model_dir_missing(model, context):
    context.artifacts = {"code": "/mock/code_dir"}  # No model_dir
    with pytest.raises(ValueError):
        model.load_context(context)


def test_load_artifacts_handles_import_error(model, context, monkeypatch):
    # Mock import to fail
    def mock_failed_import(*args, **kwargs):
        raise ImportError("Could not import module")
    
    monkeypatch.setattr(importlib, "import_module", mock_failed_import)
    
    with pytest.raises(ImportError):
        model.load_context(context)
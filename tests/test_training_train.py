import torch
import pytest

from common import build_unet3d_model


def test_build_unet3d_model_configures_correctly():
    input_shape = (64, 128, 128)
    model = build_unet3d_model(input_shape)
    
    # Check model type
    assert isinstance(model, torch.nn.Module)
    
    # Check input handling with a sample input
    sample_input = torch.zeros((1, 1, *input_shape))
    with torch.no_grad():
        output = model(sample_input)
    
    # Check output shape (should match input spatial dimensions)
    assert output.shape == (1, 1, *input_shape)


def test_build_unet3d_model_physics_option():
    # Test with physics option enabled
    input_shape = (64, 128, 128)
    model_with_physics = build_unet3d_model(input_shape, use_physics=True)
    model_without_physics = build_unet3d_model(input_shape, use_physics=False)
    
    # Both should be valid models
    assert isinstance(model_with_physics, torch.nn.Module)
    assert isinstance(model_without_physics, torch.nn.Module)


def test_create_model_initializes_model(training_run):
    data = training_run["create_model"].task.data
    assert hasattr(data, "model")
    assert isinstance(data.model, torch.nn.Module)


def test_create_model_initializes_optimizer(training_run):
    data = training_run["create_model"].task.data
    assert hasattr(data, "optimizer")
    assert isinstance(data.optimizer, torch.optim.Optimizer)


def test_train_model_creates_mlflow_run(training_run):
    data = training_run["train_model"].task.data
    assert hasattr(data, "mlflow_run_id")
    assert data.mlflow_run_id is not None


def test_train_model_stores_best_model(training_run):
    data = training_run["train_model"].task.data
    assert hasattr(data, "best_model_path")
    assert data.best_model_path is not None
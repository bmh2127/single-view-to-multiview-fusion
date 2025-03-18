def test_start_loads_dataset(training_run):
    data = training_run["start"].task.data
    
    # Check that the data loaders were created
    assert hasattr(data, "train_loader")
    assert hasattr(data, "val_loader")
    
    # Verify the data loaders have content
    assert data.train_loader is not None
    assert data.val_loader is not None



def test_start_creates_mlflow_run(training_run):
    # Check that MLflow tracking URI is set
    data = training_run["start"].task.data
    assert hasattr(data, "mlflow_tracking_uri")
    assert data.mlflow_tracking_uri is not None
    
    # Check logs for evidence of MLflow run creation
    logs = training_run["start"].task.stdout
    assert "MLflow tracking server" in logs


def test_start_sets_training_parameters(training_run):
    data = training_run["start"].task.data
    
    # Verify training parameters are set
    assert hasattr(data, "training_epochs")
    assert hasattr(data, "training_batch_size")
    assert hasattr(data, "learning_rate")
    assert hasattr(data, "n_ssim_threshold")
    
    # Parameters should have reasonable values
    assert data.training_epochs > 0
    assert data.training_batch_size > 0
    assert data.learning_rate > 0
    assert 0 <= data.n_ssim_threshold <= 1
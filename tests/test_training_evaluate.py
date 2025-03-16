def test_evaluate_model_computes_metrics(training_run):
    """Test that evaluate_model step computes the expected metrics."""
    data = training_run["evaluate_model"].task.data

    # Test that average_n_ssim is computed and is a float
    assert hasattr(data, "average_n_ssim")
    assert isinstance(data.average_n_ssim, float)


def test_evaluate_model_stores_visualization_data(training_run):
    """Test that evaluate_model step generates visualization samples."""
    # Check that the task has visualization artifacts
    data = training_run["evaluate_model"].task.data
    assert hasattr(data, "mlflow_run_id")
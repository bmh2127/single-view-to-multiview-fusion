# test_fixtures.py
import pytest
from metaflow import Run
import os

def test_mlflow_directory_fixture(mlflow_directory):
    """Test that mlflow_directory fixture returns a valid path."""
    assert isinstance(mlflow_directory, str)
    assert os.path.exists(os.path.dirname(mlflow_directory))
    print(f"MLflow directory: {mlflow_directory}")

def test_training_run_fixture(training_run):
    """Test that training_run fixture returns a valid Metaflow Run."""
    assert isinstance(training_run, Run)
    print(f"Training run ID: {training_run.id}")
    print(f"Training steps: {list(training_run.steps())}")
    # Print data from the start step to see what's happening
    if 'start' in training_run:
        print(f"Start step task successful: {training_run['start'].task.successful}")
        # Print any exceptions or logs if available
        if hasattr(training_run['start'].task, 'stderr'):
            print(f"Start step stderr: {training_run['start'].task.stderr}")

# def test_monitoring_run_fixture(monitoring_run):
#     """Test that monitoring_run fixture returns a valid Metaflow Run."""
#     assert isinstance(monitoring_run, Run)
#     print(f"Monitoring run ID: {monitoring_run.id}")
#     print(f"Monitoring steps: {list(monitoring_run.steps())}")
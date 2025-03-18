import tempfile
from pathlib import Path
import tifffile
import numpy as np
import pytest
from metaflow import Runner


@pytest.fixture(scope="session")
def mlflow_directory():
    temporal_directory = tempfile.gettempdir()
    return (Path(temporal_directory) / "mlflow").as_posix()


@pytest.fixture(scope="session")
def training_run(mlflow_directory):
    # Create temp dir with mock images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create several pairs of angle and fused images
        for i in range(5):
            angle_img = np.random.rand(32, 64, 64).astype(np.float32)
            angle_path = Path(temp_dir) / f"image_{i}_nucleus_angle.tif"
            tifffile.imwrite(angle_path, angle_img)
            
            fused_img = np.random.rand(32, 64, 64).astype(np.float32)
            fused_path = Path(temp_dir) / f"image_{i}_nucleus_fused.tif"
            tifffile.imwrite(fused_path, fused_img)
        
        # Run the flow with the temp directory as dataset-dir
        with Runner(
            "pipelines/training.py",
            environment="conda",
            show_output=True,
        ).run(
            mlflow_tracking_uri=mlflow_directory,
            training_epochs=1,
            n_ssim_threshold=0.5,
            dataset_dir=temp_dir,
            training_batch_size=2,
            test_mode=True
        ) as running:
            run = running.run
            
            # Log information about the run
            print(f"\nRun ID: {run.id}")
            print(f"Run successful: {run.successful}")
            
            # Get available steps
            available_steps = list(run.steps())
            print(f"Available steps: {available_steps}")
            
            # Expected steps based on your pipeline
            expected_steps = ['start', 'create_model', 'train_model', 'evaluate_model', 'register_model', 'end']
            
            # Check for missing steps
            missing_steps = [step for step in expected_steps if step not in available_steps]
            if missing_steps:
                print(f"WARNING: Missing steps: {missing_steps}")
            else:
                print("All expected steps are available")
                
            # Verify you can access each step's data
            for step_name in available_steps:
                try:
                    step = run[step_name]
                    if hasattr(step, 'task') and hasattr(step.task, 'data'):
                        print(f"Step '{step_name}' has task data")
                    else:
                        print(f"Step '{step_name}' missing task or data")
                except Exception as e:
                    print(f"Error accessing step '{step_name}': {e}")
            
            return run


@pytest.fixture(scope="session")
def monitoring_run():
    with Runner(
        "pipelines/monitoring.py",
        environment="conda",
        show_output=False,
    ).run(
        backend="backend.Mock",
    ) as running:
        return running.run

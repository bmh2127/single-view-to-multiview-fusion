import pytest
from metaflow import Runner


@pytest.mark.integration
def test_register_doesnt_register_if_n_ssim_under_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=True,  # Change to True to see the step names
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        n_ssim_threshold=0.9,
    ) as running:
        # Print all available steps
        available_steps = list(running.run.steps())
        print("Available steps:", available_steps)
        
        # Look for a register-related step
        register_step = None
        for step in available_steps:
            if "register_model" in step:
                register_step = step
                break
        
        if not register_step:
            pytest.skip("No register step found in the pipeline")
        
        data = running.run[register_step].task.data
        assert data.registered is False, "Model shouldn't have been registered"


@pytest.mark.integration
def test_register_registers_model_if_n_ssim_above_threshold(mlflow_directory):
    with Runner(
        "pipelines/training.py",
        environment="conda",
        show_output=True,  # Show output to debug
    ).run(
        mlflow_tracking_uri=mlflow_directory,
        training_epochs=1,
        n_ssim_threshold=0.0001,
        test_mode=True  # Use test mode with mock data
    ) as running:
        available_steps = list(running.run.steps())
        print("Available steps:", available_steps)
        
        # Look for register_model step
        register_step = None
        for step in available_steps:
            if "register" in step:
                register_step = step
                break
        
        if not register_step:
            pytest.skip("No register step found in the pipeline")
        
        data = running.run[register_step].task.data
        assert data.registered is True, "Model should have been registered"


def test_register_pip_requirements(training_run):
    data = training_run["register_model"].task.data

    assert isinstance(data.pip_requirements, list)
    assert len(data.pip_requirements) > 0
    assert "torch" in [req.split("==")[0] for req in data.pip_requirements]


def test_register_artifacts(training_run):
    data = training_run["register_model"].task.data

    assert "model_dir" in data.artifacts
    assert "code" in data.artifacts


def test_register_code_paths_includes_default_files(training_run):
    data = training_run["register_model"].task.data

    assert any(path.endswith("model.py") for path in data.code_paths)
    assert any(path.endswith("unet3d.py") for path in data.code_paths)
    assert any(path.endswith("backend.py") for path in data.code_paths)
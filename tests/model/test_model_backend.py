import json
import os
from unittest.mock import Mock, patch
import tempfile
import pytest

from pipelines.inference.model import Model


@pytest.fixture
def temp_config():
    """Create a temporary config file."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
        config_data = {"database": "fusemycell-test.db"}
        tmp.write(json.dumps(config_data).encode('utf-8'))
        tmp_path = tmp.name
    
    yield tmp_path
    
    # Clean up
    if os.path.exists(tmp_path):
        os.unlink(tmp_path)


def test_model_initializes_with_no_backend():
    model = Model()
    assert model.backend is None


def test_backend_is_loaded_from_environment_variable():
    with (
        patch.dict(os.environ, {"MODEL_BACKEND": "module.Backend"}),
        patch("importlib.import_module") as mock_import,
    ):
        mock_module = Mock()
        mock_module.Backend = Mock
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_import.assert_called_once_with("module")


def test_backend_is_initialized_with_config_file(temp_config):
    with (
        patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "module.Backend",
                "MODEL_BACKEND_CONFIG": str(temp_config),
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        mock_backend = Mock()
        mock_backend_class = Mock(return_value=mock_backend)

        mock_module = Mock()
        mock_module.Backend = mock_backend_class
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_backend_class.assert_called_once_with(
            config={"database": "fusemycell-test.db"},
        )


def test_backend_is_initialized_with_invalid_config():
    with (
        patch.dict(
            os.environ,
            {
                "MODEL_BACKEND": "module.Backend",
                "MODEL_BACKEND_CONFIG": "invalid.json",
            },
        ),
        patch("importlib.import_module") as mock_import,
    ):
        mock_backend = Mock()
        mock_backend_class = Mock(return_value=mock_backend)

        mock_module = Mock()
        mock_module.Backend = mock_backend_class
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)

        assert isinstance(model.backend, Mock)
        mock_backend_class.assert_called_once_with(config=None)


def test_backend_is_set_to_none_if_class_doesnt_exist():
    with patch.dict(os.environ, {"MODEL_BACKEND": "invalid"}):
        model = Model()
        model.load_context(context=None)
        assert model.backend is None

def test_backend_handles_exception_during_initialization():
    with (
        patch.dict(os.environ, {"MODEL_BACKEND": "module.Backend"}),
        patch("importlib.import_module") as mock_import,
    ):
        mock_module = Mock()
        mock_backend_class = Mock(side_effect=Exception("Initialization error"))
        mock_module.Backend = mock_backend_class
        mock_import.return_value = mock_module
        
        model = Model()
        model.load_context(context=None)
        
        assert model.backend is None
        mock_backend_class.assert_called_once()

def test_backend_logging_when_backend_set(caplog):
    with (
        patch.dict(os.environ, {"MODEL_BACKEND": "module.Backend"}),
        patch("importlib.import_module") as mock_import,
    ):
        mock_module = Mock()
        mock_module.Backend = Mock
        mock_import.return_value = mock_module

        model = Model()
        model.load_context(context=None)
        
        assert "Backend: module.Backend" in caplog.text
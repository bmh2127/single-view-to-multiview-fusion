from unittest.mock import Mock
import torch
import numpy as np
from model import Model
import pytest
from unittest.mock import patch

def test_predict_returns_empty_list_if_input_is_empty(model):
    assert model.predict(None, model_input=[]) == []


def test_predict_handles_dict_input(model, monkeypatch):
    # Test that a single dict gets converted to a list
    input_data = {'file_path': '/tmp/test.tif'}
    
    # Mock process_input and process_output
    mock_process_input = Mock(return_value=(torch.zeros((1, 1, 64, 128, 128)), np.zeros((64, 128, 128))))
    mock_process_output = Mock(return_value={'output_path': '/tmp/output.tif'})
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)
    
    result = model.predict(None, model_input=input_data)
    
    assert isinstance(result, list)
    assert len(result) == 1
    mock_process_input.assert_called_once()


def test_predict_handles_processing_error(model, monkeypatch):
    # Test handling of errors during processing
    mock_process_input = Mock(side_effect=Exception("Processing error"))
    monkeypatch.setattr(model, "process_input", mock_process_input)
    
    input_data = [{'file_path': '/tmp/test.tif'}]
    result = model.predict(None, model_input=input_data)
    
    assert len(result) == 1
    assert 'error' in result[0]
    assert 'Processing error' in result[0]['error']


def test_predict_calls_backend_save_if_available(model, monkeypatch):
    # Test that backend.save is called when backend is available
    mock_backend = Mock()
    model.backend = mock_backend
    
    mock_process_input = Mock(return_value=(torch.zeros((1, 1, 64, 128, 128)), np.zeros((64, 128, 128))))
    mock_process_output = Mock(return_value={'output_path': '/tmp/output.tif'})
    monkeypatch.setattr(model, "process_input", mock_process_input)
    monkeypatch.setattr(model, "process_output", mock_process_output)
    
    input_data = [{'file_path': '/tmp/test.tif'}]
    model.predict(None, model_input=input_data)
    
    mock_backend.save.assert_called_once()


def test_process_input_loads_file(model, monkeypatch):
    # Test processing of file input
    with patch('tifffile.imread') as mock_imread:
        mock_imread.return_value = np.zeros((64, 128, 128))
        
        result = model.process_input({'file_path': '/tmp/test.tif'})
        
        assert result is not None
        mock_imread.assert_called_once_with('/tmp/test.tif')


def test_process_input_handles_numpy_array(model):
    # Test processing of numpy array input
    input_data = {'data': np.zeros((64, 128, 128))}
    result = model.process_input(input_data)
    
    assert result is not None
    assert isinstance(result[0], torch.Tensor)


def test_process_input_handles_invalid_input(model):
    # Test handling of invalid input
    with pytest.raises(ValueError):
        model.process_input({'invalid_key': 'value'})


def test_process_output_includes_n_ssim_with_ground_truth(model, monkeypatch):
    # Test that N_SSIM is calculated when ground truth is provided
    with patch('tifffile.imread') as mock_imread:
        mock_imread.return_value = np.zeros((64, 128, 128))
        
        mock_compute_n_ssim = Mock(return_value=0.85)
        monkeypatch.setattr(model, "_compute_n_ssim", mock_compute_n_ssim)
        
        output = torch.zeros((1, 1, 64, 128, 128))
        input_volume = np.zeros((64, 128, 128))
        input_data = {'ground_truth_path': '/tmp/ground_truth.tif'}
        
        result = model.process_output(output, input_volume, input_data)
        
        assert 'n_ssim' in result
        assert result['n_ssim'] == 0.85


def test_process_output_saves_to_temp_file(model):
    # Test that output is saved to a temporary file
    with patch('tempfile.NamedTemporaryFile') as mock_temp:
        mock_file = Mock()
        mock_file.name = '/tmp/output.tif'
        mock_temp.return_value.__enter__.return_value = mock_file
        
        with patch('tifffile.imwrite') as mock_imwrite:
            output = torch.zeros((1, 1, 64, 128, 128))
            input_volume = np.zeros((64, 128, 128))
            input_data = {}
            
            result = model.process_output(output, input_volume, input_data)
            
            assert 'output_path' in result
            assert result['output_path'] == '/tmp/output.tif'
            mock_imwrite.assert_called_once()


def test_process_output_handles_error(model, monkeypatch):
    # Test handling of errors during output processing
    monkeypatch.setattr(torch.Tensor, "cpu", Mock(side_effect=Exception("Processing error")))
    
    output = torch.zeros((1, 1, 64, 128, 128))
    input_volume = np.zeros((64, 128, 128))
    input_data = {}
    
    result = model.process_output(output, input_volume, input_data)
    
    assert 'error' in result
    assert 'Processing error' in result['error']


def test_compute_n_ssim_handles_shape_mismatch(model, monkeypatch):
    # Test handling of shape mismatches in N_SSIM calculation
    monkeypatch.setattr(model, "_percentile_normalization", lambda x: x)
    
    # Create arrays with different shapes
    prediction = np.zeros((64, 128, 128))
    ground_truth = np.zeros((70, 130, 130))
    input_image = np.zeros((64, 128, 128))
    
    with patch('skimage.metrics.structural_similarity') as mock_ssim:
        mock_ssim.return_value = 0.75
        
        result = model._compute_n_ssim(prediction, ground_truth, input_image)
        
        assert isinstance(result, float)
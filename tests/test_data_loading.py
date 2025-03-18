import pytest
import os
import tempfile
import numpy as np
import tifffile
from pathlib import Path
from pipelines.common import prepare_data_loaders

@pytest.fixture
def mock_dataset():
    """Create a temporary directory with mock TIFF files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create several pairs of angle and fused images
        for i in range(5):
            # Create mock angle image (single view)
            angle_img = np.random.rand(32, 64, 64).astype(np.float32)
            angle_path = Path(temp_dir) / f"image_{i}_nucleus_angle.tif"
            tifffile.imwrite(angle_path, angle_img)
            
            # Create mock fused image
            fused_img = np.random.rand(32, 64, 64).astype(np.float32)
            fused_path = Path(temp_dir) / f"image_{i}_nucleus_fused.tif"
            tifffile.imwrite(fused_path, fused_img)
        
        yield temp_dir

def test_prepare_data_loaders(mock_dataset):
    """Test that prepare_data_loaders can load mock data."""
    # Load the dataset with minimal workers for testing
    data_loaders = prepare_data_loaders(
        dataset_dir=mock_dataset,
        batch_size=2,
        patch_size=(16, 32, 32),
        num_workers=0,  # Use 0 for easier debugging
    )
    
    # Verify the data loaders were created
    assert 'train' in data_loaders
    assert 'val' in data_loaders
    assert 'dataset' in data_loaders
    
    # Check that we can get data from the loaders
    train_loader = data_loaders['train']
    assert len(train_loader) > 0
    
    # Get a batch (should not raise any exceptions)
    batch = next(iter(train_loader))
    
    # Check batch structure
    assert 'input' in batch
    assert 'target' in batch
    assert 'file_path' in batch
    
    # Check shapes
    assert batch['input'].shape[0] == 2  # batch size
    assert batch['input'].shape[1] == 1  # channels
    assert batch['input'].shape[2:] == (16, 32, 32)  # z, y, x dimensions
    
    print(f"Successfully loaded dataset with {len(data_loaders['dataset'])} samples")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(data_loaders['val'])}")
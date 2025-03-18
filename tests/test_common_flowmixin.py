from pathlib import Path

import pandas as pd
import pytest
from metaflow import Run, Runner
from pipelines.common import FuseMyCellDataset
from torch.utils.data import DataLoader

@pytest.fixture(scope="module")
def metaflow_data():
    with Runner("tests/flowmixin_flow.py", show_output=True).run(
        dataset_dir="data",
        training_batch_size=4
    ) as running:
        run = Run(running.run.pathspec)
        # Check if the run failed
        if run.successful:
            return run.data
        else:
            # Return None or raise an exception
            return None


def test_load_dataset(metaflow_data):
    # Check for loader attributes
    assert hasattr(metaflow_data, "train_loader")
    assert hasattr(metaflow_data, "val_loader")
    
    # Test the loaders
    assert len(metaflow_data.train_loader) > 0
    assert len(metaflow_data.val_loader) > 0
    
    # Test the types
    assert isinstance(metaflow_data.train_loader, DataLoader)
    assert isinstance(metaflow_data.val_loader, DataLoader)

def test_load_dataset_returns_same_data(metaflow_data):
    data = metaflow_data
    data_2 = metaflow_data
    assert data == data_2
    

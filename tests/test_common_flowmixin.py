from pathlib import Path

import pandas as pd
import pytest
from metaflow import Run, Runner
from pipelines.common import FuseMyCellDataset
from torch.utils.data import DataLoader

@pytest.fixture(scope="module")
def metaflow_data():
    with Runner("tests/flowmixin_flow.py", show_output=False).run() as running:
        return Run(running.run.pathspec).data


def test_load_dataset(metaflow_data):
    data = metaflow_data
    assert "train" in data
    assert "val" in data
    assert "dataset" in data
    assert len(data["train"]) > 0
    assert len(data["val"]) > 0
    assert len(data["dataset"]) > 0
    assert isinstance(data["dataset"], FuseMyCellDataset)
    assert isinstance(data["train"], DataLoader)
    assert isinstance(data["val"], DataLoader)


def test_load_dataset_returns_same_data(metaflow_data):
    data = metaflow_data
    data_2 = metaflow_data
    assert data == data_2
    

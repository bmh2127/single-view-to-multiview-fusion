# Single-View to Multiview Fusion (FuseMyCell Project)

A deep learning approach to predict fused 3D multiview light sheet microscopy images from a single view.

## Project Overview

This project addresses the challenge of predicting fused 3D images using only one 3D view in light sheet microscopy, providing a solution to reduce photobleaching and phototoxicity while maintaining high-quality imaging.

Inspired by the France-BioImaging's "Fuse My Cells" challenge at ISBI 2025.

## Repository Structure

```
single-view-to-multiview-fusion/
├── data/
│   ├── raw/                  # Original multiview datasets
│   ├── processed/            # Preprocessed and aligned data
│   └── README.md             # Data documentation and sources
├── models/
│   ├── baseline/             # Baseline model implementation
│   ├── advanced/             # Advanced architecture implementations
│   └── checkpoints/          # Saved model weights
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Initial data analysis
│   ├── 02_baseline_results.ipynb     # Baseline model evaluation
│   └── 03_advanced_model.ipynb       # Advanced model development
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py  # Data preprocessing functions
│   │   └── dataset.py        # PyTorch dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet3d.py         # 3D U-Net implementation
│   │   ├── transformer.py    # Transformer-based architecture
│   │   └── physics.py        # Physics-informed components
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py          # Training loops
│   │   └── evaluate.py       # Evaluation functions
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py  # Visualization utilities
│       └── metrics.py        # Performance metrics
├── experiments/
│   ├── config/               # Configuration files
│   └── logs/                 # Training logs and MLflow data
├── results/
│   ├── figures/              # Generated figures
│   └── predictions/          # Model predictions
├── scripts/
│   ├── download_data.sh      # Data download script
│   ├── preprocess.py         # Data preprocessing script
│   └── train_model.py        # Model training script
├── tests/                    # Unit tests
├── .gitignore                # Git ignore file
├── environment.yml           # Conda environment file
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup file
└── README.md                 # Main project documentation
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/single-view-to-multiview-fusion.git
cd single-view-to-multiview-fusion

# Create and activate conda environment
conda env create -f environment.yml
conda activate fusion-env

# Install the package in development mode
pip install -e .
```

## Usage

### Data Preparation

```bash
# Download sample dataset
bash scripts/download_data.sh

# Preprocess the data
python scripts/preprocess.py --input data/raw --output data/processed
```

### Training

```bash
# Train baseline model
python scripts/train_model.py --config experiments/config/baseline.yml

# Train advanced model
python scripts/train_model.py --config experiments/config/advanced.yml
```

### Evaluation

```bash
# Evaluate model and generate predictions
python scripts/evaluate.py --model-path models/checkpoints/best_model.pth --data-path data/processed/test
```

## Results

[Include visualization examples and performance metrics here]

## Approach and Methodology

This project explores several approaches to the single-view to multiview fusion problem:

1. **Baseline**: 3D U-Net architecture with skip connections
2. **Advanced**: 
   - View-Synthesis Transformer Architecture
   - Physics-Informed Neural Network
   - Multi-Stage Pipeline with Uncertainty Estimation

[Include more details about architecture, training strategy, etc.]

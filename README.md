# FuseMyCell Project

A deep learning approach for predicting fused 3D multiview light sheet microscopy images from a single view.

## Project Overview

This project addresses the challenge posed by France-BioImaging's "Fuse My Cells" challenge at ISBI 2025: predicting a fused 3D microscopy image from a single view. The goal is to enhance image quality and resolution while reducing photobleaching and phototoxicity in light sheet microscopy.

## Key Features

- 3D U-Net architecture for single-view to multiview fusion
- Physics-informed neural network option for enhanced accuracy
- MLflow integration for experiment tracking and model management
- Metaflow for workflow orchestration
- Implementation of the challenge evaluation metrics (N_SSIM)
- Memory-efficient patch-based training for local development on laptops/MacBooks

## Repository Structure

```
fusemycell/
├── data/
│   ├── raw/                  # Original multiview datasets
│   └── processed/            # Preprocessed and aligned data
├── patches/                  # Extracted patches for memory-efficient training
├── models/
│   └── checkpoints/          # Saved model weights
├── mlruns/                   # MLflow run data
├── artifacts/                # MLflow artifacts
├── common.py                 # Common utilities for the project
├── pipelines/
│   ├── training.py           # Metaflow training pipeline
│   ├── inference/            # Inference code and deployment
│   ├── local/                # Local development utilities
│   │   ├── analyze_tiff.py   # TIFF analysis script
│   │   ├── extract_patches.py # Patch extraction for memory efficiency 
│   │   └── mac_train.py      # MacBook-optimized training script
│   └── utils/                # Shared utilities
│       ├── memory_monitor.py # Memory monitoring utility
│       └── patch_dataset.py  # Patch-based dataset implementation
├── start_mlflow.sh           # Script to start MLflow server
└── README.md                 # Project documentation
```

## Installation

### Prerequisites

- Python 3.12.8
- CUDA-compatible GPU (recommended) or Apple Silicon Mac with MPS

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fusemycell.git
   cd fusemycell
   ```

2. Set up a conda environment:
   ```bash
   conda create -n fusemycell python=3.12.8
   conda activate fusemycell
   ```

3. Install dependencies:
   ```bash
   pip install metaflow mlflow torch tifffile scikit-image cellpose matplotlib pandas numpy scikit-learn scipy
   ```

## Usage

### Starting the MLflow Server

Before running any experiments, start the MLflow tracking server:

```bash
chmod +x start_mlflow.sh
./start_mlflow.sh 5000
```

This will start the MLflow UI at http://127.0.0.1:5000

### Downloading and Preparing Data

Download the light sheet microscopy datasets using the provided URLs:

```bash
# Create data directories
mkdir -p data

# Download data (replace with your data download script)
python download_biostudies.py --output-dir data
```

### Training Workflows

#### Cloud Training (Full Dataset)

For full training on cloud infrastructure with high-performance GPUs:

```bash
# Basic training
python -m pipelines.training run

# With custom parameters
python -m pipelines.training run \
  --dataset-dir data \
  --training-epochs 100 \
  --learning-rate 0.0005 \
  --patch-size 64,128,128 \
  --use-physics True
```

#### Local Development (MacBook/Laptop)

For local development and testing on resource-constrained machines:

1. **Analyze your TIFF files** to understand their characteristics:
   ```bash
   python -m pipelines.local.analyze_tiff data/image_0_nucleus_angle.tif
   ```

2. **Extract focused patches** from large TIFFs for memory-efficient training:
   ```bash
   python -m pipelines.local.extract_patches \
     --data-dir data \
     --output-dir patches \
     --patch-size 32,64,64 \
     --z-range 50,175 \
     --focus-on-signal \
     --normalize
   ```

3. **Train on the extracted patches** using an MPS-optimized script:
   ```bash
   python -m pipelines.local.mac_train \
     --data-dir patches \
     --output-dir models \
     --batch-size 2 \
     --accumulation-steps 4 \
     --use-amp \
     --monitor-interval 30
   ```

### Viewing Results

Open the MLflow UI in your browser at http://127.0.0.1:5000 to view experiment results, compare runs, and download trained models.

## Evaluation Metrics

This project implements the evaluation metrics specified in the FuseMyCell challenge:

1. **N_SSIM (Normalized Structural Similarity Index)**: Measures the improvement in structural similarity between the predicted fusion and the ground truth, compared to the input single view.

The normalization is computed as:
```
N_SSIM = (prediction_ssim - reference_ssim) / (1 - reference_ssim)
```

Where:
- `prediction_ssim` is the SSIM between the predicted fusion and the ground truth
- `reference_ssim` is the SSIM between the input single view and the ground truth

## Model Architecture

The core model is a 3D U-Net with the following specifications:

- Input: Single-view 3D volume (1 channel)
- Output: Fused 3D volume (1 channel)
- Architecture: Encoder-decoder with skip connections
- Optional physics-informed constraints

## Memory Efficiency Features

For large 3D microscopy images, we've implemented several memory optimization techniques:

- **Patch-based processing**: Extract smaller 3D patches from large volumes
- **Signal-focused extraction**: Concentrate patches on regions with meaningful content
- **Gradient accumulation**: Simulate larger batch sizes with less memory
- **MPS acceleration**: Use Apple Silicon GPU acceleration on MacBooks
- **Memory monitoring**: Track memory usage during training to avoid OOM errors

## Future Work

- Implement N_IOU metrics with Cellpose segmentation
- Add multi-view training option (using 2+ views as input)
- Explore Transformer-based architectures for better long-range dependencies
- Optimize inference speed for real-time applications

## Acknowledgments

- France-BioImaging for the challenge concept and dataset
- The EBI BioStudies repository for providing the data
- The open-source community for tools like PyTorch, MLflow and Metaflow
# UMF-Stomata

Official code repository for the paper **"UMF-Stomata: An Unsupervised Multi-Focus Fusion Framework for Microscopic Stomatal Phenotyping"**.

This project provides the model definition, inference pipeline, pretrained weights, and example image stacks used for multi-focus microscopic stomatal image fusion. 

## Overview

UMF-Stomata is a PyTorch-based framework for unsupervised multi-focus fusion on microscopic stomatal image stacks. The goal is to fuse a sequence of images with different focus depths into a single all-in-focus result that is more suitable for downstream stomatal phenotyping analysis.

## Repository Structure

```text
UMF-Stomata/
|- args_fusion.py      # configuration for model and inference
|- datasets.py         # dataset loading utilities
|- net.py              # AIFNet network definition
|- ssim.py             # SSIM-related functions
|- test.py             # inference / fusion script
|- utils.py            # helper functions
|- best.pth            # pretrained model weights
|- data/               # example multi-focus image stacks
```

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- numpy
- Pillow
- opencv-python

Install dependencies with:

```bash
pip install torch torchvision numpy pillow opencv-python
```

## Data Format

Each sample is stored in a separate folder containing a stack of multi-focus images. For example:

```text
data/
|- 000000/
|  |- 000000_00.jpg
|  |- 000000_01.jpg
|  |- ...
|  |- 000000_20.jpg
```

Default settings in the current codebase:

- Number of images per stack: `n_stack = 21`
- Model checkpoint: `best.pth`
- Inference patch size: `128`

## Inference

1. Prepare the input image stacks.
2. Update the input and output paths in `test.py` if needed.
3. Check runtime settings in `args_fusion.py`.
4. Run:

```bash
python test.py
```

The script will generate fused output images for each stack.

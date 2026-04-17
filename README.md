# Retinal Vessel Segmentation with U-Net on DRIVE

This repository contains a PyTorch implementation of retinal vessel segmentation using the U-Net architecture on the DRIVE dataset.

## Project Structure

```text
retinal-vessel-segmentation-unet-drive/
├── data/
│   └── DRIVE/                # DRIVE dataset
├── docs/
│   ├── experiment_description.docx
│   └── report_draft_group7.docx
├── src/
│   └── train_unet_drive.py   # Training and inference script
└── README.md
```

## Overview

The project focuses on semantic segmentation of retinal blood vessels. The implementation includes:

- DRIVE dataset loading
- U-Net model construction
- Training and validation
- Segmentation metrics calculation
- Test prediction image export

## Requirements

Install the required packages with:

```powershell
pip install torch torchvision pillow numpy
```

## Run

From the repository root, run:

```powershell
python .\src\train_unet_drive.py
```

You can also specify custom hyperparameters:

```powershell
python .\src\train_unet_drive.py --epochs 50 --batch-size 2 --lr 0.001
```

## Notes

- The current DRIVE test folder in this workspace does not contain `1st_manual` annotations.
- Therefore, validation metrics are computed on the training split validation subset.
- The test set is used for prediction export only.

## Suggested GitHub Repository Name

`retinal-vessel-segmentation-unet-drive`

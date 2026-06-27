# Retinal Vessel Segmentation with U-Net on DRIVE

[中文](#中文说明) | [English](#english)

A compact PyTorch project for retinal vessel segmentation on the DRIVE dataset, built around the classic U-Net architecture and organized for coursework, reproducibility, and portfolio presentation.

## 中文说明

### 项目简介

本项目基于 `PyTorch` 实现了一个面向 `DRIVE` 数据集的视网膜血管分割实验，核心模型为经典的 `U-Net`。项目包含数据读取、模型构建、训练验证、分割指标评估和测试集预测结果导出等完整流程，适合作为医学图像分割课程实验、作业展示或入门项目参考。

如果你希望把它作为 GitHub 作品集项目展示，这个仓库可以概括为：一个结构清晰、可直接运行、适合课程展示与医学图像分割入门的 U-Net 实践项目。

### 项目亮点

- 基于 `U-Net` 的视网膜血管语义分割实现
- 适配标准 `DRIVE` 数据集目录结构
- 包含训练、验证和测试预测的完整流程
- 输出 `Precision`、`Recall`、`F1-Score`、`IoU`、`Accuracy`
- 保留精简文档说明，便于课程展示与代码理解

### 项目结构

```text
retinal-vessel-segmentation-unet-drive/
|-- data/
|   `-- DRIVE/                         # DRIVE dataset
|-- docs/
|   `-- assignment_notes.md           # 代码说明与实验备注
|-- src/
|   `-- train_unet_drive.py           # 训练与预测主脚本
|-- LICENSE
|-- requirements.txt
|-- .gitignore
`-- README.md
```

### 环境依赖

建议使用 Python 3.9 及以上版本，并执行：

```powershell
pip install -r requirements.txt
```

### 运行方法

在项目根目录下执行：

```powershell
python .\src\train_unet_drive.py
```

也可以自定义训练参数：

```powershell
python .\src\train_unet_drive.py --epochs 50 --batch-size 2 --lr 0.001
```

### 输出内容

脚本运行后会自动完成以下任务：

- 加载 `DRIVE` 训练集与测试集
- 将训练集划分为训练子集和验证子集
- 训练 `U-Net` 模型并保存最佳权重
- 在验证集上计算分割指标
- 导出测试集预测图像到 `outputs/predictions`

### 说明

- 当前仓库中的 `DRIVE/test` 不包含 `1st_manual` 人工标注。
- 因此，代码默认在验证集上统计分割指标，在测试集上仅保存预测结果。
- 为了让仓库更轻量、更适合公开展示，`.docx` 文件不纳入版本管理。
- `.idea`、缓存文件和输出结果已通过 `.gitignore` 排除。

### 适用场景

- 人工智能或医学图像处理课程实验
- U-Net 语义分割入门实践
- GitHub 项目展示与课程作业归档

---

## English

### Overview

This project provides a `PyTorch` implementation of retinal vessel segmentation on the `DRIVE` dataset using the classic `U-Net` architecture. It covers the full experimental pipeline, including dataset loading, model construction, training, validation, metric evaluation, and test prediction export.

As a portfolio-ready GitHub repository, this project is designed to be clean, runnable, and easy to understand for coursework and introductory medical image segmentation practice.

### Highlights

- U-Net-based retinal vessel semantic segmentation
- Compatible with the standard DRIVE folder structure
- End-to-end workflow for training, validation, and inference
- Reports `Precision`, `Recall`, `F1-Score`, `IoU`, and `Accuracy`
- Includes lightweight notes for academic reference and code explanation

### Project Structure

```text
retinal-vessel-segmentation-unet-drive/
|-- data/
|   `-- DRIVE/                         # DRIVE dataset
|-- docs/
|   `-- assignment_notes.md           # Notes for the implementation
|-- src/
|   `-- train_unet_drive.py           # Main training and inference script
|-- LICENSE
|-- requirements.txt
|-- .gitignore
`-- README.md
```

### Requirements

It is recommended to use Python 3.9 or later. Install dependencies with:

```powershell
pip install -r requirements.txt
```

### Usage

Run the project from the repository root:

```powershell
python .\src\train_unet_drive.py
```

You may also set custom hyperparameters:

```powershell
python .\src\train_unet_drive.py --epochs 50 --batch-size 2 --lr 0.001
```

### What the Script Does

After execution, the script will:

- Load the DRIVE training and test sets
- Split the training data into training and validation subsets
- Train the U-Net model and save the best checkpoint
- Compute segmentation metrics on the validation set
- Export predicted masks for the test set to `outputs/predictions`

### Notes

- The current `DRIVE/test` folder in this repository does not contain `1st_manual` annotations.
- As a result, segmentation metrics are computed on the validation subset instead of the test set.
- The test set is used for prediction export only.
- `.docx` files are intentionally excluded to keep the repository lightweight and presentation-friendly.
- `.idea`, cache files, and generated outputs are excluded via `.gitignore`.

### Use Cases

- AI and medical image processing coursework
- Introductory practice for U-Net-based segmentation
- Clean GitHub presentation for academic projects

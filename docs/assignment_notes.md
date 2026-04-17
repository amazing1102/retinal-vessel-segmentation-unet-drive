# 实验五代码说明

本目录提供了一个完整的 `PyTorch` 实验脚本，用于完成“基于 U-Net 的视网膜图像血管分割”实验。

## 文件说明

- `drive_unet_experiment.py`：主程序，包含数据集读取、U-Net 模型、训练、验证、测试与预测保存。

## 运行前准备

如果你的 Python 环境还没有安装 `PyTorch`，可以先执行：

```powershell
pip install torch torchvision pillow numpy
```

如果只打算用 CPU 跑实验，这样安装即可。

## 数据集路径

脚本默认读取以下数据集目录：

```text
C:\Users\33603\Desktop\课程作业\人工智能\第五次\DRIVE
```

默认输出目录：

```text
C:\Users\33603\Desktop\课程作业\人工智能\第五次\实验五代码\outputs
```

## 运行方式

在 PowerShell 中进入当前目录后执行：

```powershell
python .\drive_unet_experiment.py
```

也可以手动指定训练轮数、批大小和学习率：

```powershell
python .\drive_unet_experiment.py --epochs 50 --batch-size 2 --lr 0.001
```

## 运行结果

脚本会自动完成以下内容：

1. 读取 `training` 集并随机划分训练集与验证集。
2. 构建并训练 U-Net 模型。
3. 在验证集上选择 `F1-Score` 最优模型。
4. 在 `test` 集上计算 `Precision`、`Recall`、`F1-Score`、`mIoU`、`Accuracy`。
5. 将测试集预测结果保存到 `outputs/predictions`。

注意：你当前这份数据集的 `test` 目录没有 `1st_manual` 标注，因此脚本会使用验证集输出训练过程中的分割指标，而测试集默认只保存预测结果，不会伪造测试指标。

## 适合写进实验报告的说明

- 数据读取：从 `images`、`1st_manual` 和 `mask` 三类文件中分别读取原图、血管标注和视网膜感兴趣区域。
- 模型结构：采用经典 U-Net 编码器-解码器结构，并使用跳跃连接融合浅层与深层特征。
- 损失函数：使用 `BCEWithLogitsLoss + Dice Loss` 的组合损失，提高对血管细小结构的学习能力。
- 评价指标：输出 `Precision`、`Recall`、`F1-Score`、`mIoU` 和 `Accuracy`，满足实验文档要求。

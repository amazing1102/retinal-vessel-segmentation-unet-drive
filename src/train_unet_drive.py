import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DriveDataset(Dataset):
    def __init__(self, root: str, split: str, image_size: int = 512):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.image_dir = self.root / split / "images"
        self.mask_dir = self.root / split / "mask"
        self.label_dir = self.root / split / "1st_manual"
        self.image_paths = sorted(self.image_dir.glob("*.tif"))
        if not self.image_paths:
            raise FileNotFoundError(f"未找到图像文件: {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _manual_path(self, image_path: Path) -> Path:
        sample_id = image_path.stem.split("_")[0]
        return self.label_dir / f"{sample_id}_manual1.gif"

    def _roi_mask_path(self, image_path: Path) -> Path:
        sample_id = image_path.stem.split("_")[0]
        suffix = "training" if self.split == "training" else "test"
        return self.mask_dir / f"{sample_id}_{suffix}_mask.gif"

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)

    def _load_binary(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.NEAREST)
        array = np.asarray(image, dtype=np.float32)
        array = (array > 0).astype(np.float32)
        return torch.from_numpy(array).unsqueeze(0)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = self._load_rgb(image_path)
        roi_mask = self._load_binary(self._roi_mask_path(image_path))

        label_path = self._manual_path(image_path)
        has_label = label_path.exists()
        if label_path.exists():
            vessel_mask = self._load_binary(label_path)
        else:
            vessel_mask = torch.zeros_like(roi_mask)

        return {
            "image": image,
            "mask": vessel_mask,
            "roi_mask": roi_mask,
            "name": image_path.stem,
            "has_label": has_label,
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        current_channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_channels, feature))
            current_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx // 2]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


class DiceBCELoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        probs = probs.reshape(probs.size(0), -1)
        targets = targets.reshape(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        dice = (2 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        dice_loss = 1 - dice.mean()
        return 0.5 * bce_loss + 0.5 * dice_loss


def build_loaders(data_root: str, image_size: int, batch_size: int, seed: int):
    full_train = DriveDataset(data_root, split="training", image_size=image_size)
    train_size = int(len(full_train) * 0.8)
    val_size = len(full_train) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)
    test_set = DriveDataset(data_root, split="test", image_size=image_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def segmentation_metrics(
    logits: torch.Tensor, targets: torch.Tensor, roi_mask: torch.Tensor, threshold: float = 0.5
):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds * roi_mask
    targets = targets * roi_mask

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()
    tn = ((1 - preds) * (1 - targets)).sum().item()

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
    }


def average_metrics(metric_list):
    keys = metric_list[0].keys()
    return {key: sum(item[key] for item in metric_list) / len(metric_list) for key in keys}


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metric_list = []
    labeled_batches = 0
    for batch in loader:
        has_label = batch["has_label"]
        if hasattr(has_label, "all"):
            has_label = bool(has_label.all())
        else:
            has_label = all(has_label)

        if not has_label:
            continue

        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        roi_mask = batch["roi_mask"].to(device)
        logits = model(images)
        total_loss += criterion(logits, masks).item()
        metric_list.append(segmentation_metrics(logits, masks, roi_mask))
        labeled_batches += 1

    if not metric_list:
        return None

    metrics = average_metrics(metric_list)
    metrics["loss"] = total_loss / labeled_batches
    return metrics


@torch.no_grad()
def save_predictions(model, loader, device, output_dir: str):
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        images = batch["image"].to(device)
        roi_mask = batch["roi_mask"].to(device)
        names = batch["name"]
        logits = model(images)
        preds = (torch.sigmoid(logits) >= 0.5).float() * roi_mask

        pred_array = preds.squeeze(0).squeeze(0).cpu().numpy()
        pred_image = Image.fromarray((pred_array * 255).astype(np.uint8), mode="L")
        pred_image.save(output_path / f"{names[0]}_pred.png")


def main():
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="DRIVE 数据集 U-Net 视网膜血管分割实验")
    parser.add_argument(
        "--data-root",
        type=str,
        default=str(project_root / "data" / "DRIVE"),
        help="DRIVE 数据集根目录",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(project_root / "outputs"),
        help="模型权重与预测结果保存目录",
    )
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=2, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--image-size", type=int, default=512, help="输入图像尺寸")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前设备: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "best_unet_drive.pth"
    result_path = save_dir / "metrics.txt"

    train_loader, val_loader, test_loader = build_loaders(
        args.data_root, args.image_size, args.batch_size, args.seed
    )
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = DiceBCELoss()

    best_f1 = -1.0
    history_lines = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        if val_metrics is None:
            raise RuntimeError("验证集缺少人工标注，无法计算指标。")

        line = (
            f"Epoch [{epoch:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"precision={val_metrics['precision']:.4f} "
            f"recall={val_metrics['recall']:.4f} "
            f"f1={val_metrics['f1']:.4f} "
            f"iou={val_metrics['iou']:.4f} "
            f"accuracy={val_metrics['accuracy']:.4f}"
        )
        print(line)
        history_lines.append(line)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), model_path)

    print(f"最佳模型已保存到: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))

    test_metrics = evaluate(model, test_loader, criterion, device)
    if test_metrics is not None:
        test_line = (
            "\n测试集结果: "
            f"loss={test_metrics['loss']:.4f}, "
            f"precision={test_metrics['precision']:.4f}, "
            f"recall={test_metrics['recall']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"mIoU={test_metrics['iou']:.4f}, "
            f"accuracy={test_metrics['accuracy']:.4f}"
        )
    else:
        test_line = "\n测试集缺少人工标注，已跳过指标计算，只保存预测图。"
    print(test_line)
    history_lines.append(test_line)

    prediction_dir = save_dir / "predictions"
    save_predictions(model, test_loader, device, str(prediction_dir))
    print(f"测试集预测图已保存到: {prediction_dir}")

    result_path.write_text("\n".join(history_lines), encoding="utf-8")
    print(f"训练日志与指标已保存到: {result_path}")


if __name__ == "__main__":
    main()

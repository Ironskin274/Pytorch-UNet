"""
3D U-Net 基线训练脚本
基于完整体积的 BraTS2020 训练流程
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from unet import UNet3D
from utils.data_loading_3d import BraTS2020Dataset3D
from utils.metrics import compute_metrics, remap_labels
from utils.dice_score import dice_loss
from utils.visualization import create_segmentation_visualization
from utils.class_weights import load_class_weights, print_weights_info


DEFAULT_DATA_DIR = "/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/"
DEFAULT_TRAIN_LIST = "/data/ssd2/liying/Datasets/BraTS2020/train_list.txt"
DEFAULT_VALID_LIST = "/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt"
DEFAULT_CHECKPOINT_DIR = "./checkpoints_3d"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a 3D UNet baseline on BraTS2020")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="BraTS2020 数据根目录")
    parser.add_argument("--train-list", type=str, default=DEFAULT_TRAIN_LIST, help="训练集病例列表")
    parser.add_argument("--valid-list", type=str, default=DEFAULT_VALID_LIST, help="验证集病例列表")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="模型保存目录")

    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log-interval", type=int, default=10, help="每N个batch记录一次训练loss")
    parser.add_argument("--target-shape", type=int, nargs=3, default=(128, 128, 128), help="裁剪/填充后的体积大小")
    parser.add_argument("--num-workers", type=int, default=4, help="数据加载线程数")

    parser.add_argument("--modalities", type=str, nargs="+", default=["flair", "t1ce", "t2"], help="使用的模态列表，例如 flair t1ce t2")
    parser.add_argument("--fg-min-ratio", type=float, default=0.01, help="前景体素最小比例过滤")
    parser.add_argument("--jitter-ratio", type=float, default=0.1, help="bbox中心抖动比例(相对target_shape)")

    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    parser.add_argument("--amp", action="store_true", default=False, help="启用自动混合精度训练")
    parser.add_argument("--no-wandb", action="store_true", default=False, help="禁用 wandb 日志")
    parser.add_argument("--project", type=str, default="BraTS2020-UNet-3D", help="wandb 项目名称")
    parser.add_argument("--run-name", type=str, default=None, help="wandb 运行名称")
    
    parser.add_argument("--class-weights", type=str, default=None, help="类别权重文件路径（JSON格式），用于处理类别不平衡")
    parser.add_argument("--weight-method", type=str, default="effective_num_0.999", help="权重方法名称（默认: effective_num_0.999）")

    return parser.parse_args()


def prepare_dataloaders(args):
    train_set = BraTS2020Dataset3D(
        data_dir=args.data_dir,
        list_file=args.train_list,
        modalities=args.modalities,
        target_shape=tuple(args.target_shape),
        fg_min_ratio=args.fg_min_ratio,
        jitter_ratio=args.jitter_ratio,
    )
    val_set = BraTS2020Dataset3D(
        data_dir=args.data_dir,
        list_file=args.valid_list,
        modalities=args.modalities,
        target_shape=tuple(args.target_shape),
        fg_min_ratio=args.fg_min_ratio,
        jitter_ratio=0.0,  # 验证不抖动
    )

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    return train_loader, val_loader


def compute_loss(outputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module, n_classes: int, ce_weight: float = 0.3, dice_weight: float = 0.7) -> Dict[str, torch.Tensor]:
    ce_loss = criterion(outputs, targets)
    probs = torch.softmax(outputs, dim=1)
    one_hot = F.one_hot(targets, num_classes=n_classes).permute(0, 4, 1, 2, 3).float()
    dice = dice_loss(probs, one_hot, multiclass=True)
    # mimic reference: dice 更大权重
    loss = ce_weight * ce_loss + dice_weight * dice
    return {"loss": loss, "ce_loss": ce_loss.detach(), "dice_loss": dice.detach()}


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    criterion,
    device,
    n_classes,
    amp_enabled,
    log_interval,
    grad_clip,
    use_wandb,
    global_step,
):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        masks = remap_labels(masks)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            losses = compute_loss(outputs, masks, criterion, n_classes)

        scaler.scale(losses["loss"]).backward()
        scaler.unscale_(optimizer)
        if torch.isfinite(losses["loss"]).all():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses["loss"].item()
        preds = torch.argmax(outputs.detach(), dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        global_step += 1
        if use_wandb and (global_step % log_interval == 0):
            wandb.log({
                "train/batch_loss": float(losses["loss"].item()),
                "train/ce_loss": float(losses["ce_loss"].item()),
                "train/dice_loss": float(losses["dice_loss"].item()),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }, step=global_step)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = {k: float(v) for k, v in compute_metrics(all_preds, all_targets).items()}
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, metrics, global_step


@torch.no_grad()
def evaluate(model, loader, criterion, device, n_classes, amp_enabled, return_sample=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    sample_data = None

    for batch_idx, batch in enumerate(tqdm(loader, desc="Valid", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        masks = remap_labels(masks)

        with autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            losses = compute_loss(outputs, masks, criterion, n_classes)

        total_loss += losses["loss"].item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())

        if return_sample and sample_data is None:
            sample_data = (images[0:1].cpu(), masks[0:1].cpu(), preds[0:1].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = {k: float(v) for k, v in compute_metrics(all_preds, all_targets).items()}
    avg_loss = total_loss / max(len(loader), 1)
    if return_sample:
        return avg_loss, metrics, sample_data
    return avg_loss, metrics


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    use_wandb = not args.no_wandb
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(project=args.project, name=args.run_name, config=vars(args), reinit=True)
        logging.info("W&B run initialized: %s", wandb_run.name)
        logging.info("W&B url: %s", wandb_run.url)

    train_loader, val_loader = prepare_dataloaders(args)

    model = UNet3D(n_channels=len(args.modalities), n_classes=4, base_channels=32, bilinear=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 加载类别权重（如果指定）
    if args.class_weights:
        print_weights_info(args.class_weights)
        class_weights_tensor = load_class_weights(args.class_weights, args.weight_method, device=device)
        if class_weights_tensor is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
            logging.info("✓ 使用加权CrossEntropyLoss，权重方法: %s", args.weight_method)
        else:
            # 如果加载失败，使用默认权重
            class_weights = torch.tensor([0.25, 2.0, 1.5, 2.0], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logging.info("使用默认类别权重: [0.25, 2.0, 1.5, 2.0]")
    else:
        # 使用默认权重（简单提升前景影响）
        class_weights = torch.tensor([0.25, 2.0, 1.5, 2.0], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logging.info("使用默认类别权重: [0.25, 2.0, 1.5, 2.0]，建议使用 --class-weights 指定计算的权重")
    amp_enabled = args.amp and device.type == "cuda"
    if device.type == "cuda":
        scaler = GradScaler("cuda", enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=False)

    best_val_dice = 0.0
    global_step = 0

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        logging.info("Epoch [%d/%d]", epoch, args.epochs)

        train_loss, train_metrics, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            n_classes=4,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            grad_clip=args.grad_clip,
            use_wandb=use_wandb,
            global_step=global_step,
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/epoch_loss": train_loss,
                "train/dice_WT": train_metrics["dice_WT"],
                "train/dice_TC": train_metrics["dice_TC"],
                "train/dice_ET": train_metrics["dice_ET"],
                "train/dice_mean": train_metrics["dice_mean"],
                "train/dice_class_0": train_metrics["dice_class_0"],
                "train/dice_class_1": train_metrics["dice_class_1"],
                "train/dice_class_2": train_metrics["dice_class_2"],
                "train/dice_class_3": train_metrics["dice_class_3"],
            }, step=global_step)

        val_loss, val_metrics, sample_data = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            n_classes=4,
            amp_enabled=amp_enabled,
            return_sample=True,
        )

        logging.info(
            "Val Loss: %.4f | Dice WT: %.4f | Dice TC: %.4f | Dice ET: %.4f | Dice Mean: %.4f",
            val_loss,
            val_metrics["dice_WT"],
            val_metrics["dice_TC"],
            val_metrics["dice_ET"],
            val_metrics["dice_mean"],
        )

        vis_image = None
        if use_wandb and sample_data is not None:
            try:
                vis_image = create_segmentation_visualization(
                    sample_data[0], sample_data[1], sample_data[2], modality_idx=0
                )
            except Exception as e:
                logging.warning("Failed to create visualization: %s", e)

        if use_wandb:
            log_dict = {
                "val/epoch_loss": val_loss,
                "val/dice_WT": val_metrics["dice_WT"],
                "val/dice_TC": val_metrics["dice_TC"],
                "val/dice_ET": val_metrics["dice_ET"],
                "val/dice_mean": val_metrics["dice_mean"],
                "val/dice_class_0": val_metrics["dice_class_0"],
                "val/dice_class_1": val_metrics["dice_class_1"],
                "val/dice_class_2": val_metrics["dice_class_2"],
                "val/dice_class_3": val_metrics["dice_class_3"],
            }
            if vis_image is not None:
                log_dict["val/segmentation"] = vis_image
            wandb.log(log_dict, step=global_step)

        if val_metrics["dice_WT"] > best_val_dice:
            best_val_dice = val_metrics["dice_WT"]
            checkpoint_path = Path(args.checkpoint_dir) / "unet3d_best.pth"
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
            }, checkpoint_path)
            logging.info("Saved best checkpoint to %s (WT Dice: %.4f)", checkpoint_path, best_val_dice)

    if use_wandb:
        wandb.run.summary["best_val_dice_WT"] = best_val_dice
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory! Consider reducing batch size or target shape.")
        raise

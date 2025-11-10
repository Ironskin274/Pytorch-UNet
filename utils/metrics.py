"""
评估指标计算
包括 Dice 系数和 BraTS 特定的区域指标
"""
import torch
import numpy as np


def dice_coefficient(pred, target, num_classes=4, epsilon=1e-8):
    """
    计算每个类别的 Dice 系数

    参数:
        pred: 预测结果 [B, ...] 或 [...]
        target: 真实标签 [B, ...] 或 [...]
        num_classes: 类别数量
        epsilon: 平滑项

    返回:
        dice_scores: 每个类别的 Dice 分数列表
    """
    dice_scores = []

    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target

    for class_idx in range(num_classes):
        pred_class = (pred_np == class_idx).astype(np.float32)
        target_class = (target_np == class_idx).astype(np.float32)

        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class)

        if union == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection + epsilon) / (union + epsilon)

        dice_scores.append(dice)

    return dice_scores


def compute_brats_regions(pred, target, epsilon=1e-8):
    """
    计算 BraTS 特定的区域 Dice 分数

    BraTS 区域定义:
    - WT (Whole Tumor): 包含所有肿瘤区域 (类别 1, 2, 3)
    - TC (Tumor Core): 肿瘤核心 (类别 1, 3)
    - ET (Enhancing Tumor): 增强肿瘤 (类别 3)

    参数:
        pred: 预测结果 [B, ...] 或 [...]
        target: 真实标签 [B, ...] 或 [...]
        epsilon: 平滑项

    返回:
        wt_dice, tc_dice, et_dice: 三个区域的 Dice 分数
    """
    if isinstance(pred, torch.Tensor):
        pred_np = pred.detach().cpu().numpy()
    else:
        pred_np = pred
    if isinstance(target, torch.Tensor):
        target_np = target.detach().cpu().numpy()
    else:
        target_np = target

    pred_wt = (pred_np > 0).astype(np.float32)
    target_wt = (target_np > 0).astype(np.float32)
    intersection_wt = np.sum(pred_wt * target_wt)
    union_wt = np.sum(pred_wt) + np.sum(target_wt)
    wt_dice = (2.0 * intersection_wt + epsilon) / (union_wt + epsilon) if union_wt > 0 else 1.0

    pred_tc = ((pred_np == 1) | (pred_np == 3)).astype(np.float32)
    target_tc = ((target_np == 1) | (target_np == 3)).astype(np.float32)
    intersection_tc = np.sum(pred_tc * target_tc)
    union_tc = np.sum(pred_tc) + np.sum(target_tc)
    tc_dice = (2.0 * intersection_tc + epsilon) / (union_tc + epsilon) if union_tc > 0 else 1.0

    pred_et = (pred_np == 3).astype(np.float32)
    target_et = (target_np == 3).astype(np.float32)
    intersection_et = np.sum(pred_et * target_et)
    union_et = np.sum(pred_et) + np.sum(target_et)
    et_dice = (2.0 * intersection_et + epsilon) / (union_et + epsilon) if union_et > 0 else 1.0

    return wt_dice, tc_dice, et_dice


def compute_metrics(pred, target, num_classes=4):
    """
    计算所有指标（每类 Dice + BraTS 区域）

    参数:
        pred: 预测结果 [B, ...]
        target: 真实标签 [B, ...]
        num_classes: 类别数量

    返回:
        metrics: 包含所有指标的字典
    """
    if isinstance(pred, torch.Tensor):
        pred_tensor = pred.detach().cpu()
    else:
        pred_tensor = torch.from_numpy(pred)
    if isinstance(target, torch.Tensor):
        target_tensor = target.detach().cpu()
    else:
        target_tensor = torch.from_numpy(target)

    pred_tensor = pred_tensor.to(torch.int64)
    target_tensor = target_tensor.to(torch.int64)

    # 每个类别的 Dice（使用 torch 计算，支持任意维度）
    dice_scores = []
    for class_idx in range(num_classes):
        pred_class = (pred_tensor == class_idx)
        target_class = (target_tensor == class_idx)

        pred_sum = pred_class.sum().item()
        target_sum = target_class.sum().item()
        intersection = torch.logical_and(pred_class, target_class).sum().item()

        union = pred_sum + target_sum
        if union == 0:
            dice_scores.append(1.0 if intersection == 0 else 0.0)
        else:
            dice_scores.append((2.0 * intersection) / union)

    # BraTS 区域
    wt_pred = (pred_tensor > 0)
    wt_target = (target_tensor > 0)
    wt_intersection = torch.logical_and(wt_pred, wt_target).sum().item()
    wt_union = wt_pred.sum().item() + wt_target.sum().item()
    wt_dice = (2.0 * wt_intersection) / wt_union if wt_union > 0 else 1.0

    tc_pred = torch.logical_or(pred_tensor == 1, pred_tensor == 3)
    tc_target = torch.logical_or(target_tensor == 1, target_tensor == 3)
    tc_intersection = torch.logical_and(tc_pred, tc_target).sum().item()
    tc_union = tc_pred.sum().item() + tc_target.sum().item()
    tc_dice = (2.0 * tc_intersection) / tc_union if tc_union > 0 else 1.0

    et_pred = (pred_tensor == 3)
    et_target = (target_tensor == 3)
    et_intersection = torch.logical_and(et_pred, et_target).sum().item()
    et_union = et_pred.sum().item() + et_target.sum().item()
    et_dice = (2.0 * et_intersection) / et_union if et_union > 0 else 1.0

    metrics = {
        "dice_class_0": float(dice_scores[0]),
        "dice_class_1": float(dice_scores[1]),
        "dice_class_2": float(dice_scores[2]),
        "dice_class_3": float(dice_scores[3]),
        "dice_mean": float(np.mean(dice_scores[1:])),  # 排除背景
        "dice_WT": float(wt_dice),
        "dice_TC": float(tc_dice),
        "dice_ET": float(et_dice),
    }

    return metrics


def remap_labels(mask):
    """
    重映射BraTS标签: {0, 1, 2, 4} -> {0, 1, 2, 3}

    参数:
        mask: 原始标签

    返回:
        重映射后的标签
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.clone()
        mask[mask == 4] = 3
    else:
        mask = mask.copy()
        mask[mask == 4] = 3
    return mask


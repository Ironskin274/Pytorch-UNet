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
        pred: 预测结果 [B, H, W] 或 [H, W]
        target: 真实标签 [B, H, W] 或 [H, W]
        num_classes: 类别数量
        epsilon: 平滑项
    
    返回:
        dice_scores: 每个类别的 Dice 分数列表
    """
    dice_scores = []
    
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).astype(np.float32)
        target_class = (target == class_idx).astype(np.float32)
        
        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class)
        
        if union == 0:
            # 如果该类别不存在，跳过
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
        pred: 预测结果 [B, H, W] 或 [H, W]
        target: 真实标签 [B, H, W] 或 [H, W]
        epsilon: 平滑项
    
    返回:
        wt_dice, tc_dice, et_dice: 三个区域的 Dice 分数
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()
    
    # WT: Whole Tumor (类别 1, 2, 3)
    pred_wt = (pred > 0).astype(np.float32)
    target_wt = (target > 0).astype(np.float32)
    
    intersection_wt = np.sum(pred_wt * target_wt)
    union_wt = np.sum(pred_wt) + np.sum(target_wt)
    wt_dice = (2.0 * intersection_wt + epsilon) / (union_wt + epsilon) if union_wt > 0 else 1.0
    
    # TC: Tumor Core (类别 1, 3)
    pred_tc = ((pred == 1) | (pred == 3)).astype(np.float32)
    target_tc = ((target == 1) | (target == 3)).astype(np.float32)
    
    intersection_tc = np.sum(pred_tc * target_tc)
    union_tc = np.sum(pred_tc) + np.sum(target_tc)
    tc_dice = (2.0 * intersection_tc + epsilon) / (union_tc + epsilon) if union_tc > 0 else 1.0
    
    # ET: Enhancing Tumor (类别 3)
    pred_et = (pred == 3).astype(np.float32)
    target_et = (target == 3).astype(np.float32)
    
    intersection_et = np.sum(pred_et * target_et)
    union_et = np.sum(pred_et) + np.sum(target_et)
    et_dice = (2.0 * intersection_et + epsilon) / (union_et + epsilon) if union_et > 0 else 1.0
    
    return wt_dice, tc_dice, et_dice


def compute_metrics(pred, target, num_classes=4):
    """
    计算所有指标（每类 Dice + BraTS 区域）
    
    参数:
        pred: 预测结果 [B, H, W] 或 [H, W]
        target: 真实标签 [B, H, W] 或 [H, W]
        num_classes: 类别数量
    
    返回:
        metrics: 包含所有指标的字典
    """
    # 每个类别的 Dice
    dice_scores = dice_coefficient(pred, target, num_classes)
    
    # BraTS 区域的 Dice
    wt_dice, tc_dice, et_dice = compute_brats_regions(pred, target)
    
    metrics = {
        "dice_class_0": dice_scores[0],
        "dice_class_1": dice_scores[1],
        "dice_class_2": dice_scores[2],
        "dice_class_3": dice_scores[3],
        "dice_mean": np.mean(dice_scores[1:]),  # 平均 Dice（排除背景）
        "dice_WT": wt_dice,
        "dice_TC": tc_dice,
        "dice_ET": et_dice,
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


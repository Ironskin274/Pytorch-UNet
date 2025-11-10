"""
可视化工具函数
用于创建分割可视化图像用于wandb记录
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import torch


# 定义固定的类别颜色
class_colors = ['black', 'green', 'blue', 'red']  # 0: 背景, 1: NCR/NET, 2: 水肿, 3: 增强肿瘤
cmap = ListedColormap(class_colors)
norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)


def create_segmentation_visualization(image, true_mask, pred_mask, modality_idx=0):
    """
    创建分割可视化图像用于wandb记录
    
    参数:
        image: [4, H, W] 或 [B, 4, H, W] 图像tensor
        true_mask: [H, W] 或 [B, H, W] 真实标签tensor
        pred_mask: [H, W] 或 [B, H, W] 预测标签tensor
        modality_idx: 要显示的模态索引 (0: FLAIR, 1: T1, 2: T1CE, 3: T2)
    
    返回:
        wandb.Image: 用于wandb记录的可视化图像
    """
    import wandb
    
    # 处理batch维度
    if image.dim() == 4:
        image = image[0]  # 取第一个样本
    if true_mask.dim() == 3:
        true_mask = true_mask[0]
    if pred_mask.dim() == 3:
        pred_mask = pred_mask[0]
    
    # 转换为numpy
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy().astype(np.uint8)
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    
    # 确保mask值在有效范围内
    true_mask = np.clip(true_mask, 0, 3)
    pred_mask = np.clip(pred_mask, 0, 3)
    
    # 获取要显示的模态
    if image.shape[0] == 4:
        img_slice = image[modality_idx]
    else:
        img_slice = image[0] if image.shape[0] > 1 else image.squeeze()
    
    # 归一化图像用于显示
    img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
    
    # 创建可视化图像
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. 原始图像 (FLAIR)
    axs[0].imshow(img_slice, cmap='gray')
    axs[0].set_title('FLAIR Image', fontsize=12)
    axs[0].axis('off')
    
    # 2. 真实标签
    axs[1].imshow(true_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axs[1].set_title('Ground Truth', fontsize=12)
    axs[1].axis('off')
    
    # 3. 预测标签
    axs[2].imshow(pred_mask, cmap=cmap, norm=norm, interpolation='nearest')
    axs[2].set_title('Prediction', fontsize=12)
    axs[2].axis('off')
    
    # 4. 叠加显示 (原始图像 + 预测标签)
    axs[3].imshow(img_slice, cmap='gray')
    overlay = np.ma.masked_where(pred_mask == 0, pred_mask)
    axs[3].imshow(overlay, cmap=cmap, norm=norm, alpha=0.6, interpolation='nearest')
    axs[3].set_title('Overlay', fontsize=12)
    axs[3].axis('off')
    
    plt.tight_layout()
    
    # 转换为wandb.Image
    wandb_image = wandb.Image(fig)
    plt.close(fig)  # 关闭图形以释放内存
    
    return wandb_image


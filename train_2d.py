"""
2D UNet训练脚本
使用refractor版本的配置：Adam优化器、CrossEntropyLoss、lr=1e-4
"""
import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from unet import UNet
from utils.data_loading_2d import BraTS2020Dataset
from utils.metrics import compute_metrics, remap_labels
from utils.visualization import create_segmentation_visualization

# BraTS2020数据集路径
dir_brats_train = '/data/ssd2/liying/Datasets/BraTS2020/MICCAI_BraTS2020_TrainingData/'
train_list_file = '/data/ssd2/liying/Datasets/BraTS2020/train_list.txt'
valid_list_file = '/data/ssd2/liying/Datasets/BraTS2020/valid_list.txt'

dir_checkpoint = Path('./checkpoints/')


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=None, use_wandb=False, experiment=None, log_interval=50, global_step=None):
    """
    训练一个epoch
    
    参数:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前epoch（用于wandb记录）
        use_wandb: 是否使用wandb
        experiment: wandb experiment对象
        log_interval: 每N个batch记录一次（默认50）
        global_step: 全局步数（用于wandb step）
    
    返回:
        avg_loss: 平均损失
        metrics: 包含Dice指标的字典
        global_step: 更新后的全局步数
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    if global_step is None:
        global_step = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc='Training', leave=False)):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        masks = remap_labels(masks).long()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        global_step += 1
        
        # 收集预测和真实标签用于计算 Dice
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds.cpu())
        all_targets.append(masks.cpu())
        
        # 按间隔记录到wandb（训练loss）
        if use_wandb and experiment is not None and (batch_idx + 1) % log_interval == 0:
            try:
                experiment.log({
                    "train/batch_loss": float(loss.item()),
                    "train/learning_rate": float(optimizer.param_groups[0]['lr']),
                }, step=global_step)
            except Exception as e:
                logging.warning(f'Failed to log batch to wandb: {e}')
    
    # 计算 Dice 指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss, metrics, global_step


def validate(model, dataloader, criterion, device, return_sample=False):
    """
    验证模型
    
    参数:
        model: 模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 计算设备
        return_sample: 是否返回样本用于可视化
    
    返回:
        avg_loss: 平均损失
        metrics: 包含Dice指标的字典
        sample_data: (可选) 用于可视化的样本数据 (image, true_mask, pred_mask)
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    sample_data = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Validation', leave=False)):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            masks = remap_labels(masks).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            # 收集预测和真实标签用于计算 Dice
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
            
            # 保存第一个batch的第一个样本用于可视化
            if return_sample and batch_idx == 0 and sample_data is None:
                sample_data = (
                    images[0:1].cpu(),  # [1, 4, H, W]
                    masks[0:1].cpu(),   # [1, H, W]
                    preds[0:1].cpu()    # [1, H, W]
                )
    
    # 计算 Dice 指标
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    
    avg_loss = total_loss / len(dataloader)
    
    if return_sample:
        return avg_loss, metrics, sample_data
    else:
        return avg_loss, metrics


def train_model(
        model,
        device,
        epochs: int = 50,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        save_checkpoint: bool = True,
        img_scale: float = 1.0,
        use_wandb: bool = True,
        log_interval: int = 50,
):
    # 1. Create dataset
    logging.info('Using BraTS2020 2D dataset (slices)')
    train_set = BraTS2020Dataset(
        data_dir=dir_brats_train,
        list_file=train_list_file,
        scale=img_scale
    )
    val_set = BraTS2020Dataset(
        data_dir=dir_brats_train,
        list_file=valid_list_file,
        scale=img_scale
    )
    n_train = len(train_set)
    n_val = len(val_set)

    # 2. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 3. Initialize wandb
    experiment = None
    if use_wandb:
        try:
            project_name = 'BraTS2020-UNet-2D'
            experiment = wandb.init(
                project=project_name,
                resume='allow',
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'save_checkpoint': save_checkpoint,
                    'img_scale': img_scale,
                    'n_train': n_train,
                    'n_val': n_val,
                    'device': str(device),
                }
            )
            logging.info(f'W&B initialized successfully!')
            logging.info(f'  Run name: {experiment.name}')
            logging.info(f'  Run ID: {experiment.id}')
            logging.info(f'  W&B URL: {experiment.url}')
        except Exception as e:
            logging.error(f'Failed to initialize wandb: {e}')
            import traceback
            logging.error(traceback.format_exc())
            use_wandb = False
            experiment = None

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # 4. Set up the optimizer and loss (refractor style)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_dice_wt = 0.0
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        logging.info(f'\nEpoch [{epoch}/{epochs}]')
        
        # 训练
        train_loss, train_metrics, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch=epoch, use_wandb=use_wandb, experiment=experiment,
            log_interval=log_interval, global_step=global_step
        )
        logging.info(f'Train Loss: {train_loss:.4f}')
        logging.info(f'Train Metrics:')
        logging.info(f'  WT Dice: {train_metrics["dice_WT"]:.4f}')
        logging.info(f'  TC Dice: {train_metrics["dice_TC"]:.4f}')
        logging.info(f'  ET Dice: {train_metrics["dice_ET"]:.4f}')
        logging.info(f'  Mean Dice: {train_metrics["dice_mean"]:.4f}')
        
        # 验证（每个epoch都记录可视化图像）
        log_images = use_wandb and experiment is not None
        if log_images:
            val_loss, val_metrics, sample_data = validate(model, val_loader, criterion, device, return_sample=True)
            # 创建可视化图像
            try:
                vis_image = create_segmentation_visualization(
                    sample_data[0],  # image
                    sample_data[1],  # true_mask
                    sample_data[2],  # pred_mask
                    modality_idx=0   # FLAIR
                )
            except Exception as e:
                logging.warning(f'Failed to create visualization: {e}')
                vis_image = None
        else:
            val_loss, val_metrics = validate(model, val_loader, criterion, device, return_sample=False)
            vis_image = None
        
        logging.info(f'Val Loss: {val_loss:.4f}')
        logging.info(f'Val Metrics:')
        logging.info(f'  WT Dice: {val_metrics["dice_WT"]:.4f}')
        logging.info(f'  TC Dice: {val_metrics["dice_TC"]:.4f}')
        logging.info(f'  ET Dice: {val_metrics["dice_ET"]:.4f}')
        logging.info(f'  Mean Dice: {val_metrics["dice_mean"]:.4f}')
        
        # 记录到 W&B
        if use_wandb and experiment is not None:
            try:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": float(train_loss),
                    "train/dice_WT": float(train_metrics['dice_WT']),
                    "train/dice_TC": float(train_metrics['dice_TC']),
                    "train/dice_ET": float(train_metrics['dice_ET']),
                    "train/dice_mean": float(train_metrics['dice_mean']),
                    "train/dice_class_0": float(train_metrics['dice_class_0']),
                    "train/dice_class_1": float(train_metrics['dice_class_1']),
                    "train/dice_class_2": float(train_metrics['dice_class_2']),
                    "train/dice_class_3": float(train_metrics['dice_class_3']),
                    "val/loss": float(val_loss),
                    "val/dice_WT": float(val_metrics['dice_WT']),
                    "val/dice_TC": float(val_metrics['dice_TC']),
                    "val/dice_ET": float(val_metrics['dice_ET']),
                    "val/dice_mean": float(val_metrics['dice_mean']),
                    "val/dice_class_0": float(val_metrics['dice_class_0']),
                    "val/dice_class_1": float(val_metrics['dice_class_1']),
                    "val/dice_class_2": float(val_metrics['dice_class_2']),
                    "val/dice_class_3": float(val_metrics['dice_class_3']),
                }
                # 添加可视化图像
                if vis_image is not None:
                    log_dict["val/segmentation"] = vis_image
                experiment.log(log_dict, step=global_step)
                if epoch == 1:
                    logging.info(f'First epoch logged to wandb: {list(log_dict.keys())[:5]}...')
            except Exception as e:
                logging.error(f'Failed to log to wandb at epoch {epoch}: {e}')
                import traceback
                logging.error(traceback.format_exc())
        
        # 保存最佳模型（基于 WT Dice）
        if val_metrics['dice_WT'] > best_dice_wt:
            best_dice_wt = val_metrics['dice_WT']
            best_val_loss = val_loss
            if save_checkpoint:
                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = train_set.mask_values
                torch.save(state_dict, str(dir_checkpoint / 'checkpoint_2d_best.pth'))
                logging.info(f'Checkpoint saved! (WT Dice: {best_dice_wt:.4f})')
                
                if use_wandb and experiment is not None:
                    try:
                        experiment.summary["best_dice_WT"] = best_dice_wt
                        experiment.summary["best_val_loss"] = best_val_loss
                    except Exception as e:
                        logging.warning(f'Failed to update wandb summary: {e}')
        
        # 定期保存checkpoint
        if save_checkpoint and (epoch % 10 == 0):
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'checkpoint_2d_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')
    
    logging.info(f'\nTraining completed!')
    logging.info(f'Best WT Dice: {best_dice_wt:.4f}')
    logging.info(f'Best Val Loss: {best_val_loss:.4f}')
    
    # 关闭 W&B
    if use_wandb and experiment is not None:
        try:
            wandb.finish()
            logging.info('W&B finished')
        except Exception as e:
            logging.warning(f'Failed to finish wandb: {e}')


def get_args():
    parser = argparse.ArgumentParser(description='Train the 2D UNet on BraTS2020')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision (not used in 2D)')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--no-wandb', action='store_true', default=False, help='Disable wandb logging')
    parser.add_argument('--log-interval', type=int, default=50, dest='log_interval',
                        help='每N个batch记录一次训练loss到wandb（默认: 50）')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # BraTS2020: 4 modalities (t1, t1ce, t2, flair), 4 classes (background + 3 tumor types)
    n_channels = 4
    n_classes = 4
    
    logging.info('Using BraTS2020 2D dataset configuration: 4 input channels, 4 output classes')
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            use_wandb=not args.no_wandb,
            log_interval=args.log_interval,
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Consider reducing batch size or enabling gradient checkpointing.')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            use_wandb=not args.no_wandb,
            log_interval=args.log_interval,
        )


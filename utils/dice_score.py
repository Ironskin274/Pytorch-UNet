import torch
from torch import Tensor
import logging


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-8):
    """
    计算Dice系数，改进的数值稳定性版本

    支持 2D 与 3D（乃至更高维）输入：
    - 当 reduce_batch_first=True 时，视第0维为批次维度，按除第0维以外的所有维度进行求和
    - 当 reduce_batch_first=False 时，对所有维度进行求和（用于无batch情形）
    """
    assert input.size() == target.size(), "input and target must have the same shape"

    # 检查输入是否包含NaN/Inf
    if torch.isnan(input).any() or torch.isinf(input).any():
        logging.warning('dice_coeff: input contains NaN/Inf')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)
    if torch.isnan(target).any() or torch.isinf(target).any():
        logging.warning('dice_coeff: target contains NaN/Inf')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    # 按维度自适应计算维度集合
    # reduce_batch_first=True: 保留第0维(batch)，对其余维度求和
    # reduce_batch_first=False: 对所有维度求和
    if input.dim() == 0:
        logging.warning('dice_coeff: input has zero dims')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    if reduce_batch_first:
        if input.dim() < 2:
            logging.warning('dice_coeff: input dim too small for reduce_batch_first')
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        sum_dim = tuple(range(1, input.dim()))
    else:
        sum_dim = tuple(range(0, input.dim()))

    # 计算交集和并集
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    # 处理空集情况
    sets_sum = torch.where(sets_sum == 0, inter + epsilon, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    dice = torch.clamp(dice, min=0.0, max=1.0)

    if torch.isnan(dice).any() or torch.isinf(dice).any():
        logging.warning('dice_coeff: result contains NaN/Inf, returning 0')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)

    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-8):
    """
    多类别Dice系数，改进的数值稳定性版本。
    兼容 2D/3D：调用 dice_coeff 时自动按维度聚合。
    """
    # 使用更稳定的方式：先把 (B, C, ...) 展平为 (B*C, ...)，再对除第0维外的所有维度求和
    try:
        # 确保输入至少有通道维
        if input.dim() < 3:
            logging.warning('multiclass_dice_coeff: input/target dimensions too small')
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)

        flattened_input = input.flatten(0, 1)
        flattened_target = target.flatten(0, 1)
        result = dice_coeff(flattened_input, flattened_target, reduce_batch_first=True, epsilon=epsilon)
        if torch.isnan(result) or torch.isinf(result):
            return torch.tensor(0.0, device=input.device, dtype=input.dtype)
        return result
    except Exception as e:
        logging.warning(f'multiclass_dice_coeff error: {e}, returning 0')
        return torch.tensor(0.0, device=input.device, dtype=input.dtype)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False, epsilon: float = 1e-8):
    """
    Dice损失函数，改进的数值稳定性版本
    """
    try:
        fn = multiclass_dice_coeff if multiclass else dice_coeff
        dice = fn(input, target, reduce_batch_first=True, epsilon=epsilon)

        loss = 1 - dice
        loss = torch.clamp(loss, min=0.0, max=1.0)

        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning('dice_loss: result is NaN/Inf, returning 1.0 (maximum loss)')
            return torch.tensor(1.0, device=input.device, dtype=input.dtype)

        return loss
    except Exception as e:
        logging.warning(f'dice_loss error: {e}, returning 1.0 (maximum loss)')
        return torch.tensor(1.0, device=input.device, dtype=input.dtype)

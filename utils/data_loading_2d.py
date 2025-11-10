"""
2D BraTS2020数据集加载器
支持多模态MRI和3D到2D切片转换
"""
import logging
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import nibabel as nib


class BraTS2020Dataset(Dataset):
    """BraTS2020数据集加载器，支持多模态MRI和3D到2D切片转换"""
    
    def __init__(self, data_dir: str, list_file: str, scale: float = 1.0, 
                 slice_range=(2, 150), modalities=['t1', 't1ce', 't2', 'flair']):
        """
        Args:
            data_dir: BraTS2020训练数据根目录
            list_file: 包含病例名称列表的文本文件
            scale: 图像缩放比例
            slice_range: 使用的切片范围 (start, end)，去除背景切片
            modalities: 使用的模态列表
        """
        self.data_dir = Path(data_dir)
        self.scale = scale
        self.slice_range = slice_range
        self.modalities = modalities
        
        # 读取病例列表
        with open(list_file, 'r') as f:
            self.case_names = [line.strip() for line in f if line.strip()]
        
        if not self.case_names:
            raise RuntimeError(f'No cases found in {list_file}')
        
        # 构建所有有效的切片索引
        self.samples = []
        logging.info(f'Loading BraTS2020 dataset from {data_dir}')
        logging.info(f'Found {len(self.case_names)} cases')
        
        for case_name in tqdm(self.case_names, desc='Indexing cases'):
            case_dir = self.data_dir / case_name
            if not case_dir.exists():
                logging.warning(f'Case directory not found: {case_dir}')
                continue
            
            # 检查所有需要的文件是否存在
            seg_file = case_dir / f'{case_name}_seg.nii'
            if not seg_file.exists():
                # 尝试.nii.gz
                seg_file = case_dir / f'{case_name}_seg.nii.gz'
                if not seg_file.exists():
                    logging.warning(f'Segmentation file not found: {case_name}_seg')
                    continue
            
            # 读取分割文件来确定有效切片范围
            seg_data = nib.load(str(seg_file)).get_fdata()
            depth = seg_data.shape[2]
            
            # 只选择包含肿瘤的切片或在指定范围内的切片
            start_slice = max(0, self.slice_range[0])
            end_slice = min(depth, self.slice_range[1])
            
            for slice_idx in range(start_slice, end_slice):
                # 检查该切片是否包含非零标签（可选优化）
                slice_mask = seg_data[:, :, slice_idx]
                if np.any(slice_mask > 0):  # 只保留有肿瘤的切片
                    self.samples.append((case_name, slice_idx))
        
        logging.info(f'Created dataset with {len(self.samples)} valid slices')
        
        # BraTS2020的标签值：0-背景, 1-坏死/非增强肿瘤核心, 2-水肿, 4-增强肿瘤
        # 我们将其映射为：0-背景, 1-坏死, 2-水肿, 3-增强肿瘤
        self.mask_values = [0, 1, 2, 3]
    
    def __len__(self):
        return len(self.samples)
    
    def _find_nifti_file(self, directory, base_name):
        """查找NIfTI文件，支持.nii和.nii.gz两种扩展名"""
        for ext in ['.nii', '.nii.gz']:
            file_path = directory / f'{base_name}{ext}'
            if file_path.exists():
                return file_path
        return None
    
    def load_slice(self, case_name, slice_idx):
        """加载特定病例的特定切片"""
        case_dir = self.data_dir / case_name
        
        # 加载所有模态
        modality_data = []
        for modality in self.modalities:
            file_path = self._find_nifti_file(case_dir, f'{case_name}_{modality}')
            if file_path is None:
                raise FileNotFoundError(f'Modality file not found: {case_name}_{modality}')
            
            nii_data = nib.load(str(file_path)).get_fdata()
            slice_data = nii_data[:, :, slice_idx]
            modality_data.append(slice_data)
        
        # 合并所有模态 (H, W, C)
        image = np.stack(modality_data, axis=-1)
        
        # 加载分割标签
        seg_file = self._find_nifti_file(case_dir, f'{case_name}_seg')
        if seg_file is None:
            raise FileNotFoundError(f'Segmentation file not found: {case_name}_seg')
        seg_data = nib.load(str(seg_file)).get_fdata()
        mask = seg_data[:, :, slice_idx]
        
        return image, mask
    
    @staticmethod
    def normalize_image(image):
        """归一化图像到[0, 1]"""
        # 对每个通道分别归一化
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[-1]):
            channel = image[:, :, i]
            # 归一化到[0, 1]，使用百分位数来处理异常值
            p1, p99 = np.percentile(channel, (1, 99))
            if p99 > p1:
                channel = np.clip(channel, p1, p99)
                channel = (channel - p1) / (p99 - p1)
            else:
                channel = np.zeros_like(channel)
            normalized[:, :, i] = channel
        return normalized
    
    @staticmethod
    def preprocess_mask(mask):
        """预处理BraTS标签：将标签4映射为3"""
        mask = mask.astype(np.int64)
        mask[mask == 4] = 3  # 将增强肿瘤标签从4映射为3
        return mask
    
    def preprocess(self, image, mask, scale):
        """预处理图像和掩码"""
        h, w = image.shape[:2]
        new_h, new_w = int(scale * h), int(scale * w)
        
        if scale != 1.0:
            # 使用PIL进行缩放
            # 图像部分
            channels = []
            for i in range(image.shape[-1]):
                pil_img = Image.fromarray((image[:, :, i] * 255).astype(np.uint8))
                pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
                channels.append(np.array(pil_img) / 255.0)
            image = np.stack(channels, axis=-1)
            
            # 掩码部分
            pil_mask = Image.fromarray(mask.astype(np.uint8))
            pil_mask = pil_mask.resize((new_w, new_h), Image.NEAREST)
            mask = np.array(pil_mask)
        
        # 转换图像维度从 (H, W, C) 到 (C, H, W)
        image = image.transpose((2, 0, 1))
        
        return image.astype(np.float32), mask.astype(np.int64)
    
    def __getitem__(self, idx):
        case_name, slice_idx = self.samples[idx]
        
        # 加载切片
        image, mask = self.load_slice(case_name, slice_idx)
        
        # 归一化图像
        image = self.normalize_image(image)
        
        # 预处理掩码
        mask = self.preprocess_mask(mask)
        
        # 缩放
        image, mask = self.preprocess(image, mask, self.scale)
        
        return {
            'image': torch.as_tensor(image.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


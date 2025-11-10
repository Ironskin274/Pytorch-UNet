"""
3D BraTS2020 数据集加载器
支持多模态MRI体积读取、标准化以及中心裁剪/填充
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class BraTS2020Dataset3D(Dataset):
    """加载BraTS2020完整3D体积用于3D U-Net训练"""

    def __init__(
        self,
        data_dir: str,
        list_file: str,
        modalities: Optional[List[str]] = None,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        normalize: bool = True,
    ):
        self.data_dir = Path(data_dir)
        if modalities is None:
            modalities = ["t1", "t1ce", "t2", "flair"]
        self.modalities = modalities
        self.target_shape = target_shape
        self.normalize = normalize

        with open(list_file, "r") as f:
            self.case_names = [line.strip() for line in f if line.strip()]

        if not self.case_names:
            raise RuntimeError(f"No cases found in {list_file}")

        missing_cases = [case for case in self.case_names if not (self.data_dir / case).exists()]
        for case in missing_cases:
            logging.warning("Case directory not found: %s", case)
        self.case_names = [case for case in self.case_names if case not in missing_cases]

        logging.info("BraTS2020 3D dataset initialized with %d cases", len(self.case_names))

    def __len__(self) -> int:
        return len(self.case_names)

    @staticmethod
    def _find_nifti_file(case_dir: Path, base_name: str) -> Optional[Path]:
        for ext in (".nii.gz", ".nii"):
            file_path = case_dir / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
        return None

    def _load_volume(self, case_name: str) -> Tuple[np.ndarray, np.ndarray]:
        case_dir = self.data_dir / case_name
        modality_volumes = []
        for modality in self.modalities:
            file_path = self._find_nifti_file(case_dir, f"{case_name}_{modality}")
            if file_path is None:
                raise FileNotFoundError(f"Modality file not found: {case_name}_{modality}")
            data = nib.load(str(file_path)).get_fdata(dtype=np.float32)
            modality_volumes.append(data)

        volume = np.stack(modality_volumes, axis=0)  # (C, H, W, D)

        seg_file = self._find_nifti_file(case_dir, f"{case_name}_seg")
        if seg_file is None:
            raise FileNotFoundError(f"Segmentation file not found: {case_name}_seg")
        mask = nib.load(str(seg_file)).get_fdata(dtype=np.float32)

        return volume, mask

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(channel, (1, 99))
        if p99 > p1:
            channel = np.clip(channel, p1, p99)
            channel = (channel - p1) / (p99 - p1)
        else:
            channel = np.zeros_like(channel)
        return channel.astype(np.float32)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return volume.astype(np.float32)
        normalized = np.zeros_like(volume, dtype=np.float32)
        for c in range(volume.shape[0]):
            normalized[c] = self._normalize_channel(volume[c])
        return normalized

    @staticmethod
    def _remap_labels(mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.int64)
        mask[mask == 4] = 3
        return mask

    @staticmethod
    def _center_crop_or_pad(volume: np.ndarray, mask: np.ndarray, target_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """中心裁剪或对体积进行零填充，使其匹配目标形状"""
        _, h, w, d = volume.shape
        th, tw, td = target_shape

        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        pad_d = max(td - d, 0)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            volume = np.pad(
                volume,
                ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_d // 2, pad_d - pad_d // 2)),
                mode="constant",
            )
            mask = np.pad(
                mask,
                ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_d // 2, pad_d - pad_d // 2)),
                mode="constant",
            )

        _, h, w, d = volume.shape
        start_h = max((h - th) // 2, 0)
        start_w = max((w - tw) // 2, 0)
        start_d = max((d - td) // 2, 0)

        volume = volume[:, start_h:start_h + th, start_w:start_w + tw, start_d:start_d + td]
        mask = mask[start_h:start_h + th, start_w:start_w + tw, start_d:start_d + td]
        return volume, mask

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx]
        volume, mask = self._load_volume(case_name)
        volume = self._normalize(volume)
        mask = self._remap_labels(mask)
        volume, mask = self._center_crop_or_pad(volume, mask, self.target_shape)

        # 转换为 (C, D, H, W)
        volume = np.transpose(volume, (0, 3, 1, 2))

        return {
            "image": torch.as_tensor(volume.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
            "case_name": case_name,
        }

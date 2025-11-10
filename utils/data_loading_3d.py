"""
3D BraTS2020 数据集加载器
支持多模态MRI体积读取、标准化以及中心裁剪/填充
"""
from __future__ import annotations

import logging
import random
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
        fg_min_ratio: float = 0.01,
        jitter_ratio: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        if modalities is None:
            # 参考实现使用3通道：flair, t1ce, t2
            modalities = ["flair", "t1ce", "t2"]
        self.modalities = modalities
        self.target_shape = target_shape
        self.normalize = normalize
        self.fg_min_ratio = fg_min_ratio
        self.jitter_ratio = jitter_ratio

        with open(list_file, "r") as f:
            all_cases = [line.strip() for line in f if line.strip()]

        if not all_cases:
            raise RuntimeError(f"No cases found in {list_file}")

        valid_cases: List[str] = []
        for case in all_cases:
            case_dir = self.data_dir / case
            if not case_dir.exists():
                logging.warning("Case directory not found: %s", case)
                continue
            try:
                # 预检查前景占比（以中心裁剪或bbox裁剪估计）
                _, mask = self._load_raw(case)
                mask = self._remap_labels(mask)
                # 估算裁剪窗口中心：优先用bbox中心，否则体积中心
                bbox = self._compute_bbox(mask)
                if bbox is not None:
                    center = tuple(int((lo + hi) / 2) for lo, hi in bbox)
                else:
                    h, w, d = mask.shape
                    center = (h // 2, w // 2, d // 2)
                crop_mask = self._crop_by_center(mask, center, self.target_shape)
                fg_ratio = float((crop_mask > 0).sum()) / float(np.prod(crop_mask.shape))
                if fg_ratio >= self.fg_min_ratio:
                    valid_cases.append(case)
                else:
                    logging.info("Filtered case %s due to low foreground ratio: %.6f", case, fg_ratio)
            except Exception as e:
                logging.warning("Skip case %s due to error: %s", case, e)

        self.case_names = valid_cases
        logging.info("BraTS2020 3D dataset initialized with %d cases (filtered by fg_min_ratio=%.3f)", len(self.case_names), self.fg_min_ratio)

    def __len__(self) -> int:
        return len(self.case_names)

    @staticmethod
    def _find_nifti_file(case_dir: Path, base_name: str) -> Optional[Path]:
        for ext in (".nii.gz", ".nii"):
            file_path = case_dir / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
        return None

    def _load_raw(self, case_name: str) -> Tuple[np.ndarray, np.ndarray]:
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
    def _compute_bbox(mask: np.ndarray) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]:
        pos = np.where(mask > 0)
        if pos[0].size == 0:
            return None
        hmin, hmax = int(pos[0].min()), int(pos[0].max())
        wmin, wmax = int(pos[1].min()), int(pos[1].max())
        dmin, dmax = int(pos[2].min()), int(pos[2].max())
        return (hmin, hmax), (wmin, wmax), (dmin, dmax)

    @staticmethod
    def _crop_by_center(arr: np.ndarray, center: Tuple[int, int, int], target_shape: Tuple[int, int, int]) -> np.ndarray:
        h, w, d = arr.shape
        th, tw, td = target_shape
        ch, cw, cd = center
        sh = max(min(ch - th // 2, h - th), 0)
        sw = max(min(cw - tw // 2, w - tw), 0)
        sd = max(min(cd - td // 2, d - td), 0)
        return arr[sh:sh + th, sw:sw + tw, sd:sd + td]

    def _center_crop_or_pad_with_bbox(self, volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # 计算中心：优先bbox中心
        bbox = self._compute_bbox(mask)
        if bbox is not None:
            (hmin, hmax), (wmin, wmax), (dmin, dmax) = bbox
            center = (int((hmin + hmax) / 2), int((wmin + wmax) / 2), int((dmin + dmax) / 2))
            # 抖动
            th, tw, td = self.target_shape
            jh = int(self.jitter_ratio * th)
            jw = int(self.jitter_ratio * tw)
            jd = int(self.jitter_ratio * td)
            center = (
                max(0, min(volume.shape[1] - 1, center[0] + random.randint(-jh, jh))),
                max(0, min(volume.shape[2] - 1, center[1] + random.randint(-jw, jw))),
                max(0, min(volume.shape[3] - 1, center[2] + random.randint(-jd, jd))),
            )
        else:
            # 无前景，退化为体积中心
            center = (volume.shape[1] // 2, volume.shape[2] // 2, volume.shape[3] // 2)

        # 若体积小于目标形状，先pad到至少目标尺寸
        _, h, w, d = volume.shape
        th, tw, td = self.target_shape
        pad_h = max(th - h, 0)
        pad_w = max(tw - w, 0)
        pad_d = max(td - d, 0)
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            volume = np.pad(volume, ((0, 0), (pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_d // 2, pad_d - pad_d // 2)), mode="constant")
            mask = np.pad(mask, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (pad_d // 2, pad_d - pad_d // 2)), mode="constant")
            # 更新中心
            center = (center[0] + pad_h // 2, center[1] + pad_w // 2, center[2] + pad_d // 2)

        # 以中心裁剪
        crop_vol = np.zeros((volume.shape[0],) + self.target_shape, dtype=volume.dtype)
        crop_msk = np.zeros(self.target_shape, dtype=mask.dtype)
        for c in range(volume.shape[0]):
            crop_vol[c] = self._crop_by_center(volume[c], center, self.target_shape)
        crop_msk = self._crop_by_center(mask, center, self.target_shape)
        return crop_vol, crop_msk

    def __getitem__(self, idx: int):
        case_name = self.case_names[idx]
        volume, mask = self._load_raw(case_name)
        volume = self._normalize(volume)
        mask = self._remap_labels(mask)
        volume, mask = self._center_crop_or_pad_with_bbox(volume, mask)

        # 转换为 (C, D, H, W)
        volume = np.transpose(volume, (0, 3, 1, 2))

        return {
            "image": torch.as_tensor(volume.copy()).float().contiguous(),
            "mask": torch.as_tensor(mask.copy()).long().contiguous(),
            "case_name": case_name,
        }

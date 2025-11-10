from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """(Conv3d => InstanceNorm3d => LeakyReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(mid_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=2, stride=2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        # Pad x1 to the same size as x2 if necessary
        diff_depth = x2.size(2) - x1.size(2)
        diff_height = x2.size(3) - x1.size(3)
        diff_width = x2.size(4) - x1.size(4)

        x1 = F.pad(
            x1,
            [
                diff_width // 2,
                diff_width - diff_width // 2,
                diff_height // 2,
                diff_height - diff_height // 2,
                diff_depth // 2,
                diff_depth - diff_depth // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet3D(nn.Module):
    """3D U-Net architecture for volumetric segmentation."""

    def __init__(
        self,
        n_channels: int = 4,
        n_classes: int = 4,
        base_channels: int = 32,
        bilinear: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv3D(n_channels, base_channels)
        self.down1 = Down3D(base_channels, base_channels * 2)
        self.down2 = Down3D(base_channels * 2, base_channels * 4)
        self.down3 = Down3D(base_channels * 4, base_channels * 8)
        self.down4 = Down3D(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up3D(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up3D(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up3D(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up3D(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv3D(base_channels, n_classes)

        if dropout > 0:
            self.dropout = nn.Dropout3d(p=dropout)
        else:
            self.dropout = None

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    @staticmethod
    def _init_layer(layer: nn.Module) -> None:
        if isinstance(layer, nn.Conv3d) or isinstance(layer, nn.ConvTranspose3d):
            nn.init.kaiming_normal_(layer.weight, a=0.1)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.InstanceNorm3d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            self._init_layer(module)

    def use_checkpointing(self) -> None:
        self.inc = torch.utils.checkpoint.checkpoint_sequential(self.inc, segments=1)
        self.down1 = torch.utils.checkpoint.checkpoint_sequential(self.down1, segments=1)
        self.down2 = torch.utils.checkpoint.checkpoint_sequential(self.down2, segments=1)
        self.down3 = torch.utils.checkpoint.checkpoint_sequential(self.down3, segments=1)
        self.down4 = torch.utils.checkpoint.checkpoint_sequential(self.down4, segments=1)
        self.up1 = torch.utils.checkpoint.checkpoint_sequential(self.up1, segments=1)
        self.up2 = torch.utils.checkpoint.checkpoint_sequential(self.up2, segments=1)
        self.up3 = torch.utils.checkpoint.checkpoint_sequential(self.up3, segments=1)
        self.up4 = torch.utils.checkpoint.checkpoint_sequential(self.up4, segments=1)
        self.outc = torch.utils.checkpoint.checkpoint_sequential(self.outc, segments=1)


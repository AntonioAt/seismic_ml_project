"""
models/unet3d.py
================
3D U-Net segmentation model for volumetric seismic interpretation.

Architecture:
    Encoder path  — ConvBlock3D + MaxPool3D (configurable depth)
    Bottleneck    — ConvBlock3D at lowest resolution
    Decoder path  — ConvTranspose3D + skip connection + ConvBlock3D
    Head          — 1×1×1 Conv → n_classes channels

Usage:
    from seismic_ml.models.unet3d import UNet3D

    model = UNet3D(in_channels=1, n_classes=3, base_features=32, depth=4)
    logits = model(patch)   # [B, n_classes, I, X, T]
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Two successive 3-D Conv → BN → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock3D(in_ch, out_ch)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x    = self.up(x)
        diff = [s - x.size(i + 2) for i, s in enumerate(skip.shape[2:])]
        x    = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x    = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3-D U-Net for volumetric seismic segmentation.

    Parameters
    ----------
    in_channels   : input channels (1 for single amplitude stack)
    n_classes     : output segmentation classes
    base_features : feature count at first encoder level
    depth         : number of encoder / decoder levels
    """

    def __init__(
        self,
        in_channels:   int = 1,
        n_classes:     int = 3,
        base_features: int = 32,
        depth:         int = 4,
    ) -> None:
        super().__init__()
        self.depth   = depth
        features     = [base_features * (2 ** i) for i in range(depth)]

        self.encoders   = nn.ModuleList()
        self.decoders   = nn.ModuleList()

        ch = in_channels
        for feat in features:
            self.encoders.append(EncoderBlock(ch, feat))
            ch = feat

        self.bottleneck = ConvBlock3D(ch, ch * 2)
        ch = ch * 2

        for feat in reversed(features):
            self.decoders.append(DecoderBlock(ch, feat, feat))
            ch = feat

        self.head = nn.Conv3d(ch, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        return self.head(x)

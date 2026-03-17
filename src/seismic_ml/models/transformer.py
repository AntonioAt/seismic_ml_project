"""
models/transformer.py
=====================
SeismicTransformer — Priority #2 Module

Attention-based 3D models as alternatives to U-Net for seismic segmentation.
Transformers capture long-range spatial dependencies that local 3D convolutions
miss — critical for tracing continuous horizon surfaces and fault planes that
span hundreds of inlines/crosslines.

Two architectures provided:

    SeismicTransformer
        Swin3D-inspired shifted-window self-attention with patch embedding.
        Best for large volumes where global context matters.
        Encoder-decoder with hierarchical feature maps.

    SeismicViT
        Lightweight 3D Vision Transformer (ViT) adapted for seismic patches.
        Simpler, faster to train on smaller datasets.
        Suitable for patch-level classification / dense prediction.

Key design decisions vs U-Net:
    ✓ Window self-attention  → O(n) complexity vs O(n²) global attention
    ✓ Shifted windows        → cross-window information flow
    ✓ Patch merging          → hierarchical downsampling like pooling
    ✓ Relative position bias → encodes 3D spatial relationships
    ✓ Skip connections kept  → preserves fine-grained boundary detail

Usage:
    from seismic_ml.models.transformer import SeismicTransformer, SeismicViT

    model = SeismicTransformer(in_channels=1, n_classes=3)
    model = SeismicViT(in_channels=1, n_classes=3, patch_size=(4,4,4))

    logits = model(patch)   # [B, n_classes, I, X, T]
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
#  SHARED PRIMITIVES
# ─────────────────────────────────────────────────────────────

class PatchEmbed3D(nn.Module):
    """
    Partition a 3D volume into non-overlapping patches and project
    each patch to an embedding vector.

    Input : [B, C, I, X, T]
    Output: [B, n_patches, embed_dim]
             where n_patches = (I/ps) * (X/ps) * (T/ps)
    """

    def __init__(
        self,
        in_channels: int = 1,
        patch_size:  Tuple[int, int, int] = (4, 4, 4),
        embed_dim:   int = 96,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, C, I, X, T = x.shape
        # Pad to multiples of patch_size
        pi, px, pt = self.patch_size
        pad_i = (pi - I % pi) % pi
        pad_x = (px - X % px) % px
        pad_t = (pt - T % pt) % pt
        x = F.pad(x, (0, pad_t, 0, pad_x, 0, pad_i))

        x = self.proj(x)                       # [B, embed_dim, nI, nX, nT]
        nI, nX, nT = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)       # [B, nI*nX*nT, embed_dim]
        x = self.norm(x)
        return x, (nI, nX, nT)


class PatchMerging3D(nn.Module):
    """
    Hierarchical downsampling — merges 2×2×2 neighbouring patch tokens
    into one, halving the spatial resolution and doubling channels.
    Analogous to max-pooling in CNN.

    Input : [B, nI*nX*nT, C]
    Output: [B, (nI//2)*(nX//2)*(nT//2), 2C]
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm     = nn.LayerNorm(8 * dim)
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)

    def forward(
        self,
        x:    torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        nI, nX, nT = grid
        B, L, C    = x.shape
        x = x.view(B, nI, nX, nT, C)

        # Pad odd dimensions
        if nI % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        if nX % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, 1))
        if nT % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))

        nI2, nX2, nT2 = x.shape[1] // 2, x.shape[2] // 2, x.shape[3] // 2

        # Gather 2×2×2 blocks
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 0::2, 1::2, 1::2, :]
        x6 = x[:, 1::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        merged = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        merged = merged.view(B, nI2 * nX2 * nT2, 8 * C)
        merged = self.norm(merged)
        merged = self.reduction(merged)
        return merged, (nI2, nX2, nT2)


class RelativePositionBias3D(nn.Module):
    """
    Learnable relative position bias for 3D window attention.
    Captures the spatial relationship between query and key tokens
    within a local window — critical for seismic where spatial
    continuity is a strong geological prior.
    """

    def __init__(
        self,
        window_size: Tuple[int, int, int],
        n_heads:     int,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        wI, wX, wT = window_size

        # Table size: (2*wI-1) * (2*wX-1) * (2*wT-1)
        self.bias_table = nn.Parameter(
            torch.zeros(
                (2 * wI - 1) * (2 * wX - 1) * (2 * wT - 1),
                n_heads,
            )
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute relative position indices
        coords_i = torch.arange(wI)
        coords_x = torch.arange(wX)
        coords_t = torch.arange(wT)
        grid     = torch.stack(
            torch.meshgrid(coords_i, coords_x, coords_t, indexing="ij")
        )                                          # [3, wI, wX, wT]
        flat     = grid.flatten(1)                 # [3, wI*wX*wT]
        rel      = flat[:, :, None] - flat[:, None, :]  # [3, N, N]

        rel[0] += wI - 1
        rel[1] += wX - 1
        rel[2] += wT - 1
        rel[0] *= (2 * wX - 1) * (2 * wT - 1)
        rel[1] *= (2 * wT - 1)
        index = rel.sum(dim=0)                     # [N, N]
        self.register_buffer("rel_idx", index)

    def forward(self) -> torch.Tensor:
        """Returns bias tensor [n_heads, N, N]."""
        bias = self.bias_table[self.rel_idx.view(-1)]
        N    = self.window_size[0] * self.window_size[1] * self.window_size[2]
        return bias.view(N, N, -1).permute(2, 0, 1)   # [H, N, N]


# ─────────────────────────────────────────────────────────────
#  WINDOW ATTENTION
# ─────────────────────────────────────────────────────────────

def window_partition_3d(
    x:           torch.Tensor,
    window_size: Tuple[int, int, int],
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """
    Partition volume tokens into non-overlapping windows.

    Input : [B, nI, nX, nT, C]
    Output: [B*nW, wI*wX*wT, C]   where nW = number of windows
    """
    B, nI, nX, nT, C = x.shape
    wI, wX, wT = window_size

    x = x.view(B,
               nI // wI, wI,
               nX // wX, wX,
               nT // wT, wT,
               C)
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7)
         .contiguous()
         .view(-1, wI * wX * wT, C)
    )
    return windows, (nI // wI, nX // wX, nT // wT)


def window_reverse_3d(
    windows:     torch.Tensor,
    window_size: Tuple[int, int, int],
    grid:        Tuple[int, int, int],
) -> torch.Tensor:
    """Reverse of window_partition_3d."""
    wI, wX, wT     = window_size
    nwI, nwX, nwT  = grid
    B_nW, _, C     = windows.shape
    B              = B_nW // (nwI * nwX * nwT)

    x = windows.view(B, nwI, nwX, nwT, wI, wX, wT, C)
    x = (
        x.permute(0, 1, 4, 2, 5, 3, 6, 7)
         .contiguous()
         .view(B, nwI * wI, nwX * wX, nwT * wT, C)
    )
    return x


class WindowAttention3D(nn.Module):
    """
    3D Shifted-Window Multi-Head Self-Attention (W-MSA / SW-MSA).

    Within each local window, full self-attention is computed.
    Shift alternates between layers to create cross-window connections
    without the quadratic cost of global attention.
    """

    def __init__(
        self,
        dim:         int,
        window_size: Tuple[int, int, int],
        n_heads:     int,
        qkv_bias:    bool  = True,
        attn_drop:   float = 0.0,
        proj_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.n_heads     = n_heads
        self.scale       = (dim // n_heads) ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pos_bias = RelativePositionBias3D(window_size, n_heads)
        self.softmax  = nn.Softmax(dim=-1)

    def forward(
        self,
        x:    torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B_, N, C = x.shape
        H        = self.n_heads
        head_dim = C // H

        qkv = (
            self.qkv(x)
                .reshape(B_, N, 3, H, head_dim)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self.pos_bias()

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, H, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, H, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ─────────────────────────────────────────────────────────────
#  SWIN TRANSFORMER BLOCK
# ─────────────────────────────────────────────────────────────

class SwinTransformerBlock3D(nn.Module):
    """
    One Swin Transformer block:
        LayerNorm → Window Attention (shifted or not) → residual
        LayerNorm → MLP (FFN)                         → residual

    Alternating shift_size=0 and shift_size=window_size//2
    creates the SW-MSA pattern that connects adjacent windows.
    """

    def __init__(
        self,
        dim:         int,
        n_heads:     int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
        shift_size:  Tuple[int, int, int] = (0, 0, 0),
        mlp_ratio:   float = 4.0,
        drop:        float = 0.0,
        attn_drop:   float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size  = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention3D(
            dim, window_size, n_heads,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x:    torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        B, L, C  = x.shape
        nI, nX, nT = grid
        shortcut = x

        x = self.norm1(x).view(B, nI, nX, nT, C)

        # Cyclic shift
        sI, sX, sT = self.shift_size
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(-sI, -sX, -sT), dims=(1, 2, 3))

        # Partition into windows
        windows, n_wins = window_partition_3d(x, self.window_size)

        # Attention
        attn_out = self.attn(windows)

        # Reverse windows
        x = window_reverse_3d(attn_out, self.window_size, n_wins)

        # Reverse cyclic shift
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(x, shifts=(sI, sX, sT), dims=(1, 2, 3))

        x = x.view(B, L, C) + shortcut

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class SwinStage3D(nn.Module):
    """
    One hierarchical stage = N alternating W-MSA / SW-MSA blocks.
    """

    def __init__(
        self,
        dim:          int,
        depth:        int,
        n_heads:      int,
        window_size:  Tuple[int, int, int] = (4, 4, 4),
        mlp_ratio:    float = 4.0,
        drop:         float = 0.0,
        attn_drop:    float = 0.0,
        downsample:   bool  = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            shift = (0, 0, 0) if i % 2 == 0 else tuple(
                s // 2 for s in window_size
            )
            self.blocks.append(
                SwinTransformerBlock3D(
                    dim=dim,
                    n_heads=n_heads,
                    window_size=window_size,
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                )
            )
        self.downsample = PatchMerging3D(dim) if downsample else None

    def forward(
        self,
        x:    torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        for blk in self.blocks:
            x = blk(x, grid)
        if self.downsample is not None:
            x, grid = self.downsample(x, grid)
        return x, grid


# ─────────────────────────────────────────────────────────────
#  DECODER
# ─────────────────────────────────────────────────────────────

class TransformerDecoderStage(nn.Module):
    """
    Decoder stage: upsample tokens + add skip connection + SwinBlocks.
    Mirrors the SwinStage3D in the encoder for symmetric U-Net-like design.
    """

    def __init__(
        self,
        dim:         int,
        skip_dim:    int,
        out_dim:     int,
        depth:       int,
        n_heads:     int,
        window_size: Tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        super().__init__()
        self.upsample  = nn.Linear(dim, out_dim * 8)    # token expansion
        self.skip_proj = nn.Linear(skip_dim, out_dim)
        self.norm      = nn.LayerNorm(out_dim)
        self.blocks    = nn.ModuleList([
            SwinTransformerBlock3D(
                dim=out_dim,
                n_heads=n_heads,
                window_size=window_size,
                shift_size=(0, 0, 0) if i % 2 == 0 else tuple(
                    s // 2 for s in window_size
                ),
            )
            for i in range(depth)
        ])

    def forward(
        self,
        x:         torch.Tensor,
        skip:      torch.Tensor,
        grid:      Tuple[int, int, int],
        skip_grid: Tuple[int, int, int],
    ) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        B, L, C = x.shape
        nI, nX, nT = grid

        # Expand tokens (trilinear-equivalent via reshape)
        x = self.upsample(x)                                     # [B, L, out*8]
        x = x.view(B, nI, nX, nT, -1, 2, 2, 2)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        new_grid = (nI * 2, nX * 2, nT * 2)
        x = x.view(B, new_grid[0] * new_grid[1] * new_grid[2], -1)

        # Skip connection
        skip = self.skip_proj(skip)
        x    = self.norm(x + skip)

        for blk in self.blocks:
            x = blk(x, new_grid)

        return x, new_grid


# ─────────────────────────────────────────────────────────────
#  SEISMIC TRANSFORMER  (full model)
# ─────────────────────────────────────────────────────────────

class SeismicTransformer(nn.Module):
    """
    Swin3D-inspired encoder-decoder for volumetric seismic segmentation.

    Architecture summary:
        PatchEmbed3D       → [B, N, embed_dim]
        SwinStage x4       → hierarchical feature extraction with
                             shifted-window attention + patch merging
        TransformerDecoder → symmetric upsampling with skip connections
        Segmentation head  → linear projection → n_classes

    Advantages over UNet3D:
        • Long-range spatial dependencies via attention
        • Better at tracing laterally continuous horizons
        • More robust to noise via attention masking
        • Scales better to very large volumes

    Parameters
    ----------
    in_channels  : seismic input channels (1 = single amplitude stack)
    n_classes    : output segmentation classes
    embed_dim    : base embedding dimension (doubles per stage)
    depths       : transformer blocks per encoder stage
    n_heads      : attention heads per stage
    window_size  : local attention window size [I, X, T]
    patch_size   : initial patch embedding size
    mlp_ratio    : FFN hidden dim multiplier
    drop_rate    : dropout rate
    """

    def __init__(
        self,
        in_channels: int                      = 1,
        n_classes:   int                      = 3,
        embed_dim:   int                      = 48,
        depths:      Tuple[int, ...]          = (2, 2, 4, 2),
        n_heads:     Tuple[int, ...]          = (3, 6, 12, 24),
        window_size: Tuple[int, int, int]     = (4, 4, 4),
        patch_size:  Tuple[int, int, int]     = (4, 4, 4),
        mlp_ratio:   float                    = 4.0,
        drop_rate:   float                    = 0.0,
        attn_drop:   float                    = 0.0,
    ) -> None:
        super().__init__()
        assert len(depths) == len(n_heads), "depths and n_heads must have same length"

        n_stages = len(depths)
        dims     = [embed_dim * (2 ** i) for i in range(n_stages)]

        # ── Patch embedding ──────────────────────────────────────
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        self.pos_drop    = nn.Dropout(drop_rate)

        # ── Encoder ──────────────────────────────────────────────
        self.encoder_stages = nn.ModuleList()
        for i, (d, h) in enumerate(zip(depths, n_heads)):
            self.encoder_stages.append(
                SwinStage3D(
                    dim=dims[i],
                    depth=d,
                    n_heads=h,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop,
                    downsample=(i < n_stages - 1),
                )
            )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder_stages = nn.ModuleList()
        dec_depths = list(reversed(depths[:-1]))
        dec_heads  = list(reversed(n_heads[:-1]))
        for i in range(n_stages - 1):
            in_d   = dims[n_stages - 1 - i]
            skip_d = dims[n_stages - 2 - i]
            out_d  = dims[n_stages - 2 - i]
            self.decoder_stages.append(
                TransformerDecoderStage(
                    dim=in_d,
                    skip_dim=skip_d,
                    out_dim=out_d,
                    depth=dec_depths[i],
                    n_heads=dec_heads[i],
                    window_size=window_size,
                )
            )

        # ── Segmentation head ────────────────────────────────────
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head      = nn.Linear(embed_dim, n_classes)

        self._patch_size = patch_size
        self._n_classes  = n_classes
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, C, I, X, T]

        Returns
        -------
        logits : [B, n_classes, I, X, T]
        """
        B, C, I, X, T = x.shape

        # Patch embedding
        x, grid = self.patch_embed(x)
        x = self.pos_drop(x)

        # Encoder — collect skip connections
        skips  = []
        grids  = []
        for stage in self.encoder_stages[:-1]:
            skips.append((x, grid))
            grids.append(grid)
            x, grid = stage(x, grid)

        # Bottleneck (last encoder stage, no downsampling)
        x, grid = self.encoder_stages[-1](x, grid)

        # Decoder
        for i, dec in enumerate(self.decoder_stages):
            skip_x, skip_grid = skips[-(i + 1)]
            x, grid = dec(x, skip_x, grid, skip_grid)

        # Head — project to n_classes then reshape to volume
        x      = self.head_norm(x)
        x      = self.head(x)                   # [B, N, n_classes]
        nI, nX, nT = grid
        x      = x.view(B, nI, nX, nT, self._n_classes)
        x      = x.permute(0, 4, 1, 2, 3)       # [B, n_classes, nI, nX, nT]

        # Upsample back to input resolution
        x = F.interpolate(
            x.float(),
            size=(I, X, T),
            mode="trilinear",
            align_corners=False,
        )
        return x


# ─────────────────────────────────────────────────────────────
#  SEISMIC ViT  (lightweight alternative)
# ─────────────────────────────────────────────────────────────

class SeismicViTBlock(nn.Module):
    """Standard ViT transformer block: LayerNorm → MHA → LN → MLP."""

    def __init__(
        self,
        dim:       int,
        n_heads:   int,
        mlp_ratio: float = 4.0,
        drop:      float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(
            dim, n_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(dim)
        hidden     = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.norm1(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class SeismicViT(nn.Module):
    """
    Lightweight 3D Vision Transformer for seismic patch segmentation.

    Treats each 3D patch as a token with global self-attention.
    Simpler and faster than SeismicTransformer — ideal for:
        • Smaller datasets
        • Quick benchmarking against U-Net
        • Fine-tuning from pretrained ViT weights

    Parameters
    ----------
    in_channels : seismic input channels
    n_classes   : output segmentation classes
    patch_size  : 3D patch token size
    embed_dim   : token embedding dimension
    depth       : number of transformer blocks
    n_heads     : attention heads
    """

    def __init__(
        self,
        in_channels: int                  = 1,
        n_classes:   int                  = 3,
        patch_size:  Tuple[int, int, int] = (4, 4, 4),
        embed_dim:   int                  = 192,
        depth:       int                  = 8,
        n_heads:     int                  = 6,
        mlp_ratio:   float                = 4.0,
        drop_rate:   float                = 0.0,
        attn_drop:   float                = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop    = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            SeismicViTBlock(embed_dim, n_heads, mlp_ratio, drop_rate, attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self._patch_size = patch_size
        self._n_classes  = n_classes

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, I, X, T = x.shape

        tokens, grid = self.patch_embed(x)          # [B, N, D]
        tokens = self.pos_drop(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)
        tokens = self.head(tokens)                  # [B, N, n_classes]

        nI, nX, nT = grid
        out = tokens.view(B, nI, nX, nT, self._n_classes)
        out = out.permute(0, 4, 1, 2, 3)            # [B, n_classes, nI, nX, nT]

        out = F.interpolate(
            out.float(),
            size=(I, X, T),
            mode="trilinear",
            align_corners=False,
        )
        return out


# ─────────────────────────────────────────────────────────────
#  MODEL COMPARISON UTILITY
# ─────────────────────────────────────────────────────────────

def compare_models(
    patch_size: Tuple[int, int, int] = (32, 32, 32),
    n_classes:  int = 3,
    device:     str = "cpu",
) -> None:
    """
    Print a side-by-side comparison of UNet3D, SeismicTransformer,
    and SeismicViT — parameter count, output shape, forward pass time.

    Usage:
        from seismic_ml.models.transformer import compare_models
        compare_models(patch_size=(32, 32, 32))
    """
    import time
    from seismic_ml.models.unet3d import UNet3D

    dummy = torch.randn(1, 1, *patch_size).to(device)
    models = {
        "UNet3D":             UNet3D(n_classes=n_classes, base_features=16, depth=3),
        "SeismicTransformer": SeismicTransformer(
                                  n_classes=n_classes,
                                  embed_dim=32,
                                  depths=(2, 2, 2, 2),
                                  n_heads=(2, 4, 8, 16),
                                  window_size=(4, 4, 4),
                              ),
        "SeismicViT":         SeismicViT(
                                  n_classes=n_classes,
                                  embed_dim=64,
                                  depth=4,
                                  n_heads=4,
                              ),
    }

    print(f"\n{'Model':25s} {'Params':>12s} {'Output Shape':>20s} {'ms/pass':>10s}")
    print("-" * 72)

    for name, model in models.items():
        model = model.to(device).eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        with torch.no_grad():
            t0  = time.perf_counter()
            out = model(dummy)
            t1  = time.perf_counter()

        print(
            f"{name:25s} "
            f"{n_params:>12,} "
            f"{str(tuple(out.shape)):>20s} "
            f"{(t1 - t0) * 1000:>10.1f}"
        )

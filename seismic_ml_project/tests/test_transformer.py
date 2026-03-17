"""
tests/test_transformer.py
=========================
Unit tests for SeismicTransformer and SeismicViT.

Run with:
    pytest tests/test_transformer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seismic_ml.models.transformer import (
    PatchEmbed3D,
    PatchMerging3D,
    SeismicTransformer,
    SeismicViT,
    SwinTransformerBlock3D,
    WindowAttention3D,
    window_partition_3d,
    window_reverse_3d,
)
from seismic_ml.models.unet3d import UNet3D
from seismic_ml.models import build_model


# ─────────────────────────────────────────────
#  PatchEmbed3D
# ─────────────────────────────────────────────

class TestPatchEmbed3D:

    def test_output_shape(self):
        embed = PatchEmbed3D(in_channels=1, patch_size=(4, 4, 4), embed_dim=96)
        x     = torch.randn(2, 1, 32, 32, 32)
        out, grid = embed(x)
        assert out.shape == (2, 8 * 8 * 8, 96)
        assert grid == (8, 8, 8)

    def test_handles_non_divisible_input(self):
        """Should pad and not crash on volumes not divisible by patch_size."""
        embed = PatchEmbed3D(in_channels=1, patch_size=(4, 4, 4), embed_dim=64)
        x     = torch.randn(1, 1, 30, 30, 30)   # not divisible by 4
        out, grid = embed(x)
        assert out.ndim == 3
        assert out.shape[-1] == 64

    def test_output_is_normalized(self):
        embed = PatchEmbed3D(in_channels=1, patch_size=(4, 4, 4), embed_dim=64)
        x     = torch.randn(1, 1, 16, 16, 16)
        out, _ = embed(x)
        # LayerNorm output should have reasonable mean
        assert abs(float(out.mean())) < 1.0


# ─────────────────────────────────────────────
#  PatchMerging3D
# ─────────────────────────────────────────────

class TestPatchMerging3D:

    def test_halves_spatial_resolution(self):
        merge = PatchMerging3D(dim=96)
        x     = torch.randn(2, 8 * 8 * 8, 96)
        out, grid = merge(x, (8, 8, 8))
        assert grid == (4, 4, 4)
        assert out.shape == (2, 4 * 4 * 4, 192)

    def test_doubles_channels(self):
        merge = PatchMerging3D(dim=48)
        x     = torch.randn(1, 4 * 4 * 4, 48)
        out, _ = merge(x, (4, 4, 4))
        assert out.shape[-1] == 96


# ─────────────────────────────────────────────
#  Window partition / reverse
# ─────────────────────────────────────────────

class TestWindowPartition:

    def test_partition_reverse_roundtrip(self):
        x   = torch.randn(2, 8, 8, 8, 32)
        ws  = (4, 4, 4)
        win, n_wins = window_partition_3d(x, ws)
        rec = window_reverse_3d(win, ws, n_wins)
        assert rec.shape == x.shape
        assert torch.allclose(rec, x, atol=1e-5)

    def test_window_token_count(self):
        x   = torch.randn(1, 8, 8, 8, 16)
        ws  = (4, 4, 4)
        win, _ = window_partition_3d(x, ws)
        n_windows   = (8 // 4) ** 3           # 2^3 = 8
        tokens_each = 4 * 4 * 4              # 64
        assert win.shape == (n_windows, tokens_each, 16)


# ─────────────────────────────────────────────
#  WindowAttention3D
# ─────────────────────────────────────────────

class TestWindowAttention3D:

    def test_output_shape(self):
        ws   = (4, 4, 4)
        attn = WindowAttention3D(dim=32, window_size=ws, n_heads=4)
        N    = 4 * 4 * 4
        x    = torch.randn(8, N, 32)          # 8 windows
        out  = attn(x)
        assert out.shape == x.shape

    def test_masked_attention(self):
        ws   = (4, 4, 4)
        attn = WindowAttention3D(dim=32, window_size=ws, n_heads=4)
        N    = 4 * 4 * 4
        x    = torch.randn(8, N, 32)
        mask = torch.zeros(8, N, N)
        out  = attn(x, mask=mask)
        assert out.shape == x.shape


# ─────────────────────────────────────────────
#  SwinTransformerBlock3D
# ─────────────────────────────────────────────

class TestSwinTransformerBlock3D:

    def test_no_shift_output_shape(self):
        blk  = SwinTransformerBlock3D(dim=32, n_heads=4, window_size=(4, 4, 4))
        x    = torch.randn(2, 8 * 8 * 8, 32)
        out  = blk(x, (8, 8, 8))
        assert out.shape == x.shape

    def test_shifted_output_shape(self):
        blk  = SwinTransformerBlock3D(
            dim=32, n_heads=4,
            window_size=(4, 4, 4),
            shift_size=(2, 2, 2),
        )
        x   = torch.randn(2, 8 * 8 * 8, 32)
        out = blk(x, (8, 8, 8))
        assert out.shape == x.shape

    def test_residual_connection_changes_output(self):
        blk  = SwinTransformerBlock3D(dim=16, n_heads=2, window_size=(4, 4, 4))
        x    = torch.randn(1, 4 * 4 * 4, 16)
        out  = blk(x, (4, 4, 4))
        assert not torch.allclose(x, out), "Block should change output via attention+MLP"


# ─────────────────────────────────────────────
#  SeismicTransformer
# ─────────────────────────────────────────────

class TestSeismicTransformer:

    @pytest.fixture
    def small_model(self):
        return SeismicTransformer(
            in_channels=1,
            n_classes=3,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            n_heads=(1, 2, 4, 8),
            window_size=(4, 4, 4),
            patch_size=(4, 4, 4),
        )

    def test_output_shape(self, small_model):
        x   = torch.randn(1, 1, 32, 32, 32)
        out = small_model(x)
        assert out.shape == (1, 3, 32, 32, 32), f"Got {out.shape}"

    def test_output_shape_non_square(self, small_model):
        x   = torch.randn(1, 1, 32, 32, 64)
        out = small_model(x)
        assert out.shape == (1, 3, 32, 32, 64)

    def test_batch_independence(self, small_model):
        """Two samples in a batch should produce independent outputs."""
        small_model.eval()
        with torch.no_grad():
            x1 = torch.randn(1, 1, 32, 32, 32)
            x2 = torch.randn(1, 1, 32, 32, 32)
            o1 = small_model(x1)
            o2 = small_model(x2)
            batch = small_model(torch.cat([x1, x2], dim=0))
        assert torch.allclose(batch[0], o1[0], atol=1e-4)
        assert torch.allclose(batch[1], o2[0], atol=1e-4)

    def test_parameter_count_reasonable(self, small_model):
        n = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
        assert n > 1_000,    f"Model too small ({n:,} params)"
        assert n < 50_000_000, f"Model too large ({n:,} params) for smoke test"

    def test_gradient_flows(self, small_model):
        x    = torch.randn(1, 1, 32, 32, 32, requires_grad=False)
        out  = small_model(x)
        loss = out.mean()
        loss.backward()
        # Check at least one parameter has a gradient
        grads = [p.grad for p in small_model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"


# ─────────────────────────────────────────────
#  SeismicViT
# ─────────────────────────────────────────────

class TestSeismicViT:

    @pytest.fixture
    def small_vit(self):
        return SeismicViT(
            in_channels=1,
            n_classes=3,
            patch_size=(4, 4, 4),
            embed_dim=32,
            depth=2,
            n_heads=2,
        )

    def test_output_shape(self, small_vit):
        x   = torch.randn(2, 1, 32, 32, 32)
        out = small_vit(x)
        assert out.shape == (2, 3, 32, 32, 32)

    def test_single_sample(self, small_vit):
        x   = torch.randn(1, 1, 16, 16, 16)
        out = small_vit(x)
        assert out.shape == (1, 3, 16, 16, 16)

    def test_gradient_flows(self, small_vit):
        x    = torch.randn(1, 1, 16, 16, 16)
        out  = small_vit(x)
        out.mean().backward()
        grads = [p.grad for p in small_vit.parameters() if p.grad is not None]
        assert len(grads) > 0


# ─────────────────────────────────────────────
#  build_model factory
# ─────────────────────────────────────────────

class TestBuildModel:

    def test_unet3d(self):
        m = build_model("unet3d", n_classes=3, base_features=8, depth=2)
        assert isinstance(m, UNet3D)

    def test_transformer(self):
        m = build_model(
            "transformer", n_classes=3,
            embed_dim=16,
            depths=(1, 1, 1, 1),
            n_heads=(1, 2, 4, 8),
        )
        assert isinstance(m, SeismicTransformer)

    def test_vit(self):
        m = build_model("vit", n_classes=3, embed_dim=32, depth=2, n_heads=2)
        assert isinstance(m, SeismicViT)

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_model("resnet", n_classes=3)

    def test_all_models_forward(self):
        x = torch.randn(1, 1, 16, 16, 16)
        for arch, kwargs in [
            ("unet3d",      {"base_features": 8, "depth": 2}),
            ("transformer", {"embed_dim": 16, "depths": (1, 1, 1, 1),
                             "n_heads": (1, 2, 4, 8), "window_size": (4, 4, 4)}),
            ("vit",         {"embed_dim": 32, "depth": 2, "n_heads": 2}),
        ]:
            m   = build_model(arch, n_classes=3, **kwargs)
            out = m(x)
            assert out.shape == (1, 3, 16, 16, 16), \
                f"{arch} output shape mismatch: {out.shape}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

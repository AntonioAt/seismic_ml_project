"""
inference.py
============
SlidingWindowInference — Overlapping patch-based inference over full seismic volumes.
"""
from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class SlidingWindowInference:

    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        overlap: float = 0.5,
        device: Optional[torch.device] = None,
        batch_size: int = 2,
    ) -> None:
        self.model      = model
        self.patch_size = patch_size
        self.stride     = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)
        self.device     = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

    def _get_starts(self, vol_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
        starts = []
        for i in range(0, max(1, vol_shape[0] - self.patch_size[0] + 1), self.stride[0]):
            for x in range(0, max(1, vol_shape[1] - self.patch_size[1] + 1), self.stride[1]):
                for t in range(0, max(1, vol_shape[2] - self.patch_size[2] + 1), self.stride[2]):
                    starts.append((i, x, t))
        return starts

    @torch.no_grad()
    def run_inference(
        self,
        cube: np.ndarray,
        n_classes: int = 3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        vol_shape  = cube.shape
        prob_sum   = np.zeros((n_classes, *vol_shape), dtype=np.float32)
        count_map  = np.zeros(vol_shape, dtype=np.float32)
        pi, px, pt = self.patch_size
        starts     = self._get_starts(vol_shape)

        buf_patches: List[np.ndarray]           = []
        buf_coords:  List[Tuple[int, int, int]] = []

        def flush():
            if not buf_patches:
                return
            batch  = np.stack(buf_patches)[:, np.newaxis]
            t      = torch.from_numpy(batch).to(self.device)
            probs  = torch.softmax(self.model(t), dim=1).cpu().numpy()
            for b, (si, sx, st) in enumerate(buf_coords):
                prob_sum[:, si:si+pi, sx:sx+px, st:st+pt] += probs[b]
                count_map[si:si+pi, sx:sx+px, st:st+pt]   += 1.0
            buf_patches.clear()
            buf_coords.clear()

        for si, sx, st in starts:
            patch = cube[si:si+pi, sx:sx+px, st:st+pt].copy().astype(np.float32)
            if patch.shape != (pi, px, pt):
                padded = np.zeros((pi, px, pt), dtype=np.float32)
                padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                patch = padded
            buf_patches.append(patch)
            buf_coords.append((si, sx, st))
            if len(buf_patches) >= self.batch_size:
                flush()

        flush()

        count_map  = np.maximum(count_map, 1.0)
        avg_probs  = prob_sum / count_map[np.newaxis]
        seg_map    = avg_probs.argmax(axis=0).astype(np.uint8)
        conf_map   = avg_probs.max(axis=0)
        return seg_map, conf_map

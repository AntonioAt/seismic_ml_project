"""
dataset.py
==========
SeismicPatchDataset + build_dataloader — PyTorch Dataset for 3D seismic patches.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class SeismicPatchDataset(Dataset):

    def __init__(
        self,
        cube: np.ndarray,
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        n_patches: int = 1000,
        labels: Optional[np.ndarray] = None,
        augment: bool = True,
    ) -> None:
        self.cube       = cube
        self.labels     = labels
        self.patch_size = patch_size
        self.n_patches  = n_patches
        self.augment    = augment
        self._max_starts = tuple(
            max(0, s - p) for s, p in zip(cube.shape, patch_size)
        )

    def __len__(self) -> int:
        return self.n_patches

    def _random_patch(self, arr: np.ndarray, starts: Tuple[int, int, int]) -> np.ndarray:
        si, sx, st = starts
        pi, px, pt = self.patch_size
        return arr[si:si+pi, sx:sx+px, st:st+pt]

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        starts = tuple(int(np.random.randint(0, m + 1)) for m in self._max_starts)
        patch  = self._random_patch(self.cube, starts).copy()

        if self.augment:
            if np.random.rand() > 0.5:
                patch = patch[::-1].copy()
            patch += np.random.normal(0, 0.01, patch.shape).astype(np.float32)

        t_patch = torch.from_numpy(patch).unsqueeze(0)

        if self.labels is not None:
            label   = self._random_patch(self.labels, starts).copy()
            t_label = torch.from_numpy(label).long()
            return t_patch, t_label

        return t_patch


def build_dataloader(
    dataset: SeismicPatchDataset,
    batch_size: int = 4,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        drop_last   = True,
    )

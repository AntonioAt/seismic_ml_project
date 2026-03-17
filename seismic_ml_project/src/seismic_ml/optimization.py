"""
optimization.py
===============
PerformanceOptimizer — torch.compile, AMP, cuDNN tuning, vectorised NumPy utilities.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class PerformanceOptimizer:

    @staticmethod
    def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
        if hasattr(torch, "compile"):
            model = torch.compile(model, mode=mode)
            print(f"[PerformanceOptimizer] torch.compile applied (mode={mode})")
        else:
            print("[PerformanceOptimizer] torch.compile unavailable — skipped")
        return model

    @staticmethod
    def configure_backends(
        cudnn_benchmark: bool = True,
        allow_tf32: bool = True,
    ) -> None:
        torch.backends.cudnn.benchmark  = cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        print(f"[PerformanceOptimizer] cuDNN benchmark={cudnn_benchmark}  TF32={allow_tf32}")

    @staticmethod
    def amp_train_one_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        grad_clip: float = 1.0,
        scaler: Optional["torch.cuda.amp.GradScaler"] = None,
    ) -> float:
        use_amp = device.type == "cuda"
        if scaler is None and use_amp:
            scaler = torch.cuda.amp.GradScaler()
        model.train()
        total = 0.0
        for patches, labels in loader:
            patches = patches.to(device, non_blocking=True)
            labels  = labels.to(device,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(patches), labels)
            if use_amp and scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            total += loss.item()
        return total / max(len(loader), 1)

    @staticmethod
    def vectorised_energy_attribute(cube: np.ndarray, window: int = 9) -> np.ndarray:
        from numpy.lib.stride_tricks import sliding_window_view
        n_il, n_xl, n_t = cube.shape
        pad    = window // 2
        padded = np.pad(cube, ((0,0),(0,0),(pad,pad)), mode="edge")
        wins   = sliding_window_view(padded, window_shape=window, axis=2)
        return np.sqrt((wins ** 2).mean(axis=-1)).astype(np.float32)

    @staticmethod
    def vectorised_semblance(cube: np.ndarray, half_window: int = 2) -> np.ndarray:
        n_il, n_xl, n_t = cube.shape
        w      = 2 * half_window + 1
        padded = np.pad(cube, ((0,0),(half_window,half_window),(0,0)), mode="edge")
        stack  = np.stack([padded[:, i:i+n_xl, :] for i in range(w)], axis=2)
        num    = stack.sum(axis=2) ** 2
        den    = (stack ** 2).sum(axis=2) * w + 1e-10
        return (num / den).astype(np.float32)

    @staticmethod
    def gpu_memory_report() -> Dict[str, float]:
        if not torch.cuda.is_available():
            print("[PerformanceOptimizer] CUDA not available")
            return {}
        alloc = torch.cuda.memory_allocated()  / 1024**3
        res   = torch.cuda.memory_reserved()   / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[GPU] allocated={alloc:.2f}GiB  reserved={res:.2f}GiB  "
              f"free={total-res:.2f}GiB  total={total:.2f}GiB")
        return {"allocated_GiB": alloc, "reserved_GiB": res,
                "free_GiB": total-res, "total_GiB": total}

    @staticmethod
    def build_optimized_dataloader(
        dataset: Dataset,
        batch_size: int = 4,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
    ) -> DataLoader:
        pin  = torch.cuda.is_available()
        kw: Dict = {"batch_size": batch_size, "shuffle": True,
                    "num_workers": num_workers, "pin_memory": pin, "drop_last": True}
        if num_workers > 0:
            kw["prefetch_factor"]    = prefetch_factor
            kw["persistent_workers"] = persistent_workers
        return DataLoader(dataset, **kw)

"""
ingestion.py
============
SeismicIngestion — Data loading from SEG-Y, NumPy, and HDF5 formats.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np


class SeismicIngestion:

    @staticmethod
    def load_segy(
        path: str,
        inline_byte: int = 189,
        crossline_byte: int = 193,
        chunk_size: int = 50,
    ) -> np.ndarray:
        import segyio
        with segyio.open(path, "r", iline=inline_byte, xline=crossline_byte, strict=False) as f:
            f.mmap()
            n_il    = len(f.ilines)
            n_xl    = len(f.xlines)
            n_t     = f.samples.size
            cube    = np.zeros((n_il, n_xl, n_t), dtype=np.float32)
            for start in range(0, n_il, chunk_size):
                end = min(start + chunk_size, n_il)
                for idx, iline in enumerate(f.ilines[start:end]):
                    cube[start + idx] = f.iline[iline]
        return cube

    @staticmethod
    def load_numpy(path: str, mmap: bool = True) -> np.ndarray:
        mode = "r" if mmap else "r+"
        return np.load(path, mmap_mode=mode)

    @staticmethod
    def load_hdf5(path: str, dataset_key: str = "seismic", chunk_size: int = 50) -> np.ndarray:
        import h5py
        with h5py.File(path, "r") as f:
            ds    = f[dataset_key]
            shape = ds.shape
            cube  = np.zeros(shape, dtype=np.float32)
            for start in range(0, shape[0], chunk_size):
                end = min(start + chunk_size, shape[0])
                cube[start:end] = ds[start:end]
        return cube

    @staticmethod
    def load_seismic_data(path: str, fmt: Optional[str] = None, **kwargs) -> np.ndarray:
        p   = Path(path)
        ext = (fmt or p.suffix).lower()
        if ext in (".segy", ".sgy"):
            return SeismicIngestion.load_segy(path, **kwargs)
        elif ext == ".npy":
            return SeismicIngestion.load_numpy(path)
        elif ext in (".hdf5", ".h5"):
            return SeismicIngestion.load_hdf5(path, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {ext}")

"""
preprocessing.py
================
SeismicPreprocessor — Signal filtering and trace normalisation.
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt


class SeismicPreprocessor:

    @staticmethod
    def bandpass_filter(
        cube: np.ndarray,
        low_hz: float,
        high_hz: float,
        sample_rate_ms: float = 2.0,
    ) -> np.ndarray:
        nyq  = 1000.0 / (2.0 * sample_rate_ms)
        sos  = butter(4, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
        n_il, n_xl, n_t = cube.shape
        flat = cube.reshape(-1, n_t)
        return sosfilt(sos, flat, axis=-1).reshape(n_il, n_xl, n_t).astype(np.float32)

    @staticmethod
    def gaussian_smooth(
        cube: np.ndarray,
        sigma: Union[float, Tuple[float, float, float]] = 1.0,
    ) -> np.ndarray:
        return gaussian_filter(cube, sigma=sigma).astype(np.float32)

    @staticmethod
    def normalize_traces(cube: np.ndarray) -> np.ndarray:
        flat = cube.reshape(-1, cube.shape[-1])
        mean = flat.mean(axis=1, keepdims=True)
        std  = flat.std(axis=1, keepdims=True).clip(min=1e-8)
        return ((flat - mean) / std).astype(np.float32).reshape(cube.shape)

    @staticmethod
    def attenuate_noise(cube: np.ndarray, threshold_percentile: float = 98.0) -> np.ndarray:
        thresh = np.percentile(np.abs(cube), threshold_percentile)
        cube   = np.clip(cube, -thresh, thresh)
        return SeismicPreprocessor.gaussian_smooth(cube, sigma=0.5)

    @staticmethod
    def scale_amplitudes(
        cube: np.ndarray,
        target_range: Tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        lo, hi  = cube.min(), cube.max()
        span    = hi - lo if (hi - lo) > 1e-8 else 1e-8
        t_lo, t_hi = target_range
        return ((cube - lo) / span * (t_hi - t_lo) + t_lo).astype(np.float32)

    def preprocess_traces(
        self,
        cube: np.ndarray,
        bandpass: Optional[Tuple[float, float]] = (5.0, 120.0),
        smooth_sigma: float = 0.8,
        noise_attenuation: bool = True,
        amplitude_scaling: bool = False,
        sample_rate_ms: float = 2.0,
    ) -> np.ndarray:
        if bandpass:
            cube = self.bandpass_filter(cube, *bandpass, sample_rate_ms)
        cube = self.normalize_traces(cube)
        if noise_attenuation:
            cube = self.attenuate_noise(cube)
        cube = self.gaussian_smooth(cube, sigma=smooth_sigma)
        if amplitude_scaling:
            cube = self.scale_amplitudes(cube)
        return cube

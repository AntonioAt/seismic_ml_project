"""
reservoir.py
============
ReservoirPredictor — Priority #1 Module

Detects hydrocarbon-bearing formations from seismic segmentation output
using amplitude anomaly analysis, AVO proxy attributes, and structural
trap geometry recognition.

Domain concepts implemented:
    - Bright spot / dim spot detection       (amplitude anomaly)
    - AVO proxy via near/far offset gradient (fluid indicator)
    - Flat spot detection                    (gas-water contact)
    - DHI (Direct Hydrocarbon Indicator) scoring
    - Structural trap geometry recognition   (anticline, fault-bounded)
    - Reservoir zone ranking by confidence

Usage:
    from seismic_ml.reservoir import ReservoirPredictor

    predictor = ReservoirPredictor()
    results   = predictor.predict(
        cube     = seismic_cube,       # np.ndarray [IL, XL, T]
        seg_map  = segmentation_map,   # np.ndarray [IL, XL, T] uint8
        conf_map = confidence_map,     # np.ndarray [IL, XL, T] float32
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    gaussian_filter,
    label as nd_label,
)
from scipy.signal import hilbert


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class ReservoirZone:
    """Single detected reservoir candidate."""

    zone_id:          int
    centroid_il:      float
    centroid_xl:      float
    centroid_t:       float
    bbox:             Tuple[slice, slice, slice]   # bounding box in cube
    voxel_count:      int
    dhi_score:        float          # 0–1  Direct Hydrocarbon Indicator
    amplitude_anomaly: float         # normalised bright/dim spot strength
    avo_gradient:     float          # AVO proxy — negative = Class III AVO
    flat_spot_score:  float          # 0–1  gas-water contact indicator
    trap_type:        str            # "anticline" | "fault-bounded" | "stratigraphic"
    confidence:       float          # overall reservoir confidence 0–1

    def to_dict(self) -> Dict:
        return {
            "zone_id":           self.zone_id,
            "centroid":          {
                "inline":    round(self.centroid_il, 1),
                "crossline": round(self.centroid_xl, 1),
                "time_ms":   round(self.centroid_t,  1),
            },
            "voxel_count":       self.voxel_count,
            "dhi_score":         round(self.dhi_score,         3),
            "amplitude_anomaly": round(self.amplitude_anomaly, 3),
            "avo_gradient":      round(self.avo_gradient,      3),
            "flat_spot_score":   round(self.flat_spot_score,   3),
            "trap_type":         self.trap_type,
            "confidence":        round(self.confidence,        3),
        }


@dataclass
class ReservoirPrediction:
    """Full prediction output from ReservoirPredictor."""

    zones:             List[ReservoirZone]
    amplitude_map:     np.ndarray          # [IL, XL, T]  float32
    instantaneous_freq: np.ndarray         # [IL, XL, T]  float32
    envelope:          np.ndarray          # [IL, XL, T]  float32
    avo_proxy_map:     np.ndarray          # [IL, XL, T]  float32
    flat_spot_map:     np.ndarray          # [IL, XL, T]  float32
    dhi_map:           np.ndarray          # [IL, XL, T]  float32
    n_zones:           int = field(init=False)
    best_zone:         Optional[ReservoirZone] = field(init=False)

    def __post_init__(self) -> None:
        self.n_zones  = len(self.zones)
        self.best_zone = (
            max(self.zones, key=lambda z: z.dhi_score)
            if self.zones else None
        )

    def summary(self) -> str:
        lines = [
            f"ReservoirPrediction — {self.n_zones} zone(s) detected",
            "-" * 50,
        ]
        for z in sorted(self.zones, key=lambda x: -x.dhi_score):
            lines.append(
                f"  Zone {z.zone_id:02d}  |  DHI={z.dhi_score:.2f}  "
                f"conf={z.confidence:.2f}  trap={z.trap_type:15s}  "
                f"voxels={z.voxel_count:>8,}"
            )
        if self.best_zone:
            lines.append(f"\n  ★ Best candidate: Zone {self.best_zone.zone_id}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  SEISMIC ATTRIBUTE ENGINES
# ─────────────────────────────────────────────────────────────

class ComplexTraceAttributes:
    """
    Compute instantaneous seismic attributes via the Hilbert transform.

    All operations are fully vectorised — no Python loops over traces.
    Input shape: [IL, XL, T]
    """

    @staticmethod
    def compute_envelope(cube: np.ndarray) -> np.ndarray:
        """
        Instantaneous amplitude (reflection strength / envelope).
        High values → impedance contrast / potential bright spots.
        """
        analytic = hilbert(cube, axis=2)          # complex trace
        return np.abs(analytic).astype(np.float32)

    @staticmethod
    def compute_instantaneous_phase(cube: np.ndarray) -> np.ndarray:
        """Instantaneous phase in radians [-π, π]."""
        analytic = hilbert(cube, axis=2)
        return np.angle(analytic).astype(np.float32)

    @staticmethod
    def compute_instantaneous_frequency(
        cube: np.ndarray,
        sample_rate_ms: float = 2.0,
    ) -> np.ndarray:
        """
        Instantaneous frequency (Hz).
        Anomalously low frequencies below a horizon → hydrocarbon shadow.
        """
        phase    = ComplexTraceAttributes.compute_instantaneous_phase(cube)
        d_phase  = np.diff(np.unwrap(phase, axis=2), axis=2)
        freq     = d_phase / (2.0 * np.pi * sample_rate_ms * 1e-3)
        # Pad last sample to maintain shape
        freq     = np.concatenate([freq, freq[:, :, -1:]], axis=2)
        return freq.astype(np.float32)

    @staticmethod
    def compute_relative_acoustic_impedance(
        cube: np.ndarray,
        smooth_sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Approximate relative acoustic impedance via cumulative integration
        of seismic trace with low-frequency trend added back.
        Used as proxy for porosity / lithology contrast.
        """
        integrated = np.cumsum(cube, axis=2).astype(np.float32)
        trend      = gaussian_filter(integrated, sigma=smooth_sigma)
        return (integrated - trend).astype(np.float32)


class AmplitudeAnomalyDetector:
    """
    Detect amplitude anomalies relative to background level.

    Bright spots: amplitudes significantly above background → gas sands
    Dim spots:    amplitudes significantly below background → tight sands
    """

    def __init__(
        self,
        bright_threshold_sigma: float = 2.0,
        dim_threshold_sigma:    float = 2.0,
        background_window:      int   = 20,
    ) -> None:
        self.bright_thresh = bright_threshold_sigma
        self.dim_thresh    = dim_threshold_sigma
        self.bg_window     = background_window

    def compute_anomaly_map(
        self,
        envelope: np.ndarray,
    ) -> np.ndarray:
        """
        Return normalised anomaly map:
            > 0  →  bright spot (positive anomaly)
            < 0  →  dim spot    (negative anomaly)
        Shape: [IL, XL, T], float32, roughly in [-3, +3] sigma units.
        """
        # Background: smoothed envelope along time axis
        background = gaussian_filter(
            envelope,
            sigma=(0, 0, self.bg_window),
        )
        residual   = envelope - background

        # Normalise by local standard deviation
        local_std  = gaussian_filter(
            np.abs(residual),
            sigma=(2, 2, self.bg_window // 2),
        ).clip(min=1e-6)

        return (residual / local_std).astype(np.float32)

    def bright_spot_mask(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map >  self.bright_thresh

    def dim_spot_mask(self, anomaly_map: np.ndarray) -> np.ndarray:
        return anomaly_map < -self.dim_thresh


class AVOProxyAnalyzer:
    """
    AVO (Amplitude Versus Offset) proxy analysis.

    In real data, near and far offset stacks are compared.
    Here we approximate using:
        - Near  offset → low-frequency filtered cube (long wavelength)
        - Far   offset → high-frequency filtered cube (short wavelength)

    AVO gradient proxy = (far_amp - near_amp) / (near_amp + ε)
    Negative gradient in Class III AVO → gas-bearing sands.
    """

    @staticmethod
    def compute_avo_proxy(
        cube:            np.ndarray,
        near_sigma_t:    float = 4.0,
        far_sigma_t:     float = 1.0,
    ) -> np.ndarray:
        """
        Returns AVO gradient proxy map [IL, XL, T], float32.
        Values < -0.2 are indicative of Class III AVO response.
        """
        near_env = np.abs(
            gaussian_filter(cube, sigma=(0.5, 0.5, near_sigma_t))
        )
        far_env  = np.abs(
            gaussian_filter(cube, sigma=(0.5, 0.5, far_sigma_t))
        )
        denom    = near_env + 1e-6
        gradient = (far_env - near_env) / denom
        return gradient.astype(np.float32)


class FlatSpotDetector:
    """
    Flat spot detection — horizontal reflection that cuts across dipping
    strata. Indicates a gas-water or oil-water contact.

    Strategy:
        1. Compute horizontal coherence of the envelope
        2. High horizontal coherence + low vertical variance → flat spot
    """

    @staticmethod
    def compute_flat_spot_map(
        envelope:    np.ndarray,
        h_sigma:     float = 0.5,
        v_sigma:     float = 3.0,
        threshold:   float = 0.65,
    ) -> np.ndarray:
        """
        Returns flat spot probability map [IL, XL, T], float32 in [0, 1].
        """
        # Horizontal smoothness (high = laterally continuous = flat)
        h_smooth  = gaussian_filter(envelope, sigma=(h_sigma, h_sigma, 0))
        h_diff    = np.abs(envelope - h_smooth)
        h_score   = 1.0 - (h_diff / (h_diff.max() + 1e-8))

        # Vertical gradient (low = flat = no dip)
        v_grad    = np.abs(np.gradient(envelope, axis=2))
        v_norm    = gaussian_filter(v_grad, sigma=(1, 1, v_sigma))
        v_score   = 1.0 - (v_norm / (v_norm.max() + 1e-8))

        flat_map  = (h_score * v_score).astype(np.float32)
        flat_map  = (flat_map - flat_map.min()) / (flat_map.max() - flat_map.min() + 1e-8)
        return flat_map


# ─────────────────────────────────────────────────────────────
#  TRAP GEOMETRY CLASSIFIER
# ─────────────────────────────────────────────────────────────

class TrapGeometryClassifier:
    """
    Classifies structural trap type from segmentation map.

    Rules:
        anticline       — horizon label forms convex-upward shape
                          (centroid of horizon zone above median depth)
        fault-bounded   — fault label adjacent to horizon label
        stratigraphic   — horizon present but no nearby fault and no
                          convex geometry (pinch-out / unconformity)
    """

    @staticmethod
    def classify(
        zone_mask:  np.ndarray,    # boolean [IL, XL, T] for this zone
        seg_map:    np.ndarray,    # full segmentation [IL, XL, T]
        fault_class: int = 1,
        dilation_radius: int = 8,
    ) -> str:
        fault_vol    = seg_map == fault_class
        dilated_zone = binary_dilation(
            zone_mask,
            iterations=dilation_radius,
        )
        fault_nearby = bool((dilated_zone & fault_vol).any())

        if fault_nearby:
            return "fault-bounded"

        # Convexity check: do time indices decrease toward centre?
        coords = np.argwhere(zone_mask)
        if coords.size == 0:
            return "stratigraphic"

        t_coords   = coords[:, 2]
        t_median   = np.median(t_coords)
        t_centre   = t_coords[
            np.linalg.norm(coords[:, :2] - coords[:, :2].mean(axis=0), axis=1).argmin()
        ]
        if t_centre < t_median:
            return "anticline"

        return "stratigraphic"


# ─────────────────────────────────────────────────────────────
#  DHI SCORER
# ─────────────────────────────────────────────────────────────

class DHIScorer:
    """
    Combines multiple DHI (Direct Hydrocarbon Indicator) evidence lines
    into a single composite score per reservoir zone.

    Weights are tunable to match local geology / basin type.
    """

    def __init__(
        self,
        weight_amplitude:  float = 0.30,
        weight_avo:        float = 0.30,
        weight_flat_spot:  float = 0.25,
        weight_trap:       float = 0.15,
    ) -> None:
        assert abs(
            weight_amplitude + weight_avo + weight_flat_spot + weight_trap - 1.0
        ) < 1e-5, "DHI weights must sum to 1.0"

        self.w_amp   = weight_amplitude
        self.w_avo   = weight_avo
        self.w_flat  = weight_flat_spot
        self.w_trap  = weight_trap

        self._trap_scores = {
            "anticline":      1.0,
            "fault-bounded":  0.75,
            "stratigraphic":  0.5,
        }

    def score(
        self,
        amplitude_anomaly: float,
        avo_gradient:      float,
        flat_spot_score:   float,
        trap_type:         str,
    ) -> float:
        """
        Compute composite DHI score in [0, 1].

        amplitude_anomaly : positive (bright) is better for gas sands
        avo_gradient      : more negative is better (Class III AVO)
        flat_spot_score   : higher = more evidence of fluid contact
        trap_type         : categorical trap classification
        """
        # Normalise amplitude: bright spot → score near 1
        amp_score  = float(np.clip((amplitude_anomaly + 3.0) / 6.0, 0.0, 1.0))

        # AVO: negative gradient → score near 1  (range clipped to [-1, 0])
        avo_score  = float(np.clip(-avo_gradient, 0.0, 1.0))

        # Flat spot: already in [0, 1]
        flat_score = float(np.clip(flat_spot_score, 0.0, 1.0))

        # Trap geometry score
        trap_score = self._trap_scores.get(trap_type, 0.5)

        dhi = (
            self.w_amp  * amp_score  +
            self.w_avo  * avo_score  +
            self.w_flat * flat_score +
            self.w_trap * trap_score
        )
        return float(np.clip(dhi, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
#  MAIN RESERVOIR PREDICTOR
# ─────────────────────────────────────────────────────────────

class ReservoirPredictor:
    """
    End-to-end reservoir prediction from seismic cube + segmentation.

    Pipeline:
        1. Compute complex trace attributes (envelope, inst. freq.)
        2. Detect amplitude anomalies (bright/dim spots)
        3. Compute AVO proxy gradient
        4. Detect flat spots (fluid contacts)
        5. Label connected reservoir candidate zones from segmentation
        6. Per-zone: extract attributes, classify trap, score DHI
        7. Return ranked ReservoirPrediction

    Parameters
    ----------
    horizon_class       : segmentation class index for horizons
    min_zone_voxels     : discard zones smaller than this (noise filter)
    min_dhi_score       : minimum DHI score to include in output
    sample_rate_ms      : seismic sample interval in milliseconds
    inline_spacing_m    : inline spacing in metres (for coordinates)
    crossline_spacing_m : crossline spacing in metres
    dhi_weights         : dict to override default DHI scorer weights
    """

    def __init__(
        self,
        horizon_class:        int   = 0,
        min_zone_voxels:      int   = 100,
        min_dhi_score:        float = 0.25,
        sample_rate_ms:       float = 2.0,
        inline_spacing_m:     float = 25.0,
        crossline_spacing_m:  float = 25.0,
        dhi_weights:          Optional[Dict[str, float]] = None,
    ) -> None:
        self.horizon_class       = horizon_class
        self.min_zone_voxels     = min_zone_voxels
        self.min_dhi_score       = min_dhi_score
        self.sample_rate_ms      = sample_rate_ms
        self.inline_spacing_m    = inline_spacing_m
        self.crossline_spacing_m = crossline_spacing_m

        self._attr     = ComplexTraceAttributes()
        self._anomaly  = AmplitudeAnomalyDetector()
        self._avo      = AVOProxyAnalyzer()
        self._flat     = FlatSpotDetector()
        self._trap     = TrapGeometryClassifier()
        self._dhi      = DHIScorer(**(dhi_weights or {}))

    # ── Public API ──────────────────────────────────────────────

    def predict(
        self,
        cube:     np.ndarray,
        seg_map:  np.ndarray,
        conf_map: np.ndarray,
        min_confidence: float = 0.5,
    ) -> ReservoirPrediction:
        """
        Run full reservoir prediction pipeline.

        Parameters
        ----------
        cube     : seismic amplitude volume [IL, XL, T], float32
        seg_map  : segmentation volume      [IL, XL, T], uint8
        conf_map : model confidence         [IL, XL, T], float32

        Returns
        -------
        ReservoirPrediction dataclass with all attribute maps and
        a ranked list of ReservoirZone objects.
        """
        print("[ReservoirPredictor] Step 1/6 — Computing seismic attributes …")
        envelope   = self._attr.compute_envelope(cube)
        inst_freq  = self._attr.compute_instantaneous_frequency(
            cube, self.sample_rate_ms
        )

        print("[ReservoirPredictor] Step 2/6 — Amplitude anomaly detection …")
        anomaly_map = self._anomaly.compute_anomaly_map(envelope)

        print("[ReservoirPredictor] Step 3/6 — AVO proxy analysis …")
        avo_map     = self._avo.compute_avo_proxy(cube)

        print("[ReservoirPredictor] Step 4/6 — Flat spot detection …")
        flat_map    = self._flat.compute_flat_spot_map(envelope)

        print("[ReservoirPredictor] Step 5/6 — Labelling reservoir zones …")
        horizon_mask  = (
            (seg_map == self.horizon_class) &
            (conf_map >= min_confidence)
        )
        labeled, n_raw = nd_label(horizon_mask)
        print(f"                               {n_raw} raw connected components found")

        print("[ReservoirPredictor] Step 6/6 — Scoring zones …")
        zones = self._score_zones(
            labeled, n_raw,
            cube, seg_map, conf_map,
            anomaly_map, avo_map, flat_map,
        )

        # Build composite DHI map
        dhi_map = self._build_dhi_map(zones, labeled, cube.shape)

        prediction = ReservoirPrediction(
            zones              = zones,
            amplitude_map      = envelope,
            instantaneous_freq = inst_freq,
            envelope           = envelope,
            avo_proxy_map      = avo_map,
            flat_spot_map      = flat_map,
            dhi_map            = dhi_map,
        )

        print(f"\n{prediction.summary()}")
        return prediction

    # ── Internal helpers ────────────────────────────────────────

    def _score_zones(
        self,
        labeled:     np.ndarray,
        n_raw:       int,
        cube:        np.ndarray,
        seg_map:     np.ndarray,
        conf_map:    np.ndarray,
        anomaly_map: np.ndarray,
        avo_map:     np.ndarray,
        flat_map:    np.ndarray,
    ) -> List[ReservoirZone]:
        zones      = []
        zone_id    = 0

        for lbl in range(1, n_raw + 1):
            mask = labeled == lbl
            n    = int(mask.sum())

            if n < self.min_zone_voxels:
                continue

            coords = np.argwhere(mask)
            ci, cx, ct = coords.mean(axis=0)

            # Bounding box
            bbox = (
                slice(coords[:, 0].min(), coords[:, 0].max() + 1),
                slice(coords[:, 1].min(), coords[:, 1].max() + 1),
                slice(coords[:, 2].min(), coords[:, 2].max() + 1),
            )

            # Extract per-zone attribute means
            amp_anom  = float(anomaly_map[mask].mean())
            avo_grad  = float(avo_map[mask].mean())
            flat_scr  = float(flat_map[mask].mean())

            # Trap classification
            trap = self._trap.classify(mask, seg_map)

            # DHI score
            dhi = self._dhi.score(amp_anom, avo_grad, flat_scr, trap)

            if dhi < self.min_dhi_score:
                continue

            # Overall confidence: mean model confidence weighted by DHI
            model_conf = float(conf_map[mask].mean())
            overall    = float(np.clip(0.6 * model_conf + 0.4 * dhi, 0, 1))

            zones.append(ReservoirZone(
                zone_id           = zone_id,
                centroid_il       = float(ci)  * self.inline_spacing_m,
                centroid_xl       = float(cx)  * self.crossline_spacing_m,
                centroid_t        = float(ct)  * self.sample_rate_ms,
                bbox              = bbox,
                voxel_count       = n,
                dhi_score         = dhi,
                amplitude_anomaly = amp_anom,
                avo_gradient      = avo_grad,
                flat_spot_score   = flat_scr,
                trap_type         = trap,
                confidence        = overall,
            ))
            zone_id += 1

        # Rank by DHI score descending
        zones.sort(key=lambda z: -z.dhi_score)
        print(f"                               {len(zones)} zones passed DHI threshold")
        return zones

    @staticmethod
    def _build_dhi_map(
        zones:   List[ReservoirZone],
        labeled: np.ndarray,
        shape:   Tuple[int, int, int],
    ) -> np.ndarray:
        """Paint DHI score into a volume for visualisation."""
        dhi_vol = np.zeros(shape, dtype=np.float32)
        zone_lookup = {z.zone_id + 1: z.dhi_score for z in zones}
        for lbl, score in zone_lookup.items():
            dhi_vol[labeled == lbl] = score
        return dhi_vol

    # ── Convenience accessors ───────────────────────────────────

    def get_top_zones(
        self,
        prediction: ReservoirPrediction,
        n: int = 5,
    ) -> List[Dict]:
        """Return top-n zones as list of dicts sorted by DHI score."""
        return [z.to_dict() for z in prediction.zones[:n]]

    def get_drilling_candidates(
        self,
        prediction:    ReservoirPrediction,
        min_dhi:       float = 0.5,
        min_voxels:    int   = 500,
    ) -> List[Dict]:
        """
        Filter zones suitable for well placement recommendation.
        Returns list of zone dicts with centroid coordinates.
        """
        candidates = [
            z for z in prediction.zones
            if z.dhi_score >= min_dhi and z.voxel_count >= min_voxels
        ]
        return [z.to_dict() for z in candidates]

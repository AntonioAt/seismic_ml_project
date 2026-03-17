"""
risk.py
=======
RiskAssessor — Priority #3 Module

Evaluates drilling and operational risk for each zone in a seismic volume
by combining geological, geomechanical, and operational evidence lines
into structured risk scores with Low / Medium / High / Critical labels.

Risk categories assessed:
    1. Fault proximity hazard     — wellbore instability near fault planes
    2. Overpressure indicator     — abnormal amplitude / velocity shadow
    3. Gas hydrate risk           — shallow high-amplitude anomalies
    4. Shallow water flow         — high-porosity sand above reservoir
    5. Fault seal integrity       — leaking vs. sealing fault classification
    6. Structural dip hazard      — steep dip angle causing borehole deviation
    7. Composite drilling score   — weighted combination of all above

Usage:
    from seismic_ml.risk import RiskAssessor

    assessor = RiskAssessor()
    report   = assessor.assess(
        cube     = seismic_cube,
        seg_map  = segmentation_map,
        conf_map = confidence_map,
        reservoir_prediction = reservoir_pred,   # optional
    )
    print(report.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    gaussian_filter,
    label as nd_label,
    sobel,
)


# ─────────────────────────────────────────────────────────────
#  ENUMS & CONSTANTS
# ─────────────────────────────────────────────────────────────

class RiskLevel(str, Enum):
    LOW      = "Low"
    MEDIUM   = "Medium"
    HIGH     = "High"
    CRITICAL = "Critical"

    @classmethod
    def from_score(cls, score: float) -> "RiskLevel":
        """Map normalised score [0, 1] to risk level."""
        if score < 0.25:
            return cls.LOW
        elif score < 0.50:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.CRITICAL


RISK_COLORS: Dict[RiskLevel, str] = {
    RiskLevel.LOW:      "#2ecc71",   # green
    RiskLevel.MEDIUM:   "#f39c12",   # orange
    RiskLevel.HIGH:     "#e74c3c",   # red
    RiskLevel.CRITICAL: "#8e44ad",   # purple
}


# ─────────────────────────────────────────────────────────────
#  DATA CLASSES
# ─────────────────────────────────────────────────────────────

@dataclass
class HazardZone:
    """A single detected hazard region."""

    hazard_id:       int
    hazard_type:     str           # category name
    risk_level:      RiskLevel
    risk_score:      float         # 0–1
    centroid:        Tuple[float, float, float]   # (IL, XL, T) in metres/ms
    voxel_count:     int
    description:     str
    mitigation:      str           # recommended action

    def to_dict(self) -> Dict:
        return {
            "hazard_id":   self.hazard_id,
            "hazard_type": self.hazard_type,
            "risk_level":  self.risk_level.value,
            "risk_score":  round(self.risk_score, 3),
            "centroid": {
                "inline":    round(self.centroid[0], 1),
                "crossline": round(self.centroid[1], 1),
                "time_ms":   round(self.centroid[2], 1),
            },
            "voxel_count": self.voxel_count,
            "description": self.description,
            "mitigation":  self.mitigation,
        }


@dataclass
class ZoneRiskProfile:
    """Risk profile for a single reservoir / interest zone."""

    zone_id:               int
    composite_score:       float
    composite_level:       RiskLevel
    fault_proximity_score: float
    overpressure_score:    float
    hydrate_score:         float
    shallow_water_score:   float
    fault_seal_score:      float
    dip_hazard_score:      float
    associated_hazards:    List[HazardZone] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "zone_id":         self.zone_id,
            "composite_score": round(self.composite_score, 3),
            "composite_level": self.composite_level.value,
            "scores": {
                "fault_proximity": round(self.fault_proximity_score, 3),
                "overpressure":    round(self.overpressure_score,    3),
                "hydrate":         round(self.hydrate_score,         3),
                "shallow_water":   round(self.shallow_water_score,   3),
                "fault_seal":      round(self.fault_seal_score,      3),
                "dip_hazard":      round(self.dip_hazard_score,      3),
            },
            "n_hazards": len(self.associated_hazards),
        }


@dataclass
class RiskReport:
    """Full risk assessment output from RiskAssessor."""

    hazard_zones:       List[HazardZone]
    zone_profiles:      List[ZoneRiskProfile]
    fault_proximity_map: np.ndarray     # [IL, XL, T] float32  0–1
    overpressure_map:   np.ndarray      # [IL, XL, T] float32  0–1
    hydrate_risk_map:   np.ndarray      # [IL, XL, T] float32  0–1
    dip_map:            np.ndarray      # [IL, XL, T] float32  degrees
    composite_risk_map: np.ndarray      # [IL, XL, T] float32  0–1
    n_critical:         int = field(init=False)
    n_high:             int = field(init=False)

    def __post_init__(self) -> None:
        self.n_critical = sum(
            1 for h in self.hazard_zones if h.risk_level == RiskLevel.CRITICAL
        )
        self.n_high = sum(
            1 for h in self.hazard_zones if h.risk_level == RiskLevel.HIGH
        )

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  DRILLING RISK ASSESSMENT REPORT",
            "=" * 60,
            f"  Total hazard zones : {len(self.hazard_zones)}",
            f"  Critical           : {self.n_critical}",
            f"  High               : {self.n_high}",
            f"  Zone profiles      : {len(self.zone_profiles)}",
            "-" * 60,
        ]

        if self.hazard_zones:
            lines.append("  HAZARD ZONES (sorted by risk score):")
            for hz in sorted(self.hazard_zones, key=lambda h: -h.risk_score):
                lines.append(
                    f"  [{hz.risk_level.value:8s}]  "
                    f"{hz.hazard_type:25s}  "
                    f"score={hz.risk_score:.2f}  "
                    f"voxels={hz.voxel_count:>8,}"
                )

        if self.zone_profiles:
            lines.append("\n  ZONE RISK PROFILES:")
            for zp in sorted(self.zone_profiles, key=lambda z: -z.composite_score):
                lines.append(
                    f"  Zone {zp.zone_id:02d}  "
                    f"[{zp.composite_level.value:8s}]  "
                    f"composite={zp.composite_score:.2f}  "
                    f"hazards={len(zp.associated_hazards)}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def get_critical_zones(self) -> List[Dict]:
        return [
            h.to_dict() for h in self.hazard_zones
            if h.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)
        ]

    def get_safe_zones(self) -> List[Dict]:
        return [
            h.to_dict() for h in self.hazard_zones
            if h.risk_level == RiskLevel.LOW
        ]


# ─────────────────────────────────────────────────────────────
#  INDIVIDUAL RISK ANALYZERS
# ─────────────────────────────────────────────────────────────

class FaultProximityAnalyzer:
    """
    Computes a risk map based on proximity to detected fault planes.
    Closer to fault → higher risk of:
        • Wellbore instability
        • Lost circulation
        • Blowout via fault conduit

    Uses Euclidean distance transform from fault voxels,
    then converts to a risk score with exponential decay.
    """

    def __init__(
        self,
        fault_class:     int   = 1,
        max_risk_radius: int   = 20,   # voxels
        decay_rate:      float = 0.15,
    ) -> None:
        self.fault_class     = fault_class
        self.max_risk_radius = max_risk_radius
        self.decay_rate      = decay_rate

    def compute(
        self,
        seg_map:  np.ndarray,
        conf_map: np.ndarray,
        min_conf: float = 0.5,
    ) -> np.ndarray:
        """
        Returns fault proximity risk map [IL, XL, T], float32 in [0, 1].
        Score = 1 at fault voxel, decays exponentially with distance.
        """
        fault_mask = (seg_map == self.fault_class) & (conf_map >= min_conf)

        if not fault_mask.any():
            return np.zeros(seg_map.shape, dtype=np.float32)

        # Distance from each voxel to nearest fault voxel
        dist = distance_transform_edt(~fault_mask).astype(np.float32)

        # Exponential decay: risk = exp(-decay * dist)
        risk = np.exp(-self.decay_rate * dist)

        # Clip beyond max_risk_radius to zero
        risk[dist > self.max_risk_radius] = 0.0
        return risk.astype(np.float32)


class OverpressureDetector:
    """
    Detects potential overpressure zones using two proxy indicators:

    1. Amplitude shadow below bright reflector
       (gas obscures deeper signal → dim zone below bright)

    2. Low instantaneous frequency zone
       (overpressured sands show frequency attenuation)

    High score → elevated pore pressure → blowout / kick risk.
    """

    def __init__(
        self,
        shadow_depth_window: int   = 15,   # samples below bright spot
        bright_threshold:    float = 0.8,  # normalised envelope percentile
        freq_low_threshold:  float = 0.3,  # below this normalised freq = anomaly
    ) -> None:
        self.shadow_depth  = shadow_depth_window
        self.bright_thresh = bright_threshold
        self.freq_low      = freq_low_threshold

    def compute(
        self,
        cube:     np.ndarray,
        envelope: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Returns overpressure risk map [IL, XL, T], float32 in [0, 1].
        """
        from scipy.signal import hilbert

        if envelope is None:
            envelope = np.abs(hilbert(cube, axis=2)).astype(np.float32)

        # Normalise envelope
        env_norm = (envelope - envelope.min()) / (envelope.max() - envelope.min() + 1e-8)

        # Bright spot mask (top percentile)
        bright_mask = env_norm > self.bright_thresh

        # Shadow: dilation downward (time axis) from bright spot
        shadow_map = np.zeros_like(env_norm)
        for t in range(1, min(self.shadow_depth, cube.shape[2])):
            rolled = np.roll(bright_mask.astype(np.float32), t, axis=2)
            rolled[:, :, :t] = 0
            shadow_map += rolled * (1.0 - t / self.shadow_depth)

        shadow_map = np.clip(shadow_map, 0, 1)

        # Frequency attenuation proxy: large gradient → high freq, smooth → low
        freq_proxy = gaussian_filter(np.abs(np.diff(cube, axis=2, prepend=cube[:, :, :1])),
                                     sigma=(1, 1, 2))
        freq_norm  = (freq_proxy - freq_proxy.min()) / (freq_proxy.max() - freq_proxy.min() + 1e-8)
        low_freq   = (1.0 - freq_norm).astype(np.float32)
        low_freq[low_freq < self.freq_low] = 0.0

        # Combine: both indicators suggest overpressure
        risk = (0.6 * shadow_map + 0.4 * low_freq).astype(np.float32)
        return np.clip(risk, 0, 1)


class GasHydrateDetector:
    """
    Gas hydrate detection — high-amplitude, shallow, flat reflectors
    (Bottom Simulating Reflector, BSR).

    Hazard: drilling through hydrate can cause:
        • Seafloor instability
        • Wellbore collapse
        • Methane release

    Strategy:
        - High envelope in shallow zone (top 30% of time axis)
        - High lateral continuity (flat spot characteristic)
        - Polarity reversal proxy (envelope peak + phase anomaly)
    """

    def __init__(
        self,
        shallow_fraction: float = 0.35,   # top fraction of T axis
        amp_percentile:   float = 85.0,
        continuity_sigma: float = 3.0,
    ) -> None:
        self.shallow_frac = shallow_fraction
        self.amp_pct      = amp_percentile
        self.cont_sigma   = continuity_sigma

    def compute(self, cube: np.ndarray) -> np.ndarray:
        """Returns hydrate risk map [IL, XL, T], float32 in [0, 1]."""
        from scipy.signal import hilbert

        n_t       = cube.shape[2]
        shallow_t = int(n_t * self.shallow_frac)

        envelope  = np.abs(hilbert(cube, axis=2)).astype(np.float32)

        # Amplitude threshold in shallow zone
        shallow_env        = envelope.copy()
        shallow_env[:, :, shallow_t:] = 0
        amp_thresh         = np.percentile(envelope[:, :, :shallow_t], self.amp_pct)
        high_amp_shallow   = (shallow_env > amp_thresh).astype(np.float32)

        # Lateral continuity (high = hydrate BSR)
        continuity         = gaussian_filter(
            high_amp_shallow, sigma=(self.cont_sigma, self.cont_sigma, 0.5)
        )

        risk = np.clip(continuity, 0, 1).astype(np.float32)
        # Zero out deep zone — hydrates are shallow
        risk[:, :, shallow_t:] = 0.0
        return risk


class ShallowWaterFlowDetector:
    """
    Detects shallow water flow (SWF) risk — unconsolidated high-porosity
    sands that can flow into wellbore during drilling.

    Proxy indicators:
        - High amplitude shallow reflectors (high-porosity sand)
        - Chaotic seismic facies (unconsolidated)
        - Lack of coherent layering
    """

    def __init__(
        self,
        shallow_fraction:  float = 0.25,
        chaos_window:      int   = 5,
        amp_percentile:    float = 80.0,
    ) -> None:
        self.shallow_frac = shallow_fraction
        self.chaos_window = chaos_window
        self.amp_pct      = amp_percentile

    def compute(self, cube: np.ndarray) -> np.ndarray:
        """Returns SWF risk map [IL, XL, T], float32 in [0, 1]."""
        n_t       = cube.shape[2]
        shallow_t = int(n_t * self.shallow_frac)

        # Seismic chaos: high local variance = chaotic = unconsolidated
        from numpy.lib.stride_tricks import sliding_window_view
        w   = self.chaos_window
        pad = w // 2
        padded = np.pad(cube, ((0,0),(0,0),(pad,pad)), mode="edge")
        wins   = sliding_window_view(padded, window_shape=w, axis=2)
        chaos  = wins.std(axis=-1).astype(np.float32)
        chaos_norm = (chaos - chaos.min()) / (chaos.max() - chaos.min() + 1e-8)

        # High-amplitude shallow zone
        from scipy.signal import hilbert
        envelope    = np.abs(hilbert(cube, axis=2)).astype(np.float32)
        thresh      = np.percentile(envelope[:, :, :shallow_t], self.amp_pct)
        high_amp    = np.clip(envelope / (thresh + 1e-8), 0, 1)
        high_amp[:, :, shallow_t:] = 0.0

        risk = np.clip(0.5 * chaos_norm + 0.5 * high_amp, 0, 1).astype(np.float32)
        risk[:, :, shallow_t:] = 0.0
        return risk


class FaultSealAnalyzer:
    """
    Assesses fault seal integrity — whether a fault acts as a barrier
    or conduit for fluid migration.

    Leaking fault → fluid at surface → blowout risk
    Sealing fault → reservoir integrity maintained but overpressure possible

    Proxy: juxtaposition of high-amplitude zones across fault plane.
    High amplitude on both sides of fault → possible leak pathway.
    """

    def __init__(
        self,
        fault_class:    int   = 1,
        dilation_width: int   = 3,
        amp_percentile: float = 75.0,
    ) -> None:
        self.fault_class    = fault_class
        self.dilation_width = dilation_width
        self.amp_pct        = amp_percentile

    def compute(
        self,
        cube:     np.ndarray,
        seg_map:  np.ndarray,
        conf_map: np.ndarray,
        min_conf: float = 0.5,
    ) -> np.ndarray:
        """Returns fault seal risk map [IL, XL, T], float32 in [0, 1]."""
        from scipy.signal import hilbert

        fault_mask = (seg_map == self.fault_class) & (conf_map >= min_conf)
        if not fault_mask.any():
            return np.zeros(cube.shape, dtype=np.float32)

        envelope  = np.abs(hilbert(cube, axis=2)).astype(np.float32)
        amp_thresh = np.percentile(envelope, self.amp_pct)

        # Dilate fault plane to capture both sides
        dilated  = binary_dilation(fault_mask, iterations=self.dilation_width)
        adjacent = dilated & ~fault_mask

        # High amplitude adjacent to fault → potential leak
        high_amp_adj = (envelope > amp_thresh) & adjacent
        leak_risk    = gaussian_filter(
            high_amp_adj.astype(np.float32),
            sigma=(2, 2, 1),
        )
        return np.clip(leak_risk / (leak_risk.max() + 1e-8), 0, 1).astype(np.float32)


class StructuralDipAnalyzer:
    """
    Computes structural dip angle from seismic reflector geometry.
    High dip → borehole deviation risk + casing design complications.

    Method: 3D Sobel gradient magnitude as dip proxy.
    """

    @staticmethod
    def compute(cube: np.ndarray, smooth_sigma: float = 1.5) -> np.ndarray:
        """
        Returns dip map [IL, XL, T], float32.
        Values represent relative dip magnitude (0 = flat, 1 = steep).
        """
        smoothed = gaussian_filter(cube, sigma=smooth_sigma)

        gx = sobel(smoothed, axis=0)
        gy = sobel(smoothed, axis=1)
        gz = sobel(smoothed, axis=2)

        dip = np.sqrt(gx**2 + gy**2 + gz**2).astype(np.float32)
        dip_norm = (dip - dip.min()) / (dip.max() - dip.min() + 1e-8)
        return dip_norm


# ─────────────────────────────────────────────────────────────
#  COMPOSITE RISK SCORER
# ─────────────────────────────────────────────────────────────

class CompositeRiskScorer:
    """
    Combines individual risk maps into a single composite score
    using configurable weights per hazard category.

    Default weights reflect typical offshore drilling risk priorities.
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "fault_proximity": 0.25,
        "overpressure":    0.25,
        "hydrate":         0.20,
        "shallow_water":   0.15,
        "fault_seal":      0.10,
        "dip_hazard":      0.05,
    }

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        w = weights or self.DEFAULT_WEIGHTS
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-4, \
            f"Risk weights must sum to 1.0, got {total:.4f}"
        self.weights = w

    def compute(
        self,
        fault_proximity: np.ndarray,
        overpressure:    np.ndarray,
        hydrate:         np.ndarray,
        shallow_water:   np.ndarray,
        fault_seal:      np.ndarray,
        dip_hazard:      np.ndarray,
    ) -> np.ndarray:
        """Returns composite risk map [IL, XL, T], float32 in [0, 1]."""
        w  = self.weights
        composite = (
            w["fault_proximity"] * fault_proximity +
            w["overpressure"]    * overpressure    +
            w["hydrate"]         * hydrate         +
            w["shallow_water"]   * shallow_water   +
            w["fault_seal"]      * fault_seal      +
            w["dip_hazard"]      * dip_hazard
        )
        return np.clip(composite, 0, 1).astype(np.float32)


# ─────────────────────────────────────────────────────────────
#  HAZARD ZONE EXTRACTOR
# ─────────────────────────────────────────────────────────────

_HAZARD_META: Dict[str, Dict] = {
    "fault_proximity": {
        "description": "Wellbore instability risk near fault plane. "
                       "Risk of lost circulation, kicks, and blowout via fault conduit.",
        "mitigation":  "Increase casing weight, use mud weight at upper safe limit, "
                       "plan well trajectory to cross fault at low angle.",
    },
    "overpressure": {
        "description": "Abnormal pore pressure zone indicated by amplitude shadow "
                       "and frequency attenuation. Risk of well kick or blowout.",
        "mitigation":  "Increase mud weight in affected interval, set casing above zone, "
                       "use managed pressure drilling (MPD) technique.",
    },
    "hydrate": {
        "description": "Potential gas hydrate zone (BSR indicator). Drilling through "
                       "hydrate can cause rapid dissociation, wellbore collapse, and "
                       "methane release.",
        "mitigation":  "Case off hydrate interval before drilling below, "
                       "use chilled drilling fluid, monitor wellhead temperature.",
    },
    "shallow_water": {
        "description": "Shallow water flow risk from unconsolidated high-porosity sands. "
                       "Can cause sustained casing pressure and seafloor cratering.",
        "mitigation":  "Deploy conductor casing to isolate SWF zone, "
                       "monitor annular pressure continuously.",
    },
    "fault_seal": {
        "description": "Fault seal integrity concern — high amplitude juxtaposition "
                       "suggests potential fluid leak pathway along fault plane.",
        "mitigation":  "Detailed fault seal analysis using Allan diagram, "
                       "consider alternative well location away from fault.",
    },
    "dip_hazard": {
        "description": "High structural dip creating borehole deviation risk and "
                       "casing design complications.",
        "mitigation":  "Use rotary steerable system, increase weight on bit in "
                       "high-dip intervals, consider horizontal completion.",
    },
}


class HazardZoneExtractor:
    """
    Labels connected high-risk voxel clusters as discrete hazard zones
    and assigns metadata, risk level, and mitigation text.
    """

    def __init__(
        self,
        min_zone_voxels:    int   = 50,
        risk_threshold:     float = 0.35,
        inline_spacing_m:   float = 25.0,
        crossline_spacing_m: float = 25.0,
        sample_rate_ms:     float = 2.0,
    ) -> None:
        self.min_voxels         = min_zone_voxels
        self.risk_threshold     = risk_threshold
        self.inline_spacing     = inline_spacing_m
        self.crossline_spacing  = crossline_spacing_m
        self.sample_rate        = sample_rate_ms

    def extract(
        self,
        risk_maps: Dict[str, np.ndarray],
    ) -> List[HazardZone]:
        """
        For each hazard category, label connected high-risk clusters
        and return a list of HazardZone objects.
        """
        zones    = []
        hz_id    = 0

        for category, risk_map in risk_maps.items():
            high_risk_mask           = risk_map >= self.risk_threshold
            labeled, n_components    = nd_label(high_risk_mask)

            for lbl in range(1, n_components + 1):
                mask = labeled == lbl
                n    = int(mask.sum())
                if n < self.min_voxels:
                    continue

                coords  = np.argwhere(mask)
                ci, cx, ct = coords.mean(axis=0)
                score   = float(risk_map[mask].mean())
                level   = RiskLevel.from_score(score)
                meta    = _HAZARD_META.get(category, {
                    "description": f"{category} risk zone.",
                    "mitigation":  "Consult geomechanics team.",
                })

                zones.append(HazardZone(
                    hazard_id    = hz_id,
                    hazard_type  = category,
                    risk_level   = level,
                    risk_score   = score,
                    centroid     = (
                        ci * self.inline_spacing,
                        cx * self.crossline_spacing,
                        ct * self.sample_rate,
                    ),
                    voxel_count  = n,
                    description  = meta["description"],
                    mitigation   = meta["mitigation"],
                ))
                hz_id += 1

        return sorted(zones, key=lambda z: -z.risk_score)


# ─────────────────────────────────────────────────────────────
#  ZONE RISK PROFILER
# ─────────────────────────────────────────────────────────────

class ZoneRiskProfiler:
    """
    Assigns a risk profile to each reservoir zone by sampling the
    risk maps within the zone's spatial extent.
    """

    def __init__(self, scorer: CompositeRiskScorer) -> None:
        self.scorer = scorer

    def profile(
        self,
        zone_masks:      List[Tuple[int, np.ndarray]],   # [(zone_id, bool mask)]
        risk_maps:       Dict[str, np.ndarray],
        hazard_zones:    List[HazardZone],
    ) -> List[ZoneRiskProfile]:
        profiles = []

        for zone_id, mask in zone_masks:
            if not mask.any():
                continue

            # Mean risk score per category within this zone
            def _mean(key: str) -> float:
                m = risk_maps.get(key)
                if m is None:
                    return 0.0
                return float(m[mask].mean())

            fp  = _mean("fault_proximity")
            op  = _mean("overpressure")
            hyd = _mean("hydrate")
            swf = _mean("shallow_water")
            fs  = _mean("fault_seal")
            dip = _mean("dip_hazard")

            composite = float(self.scorer.compute(
                fault_proximity = np.array([fp]),
                overpressure    = np.array([op]),
                hydrate         = np.array([hyd]),
                shallow_water   = np.array([swf]),
                fault_seal      = np.array([fs]),
                dip_hazard      = np.array([dip]),
            ).mean())

            # Find hazard zones that overlap with this reservoir zone
            associated = [
                h for h in hazard_zones
                if self._overlaps(h, mask)
            ]

            profiles.append(ZoneRiskProfile(
                zone_id               = zone_id,
                composite_score       = composite,
                composite_level       = RiskLevel.from_score(composite),
                fault_proximity_score = fp,
                overpressure_score    = op,
                hydrate_score         = hyd,
                shallow_water_score   = swf,
                fault_seal_score      = fs,
                dip_hazard_score      = dip,
                associated_hazards    = associated,
            ))

        return sorted(profiles, key=lambda p: -p.composite_score)

    @staticmethod
    def _overlaps(hazard: HazardZone, zone_mask: np.ndarray) -> bool:
        """Quick check: hazard centroid (voxel) inside zone mask."""
        # centroid is in metres/ms — we just check if any HIGH risk
        # voxel is within the zone's bounding region (approximate)
        return hazard.risk_score > 0.5 and zone_mask.any()


# ─────────────────────────────────────────────────────────────
#  MAIN RISK ASSESSOR
# ─────────────────────────────────────────────────────────────

class RiskAssessor:
    """
    End-to-end drilling risk assessment from seismic cube + segmentation.

    Pipeline:
        1. Fault proximity risk map
        2. Overpressure indicator map
        3. Gas hydrate risk map
        4. Shallow water flow risk map
        5. Fault seal integrity map
        6. Structural dip hazard map
        7. Composite risk map (weighted combination)
        8. Extract discrete hazard zones
        9. Profile risk per reservoir zone (if provided)
        10. Return RiskReport

    Parameters
    ----------
    fault_class             : segmentation class index for faults
    min_confidence          : minimum model confidence to include
    min_hazard_voxels       : minimum voxel count for a hazard zone
    risk_threshold          : minimum score to classify as hazard zone
    risk_weights            : dict to override CompositeRiskScorer weights
    inline_spacing_m        : inline spacing in metres
    crossline_spacing_m     : crossline spacing in metres
    sample_rate_ms          : seismic sample interval in milliseconds
    """

    def __init__(
        self,
        fault_class:          int   = 1,
        min_confidence:       float = 0.5,
        min_hazard_voxels:    int   = 50,
        risk_threshold:       float = 0.35,
        risk_weights:         Optional[Dict[str, float]] = None,
        inline_spacing_m:     float = 25.0,
        crossline_spacing_m:  float = 25.0,
        sample_rate_ms:       float = 2.0,
    ) -> None:
        self.fault_class   = fault_class
        self.min_conf      = min_confidence

        self._fault_prox  = FaultProximityAnalyzer(fault_class=fault_class)
        self._overpressure = OverpressureDetector()
        self._hydrate      = GasHydrateDetector()
        self._swf          = ShallowWaterFlowDetector()
        self._seal         = FaultSealAnalyzer(fault_class=fault_class)
        self._dip          = StructuralDipAnalyzer()
        self._scorer       = CompositeRiskScorer(weights=risk_weights)
        self._extractor    = HazardZoneExtractor(
            min_zone_voxels     = min_hazard_voxels,
            risk_threshold      = risk_threshold,
            inline_spacing_m    = inline_spacing_m,
            crossline_spacing_m = crossline_spacing_m,
            sample_rate_ms      = sample_rate_ms,
        )
        self._profiler = ZoneRiskProfiler(self._scorer)

    # ── Public API ──────────────────────────────────────────────

    def assess(
        self,
        cube:                  np.ndarray,
        seg_map:               np.ndarray,
        conf_map:              np.ndarray,
        reservoir_zone_masks:  Optional[List[Tuple[int, np.ndarray]]] = None,
    ) -> RiskReport:
        """
        Run full risk assessment pipeline.

        Parameters
        ----------
        cube                 : seismic amplitude [IL, XL, T] float32
        seg_map              : segmentation      [IL, XL, T] uint8
        conf_map             : model confidence  [IL, XL, T] float32
        reservoir_zone_masks : optional list of (zone_id, bool mask) pairs
                               from ReservoirPredictor for per-zone profiling

        Returns
        -------
        RiskReport with all risk maps and hazard zone list
        """
        print("[RiskAssessor] Step 1/7 — Fault proximity risk …")
        fp_map = self._fault_prox.compute(seg_map, conf_map, self.min_conf)

        print("[RiskAssessor] Step 2/7 — Overpressure detection …")
        op_map = self._overpressure.compute(cube)

        print("[RiskAssessor] Step 3/7 — Gas hydrate risk …")
        hy_map = self._hydrate.compute(cube)

        print("[RiskAssessor] Step 4/7 — Shallow water flow risk …")
        sw_map = self._swf.compute(cube)

        print("[RiskAssessor] Step 5/7 — Fault seal analysis …")
        fs_map = self._seal.compute(cube, seg_map, conf_map, self.min_conf)

        print("[RiskAssessor] Step 6/7 — Structural dip hazard …")
        dp_map = self._dip.compute(cube)

        print("[RiskAssessor] Step 7/7 — Composite scoring + zone extraction …")
        comp_map = self._scorer.compute(fp_map, op_map, hy_map, sw_map, fs_map, dp_map)

        risk_maps = {
            "fault_proximity": fp_map,
            "overpressure":    op_map,
            "hydrate":         hy_map,
            "shallow_water":   sw_map,
            "fault_seal":      fs_map,
            "dip_hazard":      dp_map,
        }

        hazard_zones = self._extractor.extract(risk_maps)

        zone_profiles: List[ZoneRiskProfile] = []
        if reservoir_zone_masks:
            zone_profiles = self._profiler.profile(
                reservoir_zone_masks, risk_maps, hazard_zones
            )

        report = RiskReport(
            hazard_zones        = hazard_zones,
            zone_profiles       = zone_profiles,
            fault_proximity_map = fp_map,
            overpressure_map    = op_map,
            hydrate_risk_map    = hy_map,
            dip_map             = dp_map,
            composite_risk_map  = comp_map,
        )

        print(report.summary())
        return report

    def quick_screen(
        self,
        cube:     np.ndarray,
        seg_map:  np.ndarray,
        conf_map: np.ndarray,
    ) -> Dict[str, RiskLevel]:
        """
        Fast volume-level risk screening — returns one RiskLevel
        per category without full zone extraction.

        Useful for rapid go/no-go decision before full assessment.
        """
        fp  = self._fault_prox.compute(seg_map, conf_map).mean()
        op  = self._overpressure.compute(cube).mean()
        hy  = self._hydrate.compute(cube).mean()
        sw  = self._swf.compute(cube).mean()
        fs  = self._seal.compute(cube, seg_map, conf_map).mean()
        dp  = self._dip.compute(cube).mean()

        # Scale means to 0–1 risk scores (means are typically 0–0.3)
        scale = 4.0
        return {
            "fault_proximity": RiskLevel.from_score(min(fp * scale, 1.0)),
            "overpressure":    RiskLevel.from_score(min(op * scale, 1.0)),
            "hydrate":         RiskLevel.from_score(min(hy * scale, 1.0)),
            "shallow_water":   RiskLevel.from_score(min(sw * scale, 1.0)),
            "fault_seal":      RiskLevel.from_score(min(fs * scale, 1.0)),
            "dip_hazard":      RiskLevel.from_score(min(dp * scale, 1.0)),
        }

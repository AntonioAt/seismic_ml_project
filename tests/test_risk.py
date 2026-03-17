"""
tests/test_risk.py
==================
Unit tests for the RiskAssessor module.

Run with:
    pytest tests/test_risk.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seismic_ml.risk import (
    CompositeRiskScorer,
    FaultProximityAnalyzer,
    FaultSealAnalyzer,
    GasHydrateDetector,
    HazardZoneExtractor,
    OverpressureDetector,
    RiskAssessor,
    RiskLevel,
    RiskReport,
    ShallowWaterFlowDetector,
    StructuralDipAnalyzer,
    ZoneRiskProfiler,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

SHAPE = (64, 64, 64)


def _make_cube(seed: int = 42) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    cube = rng.standard_normal(SHAPE).astype(np.float32)
    cube[:, :, 10:14] += 5.0   # shallow bright spot (hydrate proxy)
    cube[:, :, 30:34] += 3.0   # horizon
    return cube


def _make_seg(fault: bool = True) -> np.ndarray:
    seg = np.zeros(SHAPE, dtype=np.uint8)
    seg[:, :, 30:34] = 0    # horizon
    if fault:
        for i in range(SHAPE[0]):
            seg[i, :, min(i, SHAPE[2] - 1)] = 1   # diagonal fault
    return seg


def _make_conf(high_zones: bool = True) -> np.ndarray:
    conf = np.full(SHAPE, 0.5, dtype=np.float32)
    if high_zones:
        conf[:, :, 30:34] = 0.90
    return conf


# ─────────────────────────────────────────────
#  RiskLevel
# ─────────────────────────────────────────────

class TestRiskLevel:

    @pytest.mark.parametrize("score,expected", [
        (0.10, RiskLevel.LOW),
        (0.30, RiskLevel.MEDIUM),
        (0.60, RiskLevel.HIGH),
        (0.80, RiskLevel.CRITICAL),
    ])
    def test_from_score(self, score, expected):
        assert RiskLevel.from_score(score) == expected

    def test_boundary_values(self):
        assert RiskLevel.from_score(0.0)  == RiskLevel.LOW
        assert RiskLevel.from_score(0.25) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(0.50) == RiskLevel.HIGH
        assert RiskLevel.from_score(0.75) == RiskLevel.CRITICAL
        assert RiskLevel.from_score(1.0)  == RiskLevel.CRITICAL


# ─────────────────────────────────────────────
#  FaultProximityAnalyzer
# ─────────────────────────────────────────────

class TestFaultProximityAnalyzer:

    def test_output_shape(self):
        seg  = _make_seg()
        conf = _make_conf()
        risk = FaultProximityAnalyzer().compute(seg, conf)
        assert risk.shape == SHAPE
        assert risk.dtype == np.float32

    def test_range(self):
        seg  = _make_seg()
        conf = _make_conf()
        risk = FaultProximityAnalyzer().compute(seg, conf)
        assert float(risk.min()) >= 0.0
        assert float(risk.max()) <= 1.0 + 1e-5

    def test_high_risk_at_fault(self):
        seg  = _make_seg()
        conf = _make_conf()
        risk = FaultProximityAnalyzer().compute(seg, conf)
        # Risk at fault voxels should be high
        fault_mask  = seg == 1
        risk_at_fault = float(risk[fault_mask].mean())
        assert risk_at_fault > 0.5, f"Risk at fault too low: {risk_at_fault:.3f}"

    def test_zero_map_when_no_fault(self):
        seg_no_fault = np.zeros(SHAPE, dtype=np.uint8)
        conf         = _make_conf()
        risk         = FaultProximityAnalyzer().compute(seg_no_fault, conf)
        assert float(risk.max()) == 0.0


# ─────────────────────────────────────────────
#  OverpressureDetector
# ─────────────────────────────────────────────

class TestOverpressureDetector:

    def test_output_shape_and_range(self):
        cube = _make_cube()
        risk = OverpressureDetector().compute(cube)
        assert risk.shape == SHAPE
        assert float(risk.min()) >= 0.0
        assert float(risk.max()) <= 1.0 + 1e-5

    def test_accepts_precomputed_envelope(self):
        from scipy.signal import hilbert
        cube = _make_cube()
        env  = np.abs(hilbert(cube, axis=2)).astype(np.float32)
        risk = OverpressureDetector().compute(cube, envelope=env)
        assert risk.shape == SHAPE


# ─────────────────────────────────────────────
#  GasHydrateDetector
# ─────────────────────────────────────────────

class TestGasHydrateDetector:

    def test_output_shape_and_range(self):
        cube = _make_cube()
        risk = GasHydrateDetector().compute(cube)
        assert risk.shape == SHAPE
        assert float(risk.min()) >= 0.0
        assert float(risk.max()) <= 1.0 + 1e-5

    def test_shallow_only(self):
        cube        = _make_cube()
        det         = GasHydrateDetector(shallow_fraction=0.3)
        risk        = det.compute(cube)
        shallow_t   = int(SHAPE[2] * 0.3)
        deep_risk   = float(risk[:, :, shallow_t:].max())
        assert deep_risk == 0.0, "Hydrate risk should be zero in deep zone"

    def test_high_risk_at_shallow_bright(self):
        cube = _make_cube()   # has bright spot at T=10-14 (shallow)
        risk = GasHydrateDetector(amp_percentile=70.0).compute(cube)
        shallow_mean = float(risk[:, :, :20].mean())
        assert shallow_mean > 0.0


# ─────────────────────────────────────────────
#  ShallowWaterFlowDetector
# ─────────────────────────────────────────────

class TestShallowWaterFlowDetector:

    def test_output_shape_and_range(self):
        cube = _make_cube()
        risk = ShallowWaterFlowDetector().compute(cube)
        assert risk.shape == SHAPE
        assert float(risk.min()) >= 0.0
        assert float(risk.max()) <= 1.0 + 1e-5

    def test_deep_zone_zero(self):
        cube      = _make_cube()
        det       = ShallowWaterFlowDetector(shallow_fraction=0.2)
        risk      = det.compute(cube)
        shallow_t = int(SHAPE[2] * 0.2)
        assert float(risk[:, :, shallow_t:].max()) == 0.0


# ─────────────────────────────────────────────
#  FaultSealAnalyzer
# ─────────────────────────────────────────────

class TestFaultSealAnalyzer:

    def test_output_shape_and_range(self):
        cube = _make_cube()
        seg  = _make_seg()
        conf = _make_conf()
        risk = FaultSealAnalyzer().compute(cube, seg, conf)
        assert risk.shape == SHAPE
        assert float(risk.min()) >= 0.0
        assert float(risk.max()) <= 1.0 + 1e-5

    def test_zero_when_no_fault(self):
        cube         = _make_cube()
        seg_no_fault = np.zeros(SHAPE, dtype=np.uint8)
        conf         = _make_conf()
        risk         = FaultSealAnalyzer().compute(cube, seg_no_fault, conf)
        assert float(risk.max()) == 0.0


# ─────────────────────────────────────────────
#  StructuralDipAnalyzer
# ─────────────────────────────────────────────

class TestStructuralDipAnalyzer:

    def test_output_shape_and_range(self):
        cube = _make_cube()
        dip  = StructuralDipAnalyzer.compute(cube)
        assert dip.shape == SHAPE
        assert float(dip.min()) >= 0.0
        assert float(dip.max()) <= 1.0 + 1e-5


# ─────────────────────────────────────────────
#  CompositeRiskScorer
# ─────────────────────────────────────────────

class TestCompositeRiskScorer:

    def test_output_shape(self):
        scorer = CompositeRiskScorer()
        maps   = {k: np.random.rand(*SHAPE).astype(np.float32) for k in
                  ["fault_proximity","overpressure","hydrate",
                   "shallow_water","fault_seal","dip_hazard"]}
        comp = scorer.compute(**maps)
        assert comp.shape == SHAPE

    def test_range(self):
        scorer = CompositeRiskScorer()
        maps   = {k: np.random.rand(*SHAPE).astype(np.float32) for k in
                  ["fault_proximity","overpressure","hydrate",
                   "shallow_water","fault_seal","dip_hazard"]}
        comp = scorer.compute(**maps)
        assert float(comp.min()) >= 0.0
        assert float(comp.max()) <= 1.0 + 1e-5

    def test_all_zeros_gives_zero(self):
        scorer = CompositeRiskScorer()
        zeros  = np.zeros(SHAPE, dtype=np.float32)
        comp   = scorer.compute(zeros, zeros, zeros, zeros, zeros, zeros)
        assert float(comp.max()) == 0.0

    def test_invalid_weights_raise(self):
        with pytest.raises(AssertionError):
            CompositeRiskScorer(weights={
                "fault_proximity": 0.5,
                "overpressure":    0.5,
                "hydrate":         0.5,
                "shallow_water":   0.1,
                "fault_seal":      0.1,
                "dip_hazard":      0.1,
            })


# ─────────────────────────────────────────────
#  HazardZoneExtractor
# ─────────────────────────────────────────────

class TestHazardZoneExtractor:

    def test_returns_list(self):
        ext  = HazardZoneExtractor(min_zone_voxels=10, risk_threshold=0.3)
        maps = {"fault_proximity": np.random.rand(*SHAPE).astype(np.float32)}
        zones = ext.extract(maps)
        assert isinstance(zones, list)

    def test_zones_sorted_by_risk(self):
        ext  = HazardZoneExtractor(min_zone_voxels=5, risk_threshold=0.2)
        rng  = np.random.default_rng(0)
        maps = {"fault_proximity": rng.random(SHAPE).astype(np.float32)}
        zones = ext.extract(maps)
        scores = [z.risk_score for z in zones]
        assert scores == sorted(scores, reverse=True)

    def test_zone_has_required_fields(self):
        ext  = HazardZoneExtractor(min_zone_voxels=5, risk_threshold=0.2)
        cube = np.ones(SHAPE, dtype=np.float32)
        zones = ext.extract({"fault_proximity": cube})
        if zones:
            z = zones[0]
            assert hasattr(z, "hazard_id")
            assert hasattr(z, "risk_level")
            assert hasattr(z, "mitigation")
            assert isinstance(z.to_dict(), dict)


# ─────────────────────────────────────────────
#  RiskAssessor (integration)
# ─────────────────────────────────────────────

class TestRiskAssessor:

    def test_assess_returns_report(self):
        cube = _make_cube()
        seg  = _make_seg()
        conf = _make_conf()
        assessor = RiskAssessor(min_hazard_voxels=10, risk_threshold=0.2)
        report   = assessor.assess(cube, seg, conf)
        assert isinstance(report, RiskReport)

    def test_all_maps_correct_shape(self):
        cube = _make_cube()
        seg  = _make_seg()
        conf = _make_conf()
        report = RiskAssessor(min_hazard_voxels=10).assess(cube, seg, conf)
        for attr in ("fault_proximity_map", "overpressure_map",
                     "hydrate_risk_map", "dip_map", "composite_risk_map"):
            m = getattr(report, attr)
            assert m.shape == SHAPE, f"{attr} shape mismatch: {m.shape}"

    def test_composite_map_range(self):
        cube   = _make_cube()
        seg    = _make_seg()
        conf   = _make_conf()
        report = RiskAssessor(min_hazard_voxels=10).assess(cube, seg, conf)
        assert float(report.composite_risk_map.min()) >= 0.0
        assert float(report.composite_risk_map.max()) <= 1.0 + 1e-5

    def test_n_critical_and_n_high_non_negative(self):
        cube   = _make_cube()
        seg    = _make_seg()
        conf   = _make_conf()
        report = RiskAssessor(min_hazard_voxels=10).assess(cube, seg, conf)
        assert report.n_critical >= 0
        assert report.n_high     >= 0

    def test_get_critical_zones_returns_list(self):
        cube   = _make_cube()
        seg    = _make_seg()
        conf   = _make_conf()
        report = RiskAssessor(min_hazard_voxels=10).assess(cube, seg, conf)
        crits  = report.get_critical_zones()
        assert isinstance(crits, list)

    def test_quick_screen_returns_risk_levels(self):
        cube     = _make_cube()
        seg      = _make_seg()
        conf     = _make_conf()
        assessor = RiskAssessor()
        screen   = assessor.quick_screen(cube, seg, conf)
        assert isinstance(screen, dict)
        assert len(screen) == 6
        for v in screen.values():
            assert isinstance(v, RiskLevel)

    def test_zone_profiles_populated_when_masks_given(self):
        cube = _make_cube()
        seg  = _make_seg()
        conf = _make_conf()

        # Create synthetic zone mask
        mask = np.zeros(SHAPE, dtype=bool)
        mask[:, :, 30:34] = True
        zone_masks = [(0, mask)]

        report = RiskAssessor(min_hazard_voxels=5).assess(
            cube, seg, conf,
            reservoir_zone_masks=zone_masks,
        )
        assert len(report.zone_profiles) > 0

    def test_summary_is_string(self):
        cube   = _make_cube()
        seg    = _make_seg()
        conf   = _make_conf()
        report = RiskAssessor(min_hazard_voxels=10).assess(cube, seg, conf)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "RISK" in summary.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

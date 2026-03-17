"""
tests/test_reservoir.py
=======================
Unit tests for the ReservoirPredictor module.

Run with:
    pytest tests/test_reservoir.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seismic_ml.reservoir import (
    AmplitudeAnomalyDetector,
    AVOProxyAnalyzer,
    ComplexTraceAttributes,
    DHIScorer,
    FlatSpotDetector,
    ReservoirPredictor,
    TrapGeometryClassifier,
)


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────

SHAPE = (64, 64, 64)


def _make_cube(shape=SHAPE, seed=42) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    cube = rng.standard_normal(shape).astype(np.float32)
    # Inject bright horizon at T=32
    cube[:, :, 30:34] += 4.0
    # Inject weak horizon at T=50
    cube[:, :, 49:51] += 1.5
    return cube


def _make_seg_map(shape=SHAPE) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.uint8)
    seg[:, :, 30:34] = 0   # horizon class
    seg[:, :, 49:51] = 0
    # Diagonal fault
    for i in range(shape[0]):
        seg[i, :, min(i, shape[2] - 1)] = 1
    return seg


def _make_conf_map(shape=SHAPE) -> np.ndarray:
    conf = np.full(shape, 0.5, dtype=np.float32)
    conf[:, :, 30:34] = 0.92
    conf[:, :, 49:51] = 0.70
    return conf


# ─────────────────────────────────────────────
#  ComplexTraceAttributes
# ─────────────────────────────────────────────

class TestComplexTraceAttributes:

    def test_envelope_shape(self):
        cube = _make_cube()
        env  = ComplexTraceAttributes.compute_envelope(cube)
        assert env.shape == cube.shape
        assert env.dtype == np.float32

    def test_envelope_non_negative(self):
        cube = _make_cube()
        env  = ComplexTraceAttributes.compute_envelope(cube)
        assert (env >= 0).all(), "Envelope must be non-negative"

    def test_instantaneous_phase_range(self):
        cube  = _make_cube()
        phase = ComplexTraceAttributes.compute_instantaneous_phase(cube)
        assert phase.shape == cube.shape
        assert float(phase.min()) >= -np.pi - 1e-3
        assert float(phase.max()) <=  np.pi + 1e-3

    def test_instantaneous_frequency_shape(self):
        cube = _make_cube()
        freq = ComplexTraceAttributes.compute_instantaneous_frequency(cube)
        assert freq.shape == cube.shape, "Frequency shape must match cube"

    def test_rai_shape(self):
        cube = _make_cube()
        rai  = ComplexTraceAttributes.compute_relative_acoustic_impedance(cube)
        assert rai.shape == cube.shape


# ─────────────────────────────────────────────
#  AmplitudeAnomalyDetector
# ─────────────────────────────────────────────

class TestAmplitudeAnomalyDetector:

    def test_anomaly_map_shape(self):
        cube     = _make_cube()
        env      = ComplexTraceAttributes.compute_envelope(cube)
        detector = AmplitudeAnomalyDetector()
        amap     = detector.compute_anomaly_map(env)
        assert amap.shape == cube.shape
        assert amap.dtype == np.float32

    def test_bright_spot_detected_on_injected_horizon(self):
        cube     = _make_cube()
        env      = ComplexTraceAttributes.compute_envelope(cube)
        detector = AmplitudeAnomalyDetector(bright_threshold_sigma=1.5)
        amap     = detector.compute_anomaly_map(env)
        bright   = detector.bright_spot_mask(amap)
        # The injected horizon at T=30-33 should have bright pixels
        bright_at_horizon = bright[:, :, 30:34].sum()
        assert bright_at_horizon > 0, "Bright spot not detected at injected horizon"

    def test_masks_are_boolean(self):
        cube     = _make_cube()
        env      = ComplexTraceAttributes.compute_envelope(cube)
        detector = AmplitudeAnomalyDetector()
        amap     = detector.compute_anomaly_map(env)
        assert detector.bright_spot_mask(amap).dtype == bool
        assert detector.dim_spot_mask(amap).dtype    == bool


# ─────────────────────────────────────────────
#  AVOProxyAnalyzer
# ─────────────────────────────────────────────

class TestAVOProxyAnalyzer:

    def test_avo_proxy_shape(self):
        cube  = _make_cube()
        proxy = AVOProxyAnalyzer.compute_avo_proxy(cube)
        assert proxy.shape == cube.shape
        assert proxy.dtype == np.float32


# ─────────────────────────────────────────────
#  FlatSpotDetector
# ─────────────────────────────────────────────

class TestFlatSpotDetector:

    def test_flat_spot_map_range(self):
        cube     = _make_cube()
        env      = ComplexTraceAttributes.compute_envelope(cube)
        flat_map = FlatSpotDetector.compute_flat_spot_map(env)
        assert flat_map.shape == cube.shape
        assert float(flat_map.min()) >= 0.0
        assert float(flat_map.max()) <= 1.0 + 1e-5

    def test_flat_spot_higher_at_horizon(self):
        cube     = _make_cube()
        env      = ComplexTraceAttributes.compute_envelope(cube)
        flat_map = FlatSpotDetector.compute_flat_spot_map(env)
        score_at_horizon = float(flat_map[:, :, 30:34].mean())
        score_overall    = float(flat_map.mean())
        # Injected flat horizon should score above average
        assert score_at_horizon >= score_overall * 0.8


# ─────────────────────────────────────────────
#  TrapGeometryClassifier
# ─────────────────────────────────────────────

class TestTrapGeometryClassifier:

    def test_fault_bounded_when_fault_adjacent(self):
        seg      = _make_seg_map()
        zone_mask = np.zeros(SHAPE, dtype=bool)
        zone_mask[:, :, 30:34] = True   # horizon zone adjacent to fault
        trap = TrapGeometryClassifier.classify(zone_mask, seg, fault_class=1)
        assert trap == "fault-bounded"

    def test_stratigraphic_when_no_fault(self):
        seg       = np.zeros(SHAPE, dtype=np.uint8)   # no faults
        zone_mask = np.zeros(SHAPE, dtype=bool)
        zone_mask[30:35, 30:35, 55:60] = True          # deep, flat zone
        trap = TrapGeometryClassifier.classify(zone_mask, seg, fault_class=1)
        assert trap in ("stratigraphic", "anticline")

    def test_returns_string(self):
        seg       = _make_seg_map()
        zone_mask = np.zeros(SHAPE, dtype=bool)
        zone_mask[20:30, 20:30, 30:34] = True
        trap = TrapGeometryClassifier.classify(zone_mask, seg)
        assert isinstance(trap, str)
        assert trap in ("anticline", "fault-bounded", "stratigraphic")


# ─────────────────────────────────────────────
#  DHIScorer
# ─────────────────────────────────────────────

class TestDHIScorer:

    def test_score_range(self):
        scorer = DHIScorer()
        for amp in [-3.0, 0.0, 3.0]:
            for avo in [-1.0, 0.0, 0.5]:
                for flat in [0.0, 0.5, 1.0]:
                    for trap in ["anticline", "fault-bounded", "stratigraphic"]:
                        score = scorer.score(amp, avo, flat, trap)
                        assert 0.0 <= score <= 1.0, \
                            f"Score {score} out of [0,1] for {amp},{avo},{flat},{trap}"

    def test_high_score_for_ideal_indicators(self):
        scorer = DHIScorer()
        score = scorer.score(
            amplitude_anomaly=3.0,    # strong bright spot
            avo_gradient=-1.0,        # Class III AVO
            flat_spot_score=1.0,      # clear fluid contact
            trap_type="anticline",    # best trap
        )
        assert score >= 0.75, f"Ideal indicators should score >= 0.75, got {score:.3f}"

    def test_low_score_for_poor_indicators(self):
        scorer = DHIScorer()
        score = scorer.score(
            amplitude_anomaly=-3.0,   # dim spot (tight sand)
            avo_gradient=0.5,         # positive AVO (brine)
            flat_spot_score=0.0,      # no flat spot
            trap_type="stratigraphic",
        )
        assert score <= 0.50, f"Poor indicators should score <= 0.50, got {score:.3f}"

    def test_weights_must_sum_to_one(self):
        with pytest.raises(AssertionError):
            DHIScorer(
                weight_amplitude=0.5,
                weight_avo=0.5,
                weight_flat_spot=0.5,  # sum > 1
                weight_trap=0.5,
            )


# ─────────────────────────────────────────────
#  ReservoirPredictor  (integration)
# ─────────────────────────────────────────────

class TestReservoirPredictor:

    def test_predict_returns_prediction_object(self):
        from seismic_ml.reservoir import ReservoirPrediction
        cube     = _make_cube()
        seg      = _make_seg_map()
        conf     = _make_conf_map()
        pred     = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.1).predict(
            cube, seg, conf
        )
        assert isinstance(pred, ReservoirPrediction)

    def test_attribute_maps_correct_shape(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        pred = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.1).predict(
            cube, seg, conf
        )
        for attr in ("amplitude_map", "avo_proxy_map", "flat_spot_map", "dhi_map"):
            arr = getattr(pred, attr)
            assert arr.shape == SHAPE, f"{attr} has wrong shape: {arr.shape}"

    def test_zones_are_ranked_by_dhi(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        pred = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.0).predict(
            cube, seg, conf
        )
        scores = [z.dhi_score for z in pred.zones]
        assert scores == sorted(scores, reverse=True), "Zones not ranked by DHI"

    def test_get_top_zones_returns_list_of_dicts(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        predictor = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.0)
        pred  = predictor.predict(cube, seg, conf)
        top   = predictor.get_top_zones(pred, n=3)
        assert isinstance(top, list)
        for item in top:
            assert isinstance(item, dict)
            assert "dhi_score" in item

    def test_drilling_candidates_filter(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        predictor  = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.0)
        pred       = predictor.predict(cube, seg, conf)
        candidates = predictor.get_drilling_candidates(pred, min_dhi=0.5)
        for c in candidates:
            assert c["dhi_score"] >= 0.5

    def test_n_zones_matches_zones_list(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        pred = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.0).predict(
            cube, seg, conf
        )
        assert pred.n_zones == len(pred.zones)

    def test_best_zone_has_highest_dhi(self):
        cube = _make_cube()
        seg  = _make_seg_map()
        conf = _make_conf_map()
        pred = ReservoirPredictor(min_zone_voxels=10, min_dhi_score=0.0).predict(
            cube, seg, conf
        )
        if pred.best_zone and pred.zones:
            assert pred.best_zone.dhi_score == max(z.dhi_score for z in pred.zones)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

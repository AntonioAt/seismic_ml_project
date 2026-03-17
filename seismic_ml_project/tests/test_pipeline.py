"""
tests/test_pipeline.py
======================
Unit tests for SeismicPipeline and PipelineConfig.

Run with:
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seismic_ml.pipeline import (
    PipelineConfig,
    PipelineReport,
    SeismicPipeline,
    StageResult,
)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

SHAPE = (64, 64, 64)


def _make_cube(seed: int = 0) -> np.ndarray:
    rng  = np.random.default_rng(seed)
    cube = rng.standard_normal(SHAPE).astype(np.float32)
    cube[:, :, 20:24] += 3.0
    cube[:, :, 48:52] += 3.0
    for i in range(SHAPE[0]):
        cube[i, :, min(i, SHAPE[2] - 1)] -= 4.0
    return cube


def _make_labels() -> np.ndarray:
    labels = np.zeros(SHAPE, dtype=np.int64)
    labels[:, :, 20:24] = 0   # horizon
    labels[:, :, 48:52] = 0
    for i in range(SHAPE[0]):
        labels[i, :, min(i, SHAPE[2] - 1)] = 1   # fault
    return labels


def _small_cfg() -> PipelineConfig:
    """Minimal config for fast smoke-test."""
    return PipelineConfig(
        architecture        = "unet3d",
        n_classes           = 3,
        base_features       = 8,
        depth               = 2,
        patch_size          = (16, 16, 16),
        n_patches           = 16,
        batch_size          = 2,
        n_epochs            = 1,
        num_workers         = 0,
        inference_overlap   = 0.25,
        inference_batch_size = 2,
        min_confidence      = 0.3,
        min_reservoir_voxels = 10,
        min_dhi_score       = 0.0,
        min_hazard_voxels   = 5,
        risk_threshold      = 0.2,
        save_arrays         = False,
        output_dir          = "/tmp/seismic_test_output",
        checkpoint_dir      = "/tmp/seismic_test_ckpt",
        viz_output_dir      = "/tmp/seismic_viz",
        viz_interactive_dir = "/tmp/seismic_viz_interactive",
    )


# ─────────────────────────────────────────────
#  PipelineConfig
# ─────────────────────────────────────────────

class TestPipelineConfig:

    def test_default_instantiation(self):
        cfg = PipelineConfig()
        assert cfg.n_classes == 3
        assert cfg.architecture == "unet3d"

    def test_custom_values(self):
        cfg = PipelineConfig(n_epochs=100, learning_rate=5e-4)
        assert cfg.n_epochs == 100
        assert cfg.learning_rate == 5e-4

    def test_from_dict(self):
        pipeline = SeismicPipeline.from_dict({
            "n_classes":    3,
            "architecture": "unet3d",
            "n_epochs":     10,
            "unknown_key":  "ignored",
        })
        assert isinstance(pipeline, SeismicPipeline)
        assert pipeline.cfg.n_epochs == 10

    def test_from_yaml(self, tmp_path):
        yaml_content = """
model:
  architecture: unet3d
  n_classes: 3
  base_features: 16
training:
  n_epochs: 5
  batch_size: 2
"""
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        try:
            pipeline = SeismicPipeline.from_yaml(str(yaml_file))
            assert pipeline.cfg.n_classes == 3
            assert pipeline.cfg.n_epochs == 5
        except ImportError:
            pytest.skip("pyyaml not installed")


# ─────────────────────────────────────────────
#  StageResult
# ─────────────────────────────────────────────

class TestStageResult:

    def test_to_dict(self):
        sr = StageResult("Test Stage", "ok", 1.23, {"key": "val"})
        d  = sr.to_dict()
        assert d["stage"]     == "Test Stage"
        assert d["status"]    == "ok"
        assert d["elapsed_s"] == 1.23
        assert d["details"]["key"] == "val"


# ─────────────────────────────────────────────
#  SeismicPipeline — individual stages
# ─────────────────────────────────────────────

class TestSeismicPipelineStages:

    @pytest.fixture
    def pipeline(self):
        return SeismicPipeline(_small_cfg())

    def test_initialization(self, pipeline):
        assert pipeline.cfg.architecture == "unet3d"
        assert pipeline._cube is None
        assert pipeline._model is None

    def test_run_preprocessing_with_direct_cube(self, pipeline):
        cube = _make_cube()
        out  = pipeline.run_preprocessing(cube)
        assert out.shape == SHAPE
        assert out.dtype == np.float32

    def test_run_build_model_unet(self, pipeline):
        model = pipeline.run_build_model()
        assert model is not None
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n > 0

    def test_run_build_model_transformer(self):
        cfg = _small_cfg()
        cfg.architecture       = "transformer"
        cfg.embed_dim          = 16
        cfg.transformer_depths = (1, 1, 1, 1)
        cfg.transformer_heads  = (1, 2, 4, 8)
        cfg.window_size        = (4, 4, 4)
        cfg.patch_size_model   = (4, 4, 4)
        pipeline = SeismicPipeline(cfg)
        pipeline._cube = _make_cube()
        model = pipeline.run_build_model()
        assert model is not None

    def test_run_build_dataset_returns_loader(self, pipeline):
        cube   = _make_cube()
        labels = _make_labels()
        loader = pipeline.run_build_dataset(cube, labels)
        batch  = next(iter(loader))
        assert len(batch) == 2       # (patches, labels)
        patches, lbls = batch
        assert patches.shape[1] == 1   # single channel

    def test_run_inference_output_shapes(self, pipeline):
        cube = _make_cube()
        pipeline._cube = cube
        pipeline.run_build_model()
        seg, conf = pipeline.run_inference(cube)
        assert seg.shape  == SHAPE
        assert conf.shape == SHAPE
        assert seg.dtype  == np.uint8
        assert conf.dtype == np.float32

    def test_run_reservoir_prediction(self, pipeline):
        cube   = _make_cube()
        seg    = np.zeros(SHAPE, dtype=np.uint8)
        seg[:, :, 20:24] = 0
        conf   = np.full(SHAPE, 0.8, dtype=np.float32)
        pred   = pipeline.run_reservoir_prediction(cube, seg, conf)
        assert hasattr(pred, "n_zones")
        assert hasattr(pred, "zones")

    def test_run_risk_assessment(self, pipeline):
        cube = _make_cube()
        seg  = np.zeros(SHAPE, dtype=np.uint8)
        for i in range(SHAPE[0]):
            seg[i, :, min(i, SHAPE[2] - 1)] = 1
        conf = np.full(SHAPE, 0.6, dtype=np.float32)
        report = pipeline.run_risk_assessment(cube, seg, conf)
        assert hasattr(report, "hazard_zones")
        assert hasattr(report, "composite_risk_map")
        assert report.composite_risk_map.shape == SHAPE


# ─────────────────────────────────────────────
#  SeismicPipeline — full pipeline
# ─────────────────────────────────────────────

class TestSeismicPipelineFullRun:
    """
    Integration test: run_full_pipeline() with synthetic data.
    Uses a minimal config to keep runtime short.
    """

    @pytest.fixture
    def pipeline_and_data(self, tmp_path):
        cfg = _small_cfg()
        cfg.output_dir          = str(tmp_path / "output")
        cfg.checkpoint_dir      = str(tmp_path / "checkpoints")
        cfg.viz_output_dir      = str(tmp_path / "viz")
        cfg.viz_interactive_dir = str(tmp_path / "viz_interactive")

        # Save synthetic cube as .npy so pipeline can load it
        cube = _make_cube()
        data_path = str(tmp_path / "synthetic.npy")
        np.save(data_path, cube)

        return SeismicPipeline(cfg), data_path, cube

    def test_full_pipeline_returns_report(self, pipeline_and_data, tmp_path):
        pipeline, data_path, _ = pipeline_and_data
        labels = _make_labels()

        report = pipeline.run_full_pipeline(
            data_path = data_path,
            labels    = labels,
        )
        assert isinstance(report, PipelineReport)

    def test_all_stages_recorded(self, pipeline_and_data):
        pipeline, data_path, _ = pipeline_and_data
        labels = _make_labels()
        report = pipeline.run_full_pipeline(data_path, labels=labels)

        stage_names = [sr.name for sr in report.stage_results]
        required = [
            "Data Ingestion",
            "Preprocessing",
            "Dataset Build",
            "Model Build",
            "Training",
            "Inference",
            "Reservoir Prediction",
            "Risk Assessment",
        ]
        for name in required:
            assert name in stage_names, f"Stage '{name}' missing from report"

    def test_cube_shape_in_report(self, pipeline_and_data):
        pipeline, data_path, _ = pipeline_and_data
        report = pipeline.run_full_pipeline(data_path)
        assert report.cube_shape == SHAPE

    def test_json_report_saved(self, pipeline_and_data, tmp_path):
        pipeline, data_path, _ = pipeline_and_data
        report = pipeline.run_full_pipeline(data_path)
        json_path = Path(pipeline.cfg.output_dir) / "pipeline_report.json"
        assert json_path.exists(), "JSON report not saved"

        with open(json_path) as f:
            data = json.load(f)
        assert "stages"           in data
        assert "cube_shape"       in data
        assert "n_reservoir_zones" in data

    def test_report_summary_is_string(self, pipeline_and_data):
        pipeline, data_path, _ = pipeline_and_data
        report  = pipeline.run_full_pipeline(data_path)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "PIPELINE" in summary.upper()

    def test_skip_training_with_no_checkpoint(self, pipeline_and_data):
        """skip_training=True with no checkpoint should still train."""
        pipeline, data_path, _ = pipeline_and_data
        labels = _make_labels()
        report = pipeline.run_full_pipeline(
            data_path,
            labels        = labels,
            skip_training = True,
            checkpoint    = "/nonexistent/path.pt",
        )
        # Should complete without error — training runs as fallback
        assert report is not None

    def test_stage_results_have_correct_status(self, pipeline_and_data):
        pipeline, data_path, _ = pipeline_and_data
        report = pipeline.run_full_pipeline(data_path)
        for sr in report.stage_results:
            assert sr.status in ("ok", "skipped", "failed")

    def test_total_elapsed_positive(self, pipeline_and_data):
        pipeline, data_path, _ = pipeline_and_data
        report = pipeline.run_full_pipeline(data_path)
        assert report.total_elapsed_s > 0.0

    def test_transformer_architecture_runs(self, tmp_path):
        """Ensure transformer architecture works through full pipeline."""
        cfg = _small_cfg()
        cfg.architecture       = "transformer"
        cfg.embed_dim          = 16
        cfg.transformer_depths = (1, 1, 1, 1)
        cfg.transformer_heads  = (1, 2, 4, 8)
        cfg.window_size        = (4, 4, 4)
        cfg.patch_size_model   = (4, 4, 4)
        cfg.output_dir         = str(tmp_path / "output_transformer")
        cfg.checkpoint_dir     = str(tmp_path / "ckpt_transformer")
        cfg.viz_output_dir     = str(tmp_path / "viz_t")
        cfg.viz_interactive_dir = str(tmp_path / "viz_t_int")

        cube      = _make_cube()
        data_path = str(tmp_path / "synthetic_t.npy")
        np.save(data_path, cube)

        pipeline = SeismicPipeline(cfg)
        report   = pipeline.run_full_pipeline(data_path)
        assert report is not None
        assert report.cube_shape == SHAPE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
pipeline.py
===========
SeismicPipeline — Priority #4 Module

Single end-to-end orchestrator that connects all modules into one
coherent, configurable workflow:

    Data Ingestion  →  Preprocessing  →  Dataset  →  Model  →  Training
    →  Inference  →  ReservoirPredictor  →  RiskAssessor  →  Visualization
    →  Final Report

Design principles:
    • One entry point  : pipeline.run_full_pipeline(path)
    • Config-driven    : all parameters from YAML / dict
    • Resumable        : checkpoint detection skips completed stages
    • Observable       : stage-level timing + progress logging
    • Testable         : each stage can be called independently
    • Serialisable     : JSON report saved to output directory

Usage:
    from seismic_ml.pipeline import SeismicPipeline, PipelineConfig

    # From YAML config
    pipeline = SeismicPipeline.from_yaml("configs/default_config.yaml")
    report   = pipeline.run_full_pipeline("data/survey.segy")

    # From dict
    cfg      = PipelineConfig(n_epochs=50, n_classes=3)
    pipeline = SeismicPipeline(cfg)
    report   = pipeline.run_full_pipeline("data/survey.npy")

    # Run individual stages independently
    cube     = pipeline.run_ingestion("data/survey.segy")
    cube     = pipeline.run_preprocessing(cube)
    model    = pipeline.run_build_model()
    seg, con = pipeline.run_inference(cube)
    res_pred = pipeline.run_reservoir_prediction(cube, seg, con)
    risk_rep = pipeline.run_risk_assessment(cube, seg, con)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────
#  PIPELINE CONFIG
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Centralised configuration for the entire pipeline.
    All fields map directly to YAML keys in configs/default_config.yaml.
    """

    # ── Data ─────────────────────────────────────────────────
    inline_spacing_m:      float             = 25.0
    crossline_spacing_m:   float             = 25.0
    sample_rate_ms:        float             = 2.0
    segy_inline_byte:      int               = 189
    segy_crossline_byte:   int               = 193
    chunk_size:            int               = 50

    # ── Preprocessing ─────────────────────────────────────────
    bandpass_low_hz:       float             = 5.0
    bandpass_high_hz:      float             = 120.0
    smooth_sigma:          float             = 0.8
    noise_attenuation:     bool              = True
    amplitude_scaling:     bool              = False

    # ── Model ─────────────────────────────────────────────────
    architecture:          str               = "unet3d"
    in_channels:           int               = 1
    n_classes:             int               = 3
    base_features:         int               = 32
    depth:                 int               = 4
    embed_dim:             int               = 48
    patch_size_model:      Tuple[int,int,int] = (4, 4, 4)
    window_size:           Tuple[int,int,int] = (4, 4, 4)
    transformer_depths:    Tuple[int,...]     = (2, 2, 4, 2)
    transformer_heads:     Tuple[int,...]     = (3, 6, 12, 24)

    # ── Training ──────────────────────────────────────────────
    n_epochs:              int               = 50
    batch_size:            int               = 4
    learning_rate:         float             = 1e-3
    grad_clip:             float             = 1.0
    num_workers:           int               = 4
    n_patches:             int               = 2000
    patch_size:            Tuple[int,int,int] = (64, 64, 64)
    use_amp:               bool              = True
    checkpoint_dir:        str               = "./checkpoints"

    # ── Inference ─────────────────────────────────────────────
    inference_overlap:     float             = 0.5
    inference_batch_size:  int               = 2
    min_confidence:        float             = 0.6

    # ── Reservoir ─────────────────────────────────────────────
    horizon_class:         int               = 0
    fault_class:           int               = 1
    min_reservoir_voxels:  int               = 200
    min_dhi_score:         float             = 0.30

    # ── Risk ──────────────────────────────────────────────────
    min_hazard_voxels:     int               = 50
    risk_threshold:        float             = 0.35

    # ── Visualization ─────────────────────────────────────────
    viz_output_dir:        str               = "./seismic_viz"
    viz_interactive_dir:   str               = "./seismic_viz_interactive"
    viz_dpi:               int               = 150
    viz_fmt:               str               = "png"
    viz_min_confidence:    float             = 0.4

    # ── Output ────────────────────────────────────────────────
    output_dir:            str               = "./pipeline_output"
    save_arrays:           bool              = False


# ─────────────────────────────────────────────────────────────
#  STAGE RESULT
# ─────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Timing and status for one pipeline stage."""
    name:      str
    status:    str          # "ok" | "skipped" | "failed"
    elapsed_s: float
    details:   Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "stage":     self.name,
            "status":    self.status,
            "elapsed_s": round(self.elapsed_s, 2),
            "details":   self.details,
        }


# ─────────────────────────────────────────────────────────────
#  PIPELINE REPORT
# ─────────────────────────────────────────────────────────────

@dataclass
class PipelineReport:
    """Full output from SeismicPipeline.run_full_pipeline()."""

    input_path:            str
    config:                PipelineConfig
    stage_results:         List[StageResult]
    cube_shape:            Optional[Tuple[int, int, int]]
    n_reservoir_zones:     int
    n_hazard_zones:        int
    n_critical:            int
    top_reservoir_zones:   List[Dict]
    critical_hazards:      List[Dict]
    output_files:          Dict[str, str]
    total_elapsed_s:       float

    def summary(self) -> str:
        lines = [
            "=" * 65,
            "  SEISMIC ML PIPELINE  —  FINAL REPORT",
            "=" * 65,
            f"  Input           : {self.input_path}",
            f"  Cube shape      : {self.cube_shape}",
            f"  Total time      : {self.total_elapsed_s:.1f}s",
            "",
            "  STAGE RESULTS:",
        ]
        for sr in self.stage_results:
            icon = "✓" if sr.status == "ok" else ("↷" if sr.status == "skipped" else "✗")
            lines.append(
                f"    {icon} {sr.name:32s} {sr.elapsed_s:6.1f}s  [{sr.status}]"
            )
        lines += [
            "",
            "  GEOLOGICAL RESULTS:",
            f"    Reservoir zones  : {self.n_reservoir_zones}",
            f"    Hazard zones     : {self.n_hazard_zones}",
            f"    Critical hazards : {self.n_critical}",
        ]
        if self.top_reservoir_zones:
            lines.append("\n  TOP RESERVOIR ZONES:")
            for z in self.top_reservoir_zones[:3]:
                lines.append(
                    f"    Zone {z['zone_id']:02d}  "
                    f"DHI={z['dhi_score']:.2f}  "
                    f"trap={z['trap_type']:15s}  "
                    f"conf={z['confidence']:.2f}"
                )
        if self.critical_hazards:
            lines.append("\n  CRITICAL / HIGH HAZARDS:")
            for h in self.critical_hazards[:3]:
                lines.append(
                    f"    [{h['risk_level']:8s}]  {h['hazard_type']:25s}  "
                    f"score={h['risk_score']:.2f}"
                )
        if self.output_files:
            lines.append("\n  OUTPUT FILES:")
            for name, path in self.output_files.items():
                lines.append(f"    {name:35s} → {path}")
        lines.append("=" * 65)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "input_path":          self.input_path,
            "cube_shape":          list(self.cube_shape) if self.cube_shape else None,
            "total_elapsed_s":     round(self.total_elapsed_s, 2),
            "stages":              [sr.to_dict() for sr in self.stage_results],
            "n_reservoir_zones":   self.n_reservoir_zones,
            "n_hazard_zones":      self.n_hazard_zones,
            "n_critical":          self.n_critical,
            "top_reservoir_zones": self.top_reservoir_zones,
            "critical_hazards":    self.critical_hazards,
            "output_files":        self.output_files,
        }


# ─────────────────────────────────────────────────────────────
#  STAGE TIMER CONTEXT MANAGER
# ─────────────────────────────────────────────────────────────

class _StageTimer:
    def __init__(self, name: str) -> None:
        self.name      = name
        self._t0       = 0.0
        self.elapsed_s = 0.0

    def __enter__(self) -> "_StageTimer":
        print(f"\n{'─' * 60}")
        print(f"  ▶  STAGE : {self.name}")
        print(f"{'─' * 60}")
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, *_) -> bool:
        self.elapsed_s = time.perf_counter() - self._t0
        status = "FAILED" if exc_type else "DONE"
        print(f"  ◀  {self.name} [{status}] in {self.elapsed_s:.2f}s")
        return False


# ─────────────────────────────────────────────────────────────
#  SEISMIC PIPELINE
# ─────────────────────────────────────────────────────────────

class SeismicPipeline:
    """
    End-to-end seismic ML pipeline orchestrator.

    11 stages, each independently callable or run together via
    run_full_pipeline().

    Stage 1  — Data Ingestion
    Stage 2  — Preprocessing
    Stage 3  — Dataset & DataLoader
    Stage 4  — Model Construction
    Stage 5  — Performance Optimization
    Stage 6  — Training
    Stage 7  — Inference
    Stage 8  — Reservoir Prediction   [Priority #1]
    Stage 9  — Risk Assessment        [Priority #3]
    Stage 10 — Visualization
    Stage 11 — Report Generation
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.cfg    = config or PipelineConfig()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Internal state
        self._cube:           Optional[np.ndarray]           = None
        self._seg_map:        Optional[np.ndarray]           = None
        self._conf_map:       Optional[np.ndarray]           = None
        self._model:          Optional[torch.nn.Module]      = None
        self._history:        Dict[str, List[float]]         = {"train_loss": [], "val_loss": []}
        self._reservoir_pred: Any                            = None
        self._risk_report:    Any                            = None
        self._predictor:      Any                            = None
        self._stage_results:  List[StageResult]              = []

        print(f"[SeismicPipeline] Ready")
        print(f"  device       : {self.device}")
        print(f"  architecture : {self.cfg.architecture}")
        print(f"  n_classes    : {self.cfg.n_classes}")

    # ── Factory constructors ─────────────────────────────────

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "SeismicPipeline":
        """Load config from YAML and return a SeismicPipeline."""
        try:
            import yaml
        except ImportError:
            raise ImportError("pip install pyyaml")

        with open(yaml_path) as f:
            raw = yaml.safe_load(f)

        flat: Dict[str, Any] = {}
        for section in raw.values():
            if isinstance(section, dict):
                flat.update(section)

        known  = {fld.name for fld in PipelineConfig.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in flat.items() if k in known}
        return cls(PipelineConfig(**kwargs))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "SeismicPipeline":
        """Build pipeline from a plain dictionary."""
        known  = {fld.name for fld in PipelineConfig.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in cfg.items() if k in known}
        return cls(PipelineConfig(**kwargs))

    # ─────────────────────────────────────────────────────────
    #  STAGE 1  —  DATA INGESTION
    # ─────────────────────────────────────────────────────────

    def run_ingestion(self, path: str) -> np.ndarray:
        """Load seismic data from disk (SEG-Y / NumPy / HDF5)."""
        from seismic_ml.ingestion import SeismicIngestion

        with _StageTimer("Data Ingestion") as t:
            cube = SeismicIngestion.load_seismic_data(
                path,
                inline_byte    = self.cfg.segy_inline_byte,
                crossline_byte = self.cfg.segy_crossline_byte,
                chunk_size     = self.cfg.chunk_size,
            )
            self._cube = cube
            print(f"  shape={cube.shape}  dtype={cube.dtype}")

        self._stage_results.append(StageResult(
            "Data Ingestion", "ok", t.elapsed_s,
            {"shape": list(cube.shape), "dtype": str(cube.dtype)},
        ))
        return cube

    # ─────────────────────────────────────────────────────────
    #  STAGE 2  —  PREPROCESSING
    # ─────────────────────────────────────────────────────────

    def run_preprocessing(
        self, cube: Optional[np.ndarray] = None
    ) -> np.ndarray:
        from seismic_ml.preprocessing import SeismicPreprocessor

        data = cube if cube is not None else self._cube
        if data is None:
            raise RuntimeError("No cube. Run run_ingestion() first.")

        with _StageTimer("Preprocessing") as t:
            proc = SeismicPreprocessor()
            data = proc.preprocess_traces(
                data,
                bandpass          = (self.cfg.bandpass_low_hz, self.cfg.bandpass_high_hz),
                smooth_sigma      = self.cfg.smooth_sigma,
                noise_attenuation = self.cfg.noise_attenuation,
                amplitude_scaling = self.cfg.amplitude_scaling,
                sample_rate_ms    = self.cfg.sample_rate_ms,
            )
            self._cube = data

        self._stage_results.append(StageResult("Preprocessing", "ok", t.elapsed_s))
        return data

    # ─────────────────────────────────────────────────────────
    #  STAGE 3  —  DATASET
    # ─────────────────────────────────────────────────────────

    def run_build_dataset(
        self,
        cube:   Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ):
        from seismic_ml.dataset import SeismicPatchDataset, build_dataloader

        data = cube if cube is not None else self._cube
        if data is None:
            raise RuntimeError("No cube.")

        with _StageTimer("Dataset Build") as t:
            dataset = SeismicPatchDataset(
                data,
                patch_size = self.cfg.patch_size,
                n_patches  = self.cfg.n_patches,
                labels     = labels,
            )
            loader = build_dataloader(
                dataset,
                batch_size  = self.cfg.batch_size,
                num_workers = self.cfg.num_workers,
            )
            print(f"  patches={self.cfg.n_patches}  batch={self.cfg.batch_size}")

        self._stage_results.append(StageResult(
            "Dataset Build", "ok", t.elapsed_s,
            {"n_patches": self.cfg.n_patches},
        ))
        return loader

    # ─────────────────────────────────────────────────────────
    #  STAGE 4  —  MODEL
    # ─────────────────────────────────────────────────────────

    def run_build_model(
        self, checkpoint: Optional[str] = None
    ) -> torch.nn.Module:
        from seismic_ml.models import build_model

        with _StageTimer("Model Build") as t:
            kwargs: Dict[str, Any] = {
                "in_channels": self.cfg.in_channels,
                "n_classes":   self.cfg.n_classes,
            }
            if self.cfg.architecture == "unet3d":
                kwargs.update({"base_features": self.cfg.base_features,
                               "depth": self.cfg.depth})
            else:
                kwargs.update({
                    "embed_dim":   self.cfg.embed_dim,
                    "depths":      self.cfg.transformer_depths,
                    "n_heads":     self.cfg.transformer_heads,
                    "window_size": self.cfg.window_size,
                    "patch_size":  self.cfg.patch_size_model,
                })

            self._model = build_model(self.cfg.architecture, **kwargs)

            if checkpoint and Path(checkpoint).exists():
                state = torch.load(checkpoint, map_location=self.device)
                self._model.load_state_dict(state)
                print(f"  Checkpoint loaded: {checkpoint}")

            n_params = sum(p.numel() for p in self._model.parameters()
                           if p.requires_grad)
            print(f"  architecture={self.cfg.architecture}  params={n_params:,}")

        self._stage_results.append(StageResult(
            "Model Build", "ok", t.elapsed_s,
            {"architecture": self.cfg.architecture, "n_params": n_params},
        ))
        return self._model

    # ─────────────────────────────────────────────────────────
    #  STAGE 5  —  PERFORMANCE OPTIMIZATION
    # ─────────────────────────────────────────────────────────

    def run_optimize(self) -> torch.nn.Module:
        from seismic_ml.optimization import PerformanceOptimizer

        if self._model is None:
            raise RuntimeError("Build model first.")

        with _StageTimer("Performance Optimization") as t:
            PerformanceOptimizer.configure_backends()
            self._model = PerformanceOptimizer.compile_model(self._model)
            PerformanceOptimizer.gpu_memory_report()

        self._stage_results.append(
            StageResult("Performance Optimization", "ok", t.elapsed_s)
        )
        return self._model

    # ─────────────────────────────────────────────────────────
    #  STAGE 6  —  TRAINING
    # ─────────────────────────────────────────────────────────

    def run_training(self, train_loader, val_loader=None) -> Dict:
        from seismic_ml.training import train_model

        if self._model is None:
            raise RuntimeError("Build model first.")

        with _StageTimer("Training") as t:
            self._history = train_model(
                self._model,
                train_loader,
                val_loader     = val_loader,
                n_epochs       = self.cfg.n_epochs,
                lr             = self.cfg.learning_rate,
                device         = self.device,
                checkpoint_dir = self.cfg.checkpoint_dir,
            )
            best = min(self._history["train_loss"])
            print(f"  best_train_loss={best:.4f}")

        self._stage_results.append(StageResult(
            "Training", "ok", t.elapsed_s,
            {"n_epochs": self.cfg.n_epochs, "best_loss": round(best, 4)},
        ))
        return self._history

    # ─────────────────────────────────────────────────────────
    #  STAGE 7  —  INFERENCE
    # ─────────────────────────────────────────────────────────

    def run_inference(
        self, cube: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        from seismic_ml.inference import SlidingWindowInference

        if self._model is None:
            raise RuntimeError("Build model first.")
        data = cube if cube is not None else self._cube
        if data is None:
            raise RuntimeError("No cube.")

        with _StageTimer("Inference") as t:
            engine = SlidingWindowInference(
                self._model,
                patch_size = self.cfg.patch_size,
                overlap    = self.cfg.inference_overlap,
                device     = self.device,
                batch_size = self.cfg.inference_batch_size,
            )
            self._seg_map, self._conf_map = engine.run_inference(
                data, n_classes=self.cfg.n_classes
            )
            print(f"  seg_map={self._seg_map.shape}  "
                  f"conf_map min={self._conf_map.min():.3f} "
                  f"max={self._conf_map.max():.3f}")

        self._stage_results.append(StageResult(
            "Inference", "ok", t.elapsed_s,
            {"seg_shape": list(self._seg_map.shape),
             "overlap": self.cfg.inference_overlap},
        ))
        return self._seg_map, self._conf_map

    # ─────────────────────────────────────────────────────────
    #  STAGE 8  —  RESERVOIR PREDICTION
    # ─────────────────────────────────────────────────────────

    def run_reservoir_prediction(
        self,
        cube:     Optional[np.ndarray] = None,
        seg_map:  Optional[np.ndarray] = None,
        conf_map: Optional[np.ndarray] = None,
    ):
        from seismic_ml.reservoir import ReservoirPredictor

        cube     = cube     or self._cube
        seg_map  = seg_map  or self._seg_map
        conf_map = conf_map or self._conf_map
        if any(x is None for x in [cube, seg_map, conf_map]):
            raise RuntimeError("cube, seg_map, conf_map required.")

        with _StageTimer("Reservoir Prediction") as t:
            self._predictor = ReservoirPredictor(
                horizon_class       = self.cfg.horizon_class,
                min_zone_voxels     = self.cfg.min_reservoir_voxels,
                min_dhi_score       = self.cfg.min_dhi_score,
                sample_rate_ms      = self.cfg.sample_rate_ms,
                inline_spacing_m    = self.cfg.inline_spacing_m,
                crossline_spacing_m = self.cfg.crossline_spacing_m,
            )
            self._reservoir_pred = self._predictor.predict(
                cube, seg_map, conf_map,
                min_confidence=self.cfg.min_confidence,
            )
            print(f"  zones={self._reservoir_pred.n_zones}")

        self._stage_results.append(StageResult(
            "Reservoir Prediction", "ok", t.elapsed_s,
            {"n_zones": self._reservoir_pred.n_zones},
        ))
        return self._reservoir_pred

    # ─────────────────────────────────────────────────────────
    #  STAGE 9  —  RISK ASSESSMENT
    # ─────────────────────────────────────────────────────────

    def run_risk_assessment(
        self,
        cube:     Optional[np.ndarray] = None,
        seg_map:  Optional[np.ndarray] = None,
        conf_map: Optional[np.ndarray] = None,
    ):
        from seismic_ml.risk import RiskAssessor
        from scipy.ndimage import label as nd_label

        cube     = cube     or self._cube
        seg_map  = seg_map  or self._seg_map
        conf_map = conf_map or self._conf_map
        if any(x is None for x in [cube, seg_map, conf_map]):
            raise RuntimeError("cube, seg_map, conf_map required.")

        # Build zone masks from reservoir prediction
        zone_masks = None
        if self._reservoir_pred and self._reservoir_pred.zones:
            horizon_mask = (
                (seg_map == self.cfg.horizon_class) &
                (conf_map >= self.cfg.min_confidence)
            )
            labeled, _ = nd_label(horizon_mask)
            zone_masks = []
            for z in self._reservoir_pred.zones:
                mask = labeled == (z.zone_id + 1)
                if mask.any():
                    zone_masks.append((z.zone_id, mask))

        with _StageTimer("Risk Assessment") as t:
            assessor = RiskAssessor(
                fault_class         = self.cfg.fault_class,
                min_confidence      = self.cfg.min_confidence,
                min_hazard_voxels   = self.cfg.min_hazard_voxels,
                risk_threshold      = self.cfg.risk_threshold,
                inline_spacing_m    = self.cfg.inline_spacing_m,
                crossline_spacing_m = self.cfg.crossline_spacing_m,
                sample_rate_ms      = self.cfg.sample_rate_ms,
            )
            self._risk_report = assessor.assess(
                cube, seg_map, conf_map,
                reservoir_zone_masks=zone_masks,
            )
            print(f"  hazard_zones={len(self._risk_report.hazard_zones)}  "
                  f"critical={self._risk_report.n_critical}")

        self._stage_results.append(StageResult(
            "Risk Assessment", "ok", t.elapsed_s,
            {"n_hazard_zones": len(self._risk_report.hazard_zones),
             "n_critical":     self._risk_report.n_critical},
        ))
        return self._risk_report

    # ─────────────────────────────────────────────────────────
    #  STAGE 10  —  VISUALIZATION
    # ─────────────────────────────────────────────────────────

    def run_visualization(
        self, metrics: Optional[Dict] = None
    ) -> Dict[str, str]:
        paths: Dict[str, str] = {}

        if any(x is None for x in [self._cube, self._seg_map, self._conf_map]):
            print("[Visualization] Skipped — run inference first.")
            self._stage_results.append(
                StageResult("Visualization", "skipped", 0.0)
            )
            return paths

        with _StageTimer("Visualization") as t:
            # Matplotlib static PNG
            try:
                from seismic_ml.visualization.static_viz import SeismicVisualizer
                viz = SeismicVisualizer(
                    output_dir = self.cfg.viz_output_dir,
                    dpi        = self.cfg.viz_dpi,
                    fmt        = self.cfg.viz_fmt,
                )
                mpl = viz.save_all(
                    cube           = self._cube,
                    seg_map        = self._seg_map,
                    conf_map       = self._conf_map,
                    history        = self._history,
                    metrics        = metrics or {},
                    min_confidence = self.cfg.viz_min_confidence,
                )
                paths.update(mpl)
                print(f"  Matplotlib: {len(mpl)} PNGs saved → {self.cfg.viz_output_dir}/")
            except Exception as e:
                print(f"  [Matplotlib] Warning: {e}")

            # Plotly interactive HTML
            try:
                from seismic_ml.visualization.interactive_viz import (
                    SeismicPlotlyVisualizer,
                )
                pviz = SeismicPlotlyVisualizer(
                    output_dir=self.cfg.viz_interactive_dir
                )
                plotly = pviz.save_all(
                    cube           = self._cube,
                    seg_map        = self._seg_map,
                    conf_map       = self._conf_map,
                    history        = self._history,
                    metrics        = metrics or {},
                    min_confidence = self.cfg.viz_min_confidence,
                )
                paths.update(plotly)
                print(f"  Plotly: {len(plotly)} HTMLs saved → {self.cfg.viz_interactive_dir}/")
            except Exception as e:
                print(f"  [Plotly] Warning: {e}")

        self._stage_results.append(StageResult(
            "Visualization", "ok", t.elapsed_s, {"n_files": len(paths)}
        ))
        return paths

    # ─────────────────────────────────────────────────────────
    #  STAGE 11  —  REPORT GENERATION
    # ─────────────────────────────────────────────────────────

    def run_report(
        self,
        input_path:   str,
        output_files: Dict[str, str],
        t0:           float,
    ) -> PipelineReport:
        with _StageTimer("Report Generation") as t:
            top_zones = (
                self._predictor.get_top_zones(self._reservoir_pred, n=5)
                if self._reservoir_pred and self._predictor else []
            )
            critical = (
                self._risk_report.get_critical_zones()
                if self._risk_report else []
            )

            report = PipelineReport(
                input_path          = input_path,
                config              = self.cfg,
                stage_results       = list(self._stage_results),
                cube_shape          = tuple(self._cube.shape) if self._cube is not None else None,
                n_reservoir_zones   = self._reservoir_pred.n_zones if self._reservoir_pred else 0,
                n_hazard_zones      = len(self._risk_report.hazard_zones) if self._risk_report else 0,
                n_critical          = self._risk_report.n_critical if self._risk_report else 0,
                top_reservoir_zones = top_zones,
                critical_hazards    = critical,
                output_files        = output_files,
                total_elapsed_s     = time.perf_counter() - t0,
            )

            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            report_path = out_dir / "pipeline_report.json"
            report_dict = report.to_dict()
            report_dict["config"] = asdict(self.cfg)
            with open(report_path, "w") as f:
                json.dump(report_dict, f, indent=2, default=str)
            print(f"  JSON report → {report_path}")

        self._stage_results.append(
            StageResult("Report Generation", "ok", t.elapsed_s)
        )
        return report

    # ─────────────────────────────────────────────────────────
    #  MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def run_full_pipeline(
        self,
        data_path:     str,
        labels:        Optional[np.ndarray] = None,
        checkpoint:    Optional[str]        = None,
        skip_training: bool                 = False,
        metrics:       Optional[Dict]       = None,
    ) -> PipelineReport:
        """
        Execute the complete 11-stage pipeline end-to-end.

        Parameters
        ----------
        data_path      : seismic data file (.segy / .npy / .hdf5)
        labels         : optional ground-truth segmentation volume for training
        checkpoint     : pretrained model weights path (.pt)
        skip_training  : skip training stage if checkpoint is provided
        metrics        : evaluation metrics dict for visualization

        Returns
        -------
        PipelineReport with all stage results, geological findings,
        and output file paths.
        """
        t0 = time.perf_counter()
        self._stage_results = []

        print("\n" + "=" * 65)
        print("  SEISMIC ML PIPELINE  —  STARTING")
        print(f"  Input  : {data_path}")
        print(f"  Device : {self.device}")
        print(f"  Arch   : {self.cfg.architecture}")
        print("=" * 65)

        # Stage 1 — Ingestion
        cube = self.run_ingestion(data_path)

        # Stage 2 — Preprocessing
        cube = self.run_preprocessing(cube)

        # Stage 3 — Dataset
        train_loader = self.run_build_dataset(cube, labels)

        # Stage 4 — Model
        self.run_build_model(checkpoint=checkpoint)

        # Stage 5 — Optimization (non-fatal if unavailable)
        try:
            self.run_optimize()
        except Exception as e:
            print(f"  [Stage 5] Optimization skipped: {e}")
            self._stage_results.append(
                StageResult("Performance Optimization", "skipped", 0.0,
                            {"reason": str(e)})
            )

        # Stage 6 — Training
        if skip_training and checkpoint and Path(checkpoint).exists():
            print("\n  [Stage 6] Training skipped — using checkpoint.")
            self._history = {"train_loss": [0.0], "val_loss": []}
            self._stage_results.append(
                StageResult("Training", "skipped", 0.0,
                            {"reason": "checkpoint provided"})
            )
        else:
            self.run_training(train_loader)

        # Stage 7 — Inference
        seg_map, conf_map = self.run_inference(cube)

        # Stage 8 — Reservoir Prediction
        self.run_reservoir_prediction(cube, seg_map, conf_map)

        # Stage 9 — Risk Assessment
        self.run_risk_assessment(cube, seg_map, conf_map)

        # Stage 10 — Visualization
        output_files = self.run_visualization(metrics=metrics)

        # Save arrays if requested
        if self.cfg.save_arrays:
            out_dir = Path(self.cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / "seg_map.npy",  self._seg_map)
            np.save(out_dir / "conf_map.npy", self._conf_map)
            output_files["seg_map"]  = str(out_dir / "seg_map.npy")
            output_files["conf_map"] = str(out_dir / "conf_map.npy")

        # Stage 11 — Report
        report = self.run_report(data_path, output_files, t0)

        print(f"\n{report.summary()}")
        return report

"""
main.py
=======
Top-level entry point for the Seismic ML Pipeline project.

Quickstart
----------
    # Run full pipeline on a real SEG-Y file:
    python main.py --data data/survey.segy --config configs/default_config.yaml

    # Run full pipeline on synthetic data (smoke test):
    python main.py --synthetic

    # Run with specific architecture:
    python main.py --synthetic --arch transformer

    # Skip training using a pretrained checkpoint:
    python main.py --data data/survey.npy --checkpoint checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _make_synthetic_data(
    shape=(128, 128, 128),
    out_path: str = "/tmp/synthetic_seismic.npy",
    seed: int = 42,
):
    """Generate and save a synthetic seismic cube + label volume."""
    rng    = np.random.default_rng(seed)
    cube   = rng.standard_normal(shape).astype(np.float32)
    labels = np.zeros(shape, dtype=np.int64)

    for t in [32, 64, 96]:
        cube[:, :, t - 2: t + 2]   += 3.0
        labels[:, :, t - 2: t + 2]  = 0    # horizon

    for i in range(shape[0]):
        t_diag                     = min(i, shape[2] - 1)
        cube[i, :, t_diag]        -= 5.0
        labels[i, :, t_diag]       = 1     # fault

    np.save(out_path, cube)
    print(f"[main] Synthetic cube saved → {out_path}  shape={shape}")
    return out_path, labels


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Seismic ML Pipeline — end-to-end runner"
    )
    parser.add_argument("--data",       type=str,  help="Path to seismic data file")
    parser.add_argument("--config",     type=str,  help="Path to YAML config file",
                        default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", type=str,  help="Path to pretrained model .pt")
    parser.add_argument("--arch",       type=str,  default="unet3d",
                        choices=["unet3d", "transformer", "vit"],
                        help="Model architecture")
    parser.add_argument("--epochs",     type=int,  default=None,
                        help="Override n_epochs from config")
    parser.add_argument("--synthetic",  action="store_true",
                        help="Run on synthetic data (smoke test)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training (requires --checkpoint)")
    parser.add_argument("--output-dir", type=str,  default="./pipeline_output",
                        help="Directory for all outputs")
    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Import pipeline ──────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from seismic_ml.pipeline import SeismicPipeline, PipelineConfig

    # ── Build pipeline from config or defaults ───────────────
    cfg_path = Path(args.config)
    if cfg_path.exists():
        pipeline = SeismicPipeline.from_yaml(str(cfg_path))
    else:
        print(f"[main] Config not found at {cfg_path}, using defaults.")
        pipeline = SeismicPipeline(PipelineConfig())

    # ── Override from CLI ────────────────────────────────────
    pipeline.cfg.architecture = args.arch
    pipeline.cfg.output_dir   = args.output_dir
    if args.epochs:
        pipeline.cfg.n_epochs = args.epochs

    # ── Data source ──────────────────────────────────────────
    labels = None
    if args.synthetic:
        print("[main] Using synthetic seismic data …")
        # Use compact shape for smoke test
        pipeline.cfg.n_epochs            = args.epochs or 2
        pipeline.cfg.patch_size          = (32, 32, 32)
        pipeline.cfg.n_patches           = 64
        pipeline.cfg.batch_size          = 2
        pipeline.cfg.num_workers         = 0
        pipeline.cfg.inference_overlap   = 0.25
        pipeline.cfg.inference_batch_size = 2
        pipeline.cfg.min_reservoir_voxels = 20
        pipeline.cfg.min_hazard_voxels    = 10
        if pipeline.cfg.architecture != "unet3d":
            pipeline.cfg.embed_dim          = 16
            pipeline.cfg.transformer_depths = (1, 1, 1, 1)
            pipeline.cfg.transformer_heads  = (1, 2, 4, 8)
            pipeline.cfg.window_size        = (4, 4, 4)
            pipeline.cfg.patch_size_model   = (4, 4, 4)

        data_path, labels = _make_synthetic_data(shape=(64, 64, 64))

    elif args.data:
        data_path = args.data
    else:
        print("[main] ERROR: Provide --data <path> or use --synthetic.")
        sys.exit(1)

    # ── Run full pipeline ────────────────────────────────────
    report = pipeline.run_full_pipeline(
        data_path     = data_path,
        labels        = labels,
        checkpoint    = args.checkpoint,
        skip_training = args.skip_train,
    )

    print(f"\n[main] Pipeline complete.")
    print(f"[main] JSON report → {pipeline.cfg.output_dir}/pipeline_report.json")
    return report


if __name__ == "__main__":
    main()

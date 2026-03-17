"""
seismic_ml
==========
Scalable 3D seismic ML pipeline for geological interpretation.

Modules
-------
ingestion       — SEG-Y / NumPy / HDF5 data loading
preprocessing   — Signal filtering and trace normalisation
dataset         — PyTorch Dataset and DataLoader builders
models/unet3d   — 3D U-Net segmentation model
models/transformer — Swin3D / SeismicViT attention model  [Priority #2]
training        — Training loop with AMP and checkpointing
inference       — Sliding window volumetric inference
reservoir       — ReservoirPredictor with DHI scoring      [Priority #1]
risk            — RiskAssessor zone scoring                [Priority #3]
pipeline        — SeismicPipeline end-to-end runner        [Priority #4]
visualization/  — Matplotlib + Plotly visualizers
"""

__version__ = "0.2.0"
__author__  = "Seismic ML Pipeline"

from seismic_ml.reservoir import ReservoirPredictor, ReservoirPrediction
from seismic_ml.risk      import RiskAssessor, RiskReport, RiskLevel
from seismic_ml.pipeline  import SeismicPipeline, PipelineConfig, PipelineReport

__all__ = [
    "ReservoirPredictor",
    "ReservoirPrediction",
    "RiskAssessor",
    "RiskReport",
    "RiskLevel",
    "SeismicPipeline",
    "PipelineConfig",
    "PipelineReport",
]

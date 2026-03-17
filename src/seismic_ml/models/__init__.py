"""
seismic_ml.models
=================
Available model architectures:

    UNet3D            — Encoder-decoder with skip connections (baseline)
    SeismicTransformer — Swin3D-inspired attention model    (Priority #2)

Usage:
    from seismic_ml.models import build_model

    model = build_model("transformer", n_classes=3, patch_size=(64,64,64))
    model = build_model("unet3d",      n_classes=3)
"""

from seismic_ml.models.unet3d       import UNet3D
from seismic_ml.models.transformer  import SeismicTransformer, SeismicViT

__all__ = ["UNet3D", "SeismicTransformer", "SeismicViT", "build_model"]


def build_model(
    architecture: str = "unet3d",
    in_channels:  int = 1,
    n_classes:    int = 3,
    **kwargs,
):
    """
    Factory function — returns the requested model instance.

    Parameters
    ----------
    architecture : "unet3d" | "transformer" | "vit"
    in_channels  : number of input channels (1 for single-stack seismic)
    n_classes    : number of segmentation output classes
    **kwargs     : forwarded to the model constructor
    """
    arch = architecture.lower()
    if arch == "unet3d":
        return UNet3D(in_channels=in_channels, n_classes=n_classes, **kwargs)
    elif arch in ("transformer", "seismictransformer"):
        return SeismicTransformer(in_channels=in_channels, n_classes=n_classes, **kwargs)
    elif arch in ("vit", "seismicvit"):
        return SeismicViT(in_channels=in_channels, n_classes=n_classes, **kwargs)
    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Choose from: 'unet3d', 'transformer', 'vit'"
        )

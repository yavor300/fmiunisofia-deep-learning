"""Model definitions and factories."""

from __future__ import annotations

from typing import Any

__all__ = [
    "MajorityBaselineModel",
    "TinyUNet",
    "compute_majority_class",
    "create_model",
    "run_majority_baseline",
]


def __getattr__(name: str) -> Any:
    if name in {"compute_majority_class", "run_majority_baseline"}:
        from src.cityseg.models.majority_baseline import (
            compute_majority_class,
            run_majority_baseline,
        )

        return {
            "compute_majority_class": compute_majority_class,
            "run_majority_baseline": run_majority_baseline,
        }[name]
    if name in {"MajorityBaselineModel", "create_model"}:
        from src.cityseg.models.factory import MajorityBaselineModel, create_model

        return {
            "MajorityBaselineModel": MajorityBaselineModel,
            "create_model": create_model,
        }[name]
    if name == "TinyUNet":
        from src.cityseg.models.tiny_unet import TinyUNet

        return TinyUNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""Model definitions and factories."""

from __future__ import annotations

from typing import Any

__all__ = ["compute_majority_class", "run_majority_baseline"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from src.cityseg.models.majority_baseline import (
            compute_majority_class,
            run_majority_baseline,
        )

        return {
            "compute_majority_class": compute_majority_class,
            "run_majority_baseline": run_majority_baseline,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from __future__ import annotations

from typing import Any

from .fcn8s import FCN8s
from .yolov1 import YOLOv1


def load(model_name: str, **kwargs: Any):
    normalized = model_name.strip().lower()

    if normalized == "yolov1":
        from .yolov1 import load as load_yolov1

        return load_yolov1(model_name, **kwargs)

    if normalized == "fcn8s":
        from .fcn8s import load as load_fcn8s

        return load_fcn8s(model_name, **kwargs)

    raise ValueError(f"Unknown model '{model_name}'. Supported models: ['yolov1', 'fcn8s']")


__all__ = ["YOLOv1", "FCN8s", "load"]

"""Learning-rate scheduler builders and stepping helpers."""

from __future__ import annotations

from typing import Any

import torch


Scheduler = torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
) -> Scheduler | None:
    name = str(config.get("name", "none")).lower()
    if name in {"none", "null"}:
        return None
    if name in {"step", "step_decay", "step_lr"}:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1)),
        )
    if name in {"cosine", "cosine_annealing"}:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", config.get("T_max", 30))),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name in {"reduce_on_plateau", "reduce_lr_on_plateau", "plateau"}:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(config.get("mode", "max")),
            factor=float(config.get("factor", 0.1)),
            patience=int(config.get("patience", 3)),
            threshold=float(config.get("threshold", 0.0001)),
            min_lr=float(config.get("min_lr", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def step_scheduler(scheduler: Scheduler | None, validation_metric: float) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(validation_metric)
        return
    scheduler.step()

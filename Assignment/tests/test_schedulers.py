from __future__ import annotations

import unittest

import torch

from src.cityseg.training.schedulers import build_scheduler, step_scheduler


def _optimizer() -> torch.optim.Optimizer:
    model = torch.nn.Conv2d(3, 2, kernel_size=1)
    return torch.optim.AdamW(model.parameters(), lr=0.1)


class TestBuildScheduler(unittest.TestCase):
    def test_when_name_is_step_decay_then_step_lr_is_returned(self) -> None:
        scheduler = build_scheduler(
            _optimizer(),
            {"name": "step_decay", "step_size": 1, "gamma": 0.5},
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)

    def test_when_name_is_cosine_annealing_then_cosine_scheduler_is_returned(self) -> None:
        scheduler = build_scheduler(_optimizer(), {"name": "cosine_annealing", "t_max": 2})

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_when_name_is_reduce_on_plateau_then_plateau_scheduler_is_returned(self) -> None:
        scheduler = build_scheduler(_optimizer(), {"name": "reduce_on_plateau"})

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_when_name_is_none_then_none_is_returned(self) -> None:
        scheduler = build_scheduler(_optimizer(), {"name": "none"})

        self.assertIsNone(scheduler)


class TestStepScheduler(unittest.TestCase):
    def test_when_scheduler_is_step_lr_then_learning_rate_changes(self) -> None:
        optimizer = _optimizer()
        scheduler = build_scheduler(optimizer, {"name": "step_decay", "step_size": 1, "gamma": 0.5})

        optimizer.step()
        step_scheduler(scheduler, validation_metric=0.1)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)

    def test_when_scheduler_is_plateau_then_validation_metric_is_used(self) -> None:
        optimizer = _optimizer()
        scheduler = build_scheduler(
            optimizer,
            {"name": "reduce_on_plateau", "mode": "max", "factor": 0.5, "patience": 0},
        )

        step_scheduler(scheduler, validation_metric=0.5)
        step_scheduler(scheduler, validation_metric=0.4)

        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.05)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""Task 04: 500-epoch training and seed/no-seed convergence check."""

import numpy as np
from typing import List, Optional, Tuple

EPS = 1e-3
LEARNING_RATE = 1e-3
EPOCHS = 500


def create_dataset(n: int):
    return [(x, 2 * x) for x in range(n)]


def calculate_loss(w: float, dataset: List[Tuple[float, float]]) -> float:
    return float(np.mean([(w * x - y) ** 2 for x, y in dataset]))


def finite_difference_grad(w: float, dataset: List[Tuple[float, float]], eps: float = EPS) -> float:
    return (calculate_loss(w + eps, dataset) - calculate_loss(w - eps, dataset)) / (2.0 * eps)


def train_one_parameter(
    dataset: List[Tuple[float, float]],
    epochs: int,
    learning_rate: float,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    w = float(rng.uniform(0.0, 10.0))

    for _ in range(epochs):
        grad = finite_difference_grad(w, dataset)
        w -= learning_rate * grad

    return w


def main():
    dataset = create_dataset(6)

    w_seeded = train_one_parameter(dataset, EPOCHS, LEARNING_RATE, seed=42)
    print(f"Trained parameter with seed=42 after {EPOCHS} epochs: w={w_seeded:.10f}")
    print(f"Final loss (seeded): {calculate_loss(w_seeded, dataset):.12f}")

    print("\nRemoving seed (three runs):")
    for run in range(1, 4):
        w_unseeded = train_one_parameter(dataset, EPOCHS, LEARNING_RATE, seed=None)
        loss_unseeded = calculate_loss(w_unseeded, dataset)
        print(f"Run {run}: w={w_unseeded:.10f}, loss={loss_unseeded:.12f}")

    # Even without a fixed seed, training converges close to w=2 on this simple task.


if __name__ == "__main__":
    main()

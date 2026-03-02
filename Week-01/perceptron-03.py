#!/usr/bin/env python3
"""Task 03: finite-difference derivative and learning-rate experiment."""

import numpy as np
from typing import List, Tuple

SEED = 42
EPS = 1e-3
LEARNING_RATE = 1e-3
EPOCHS = 10


def create_dataset(n: int):
    return [(x, 2 * x) for x in range(n)]


def calculate_loss(w: float, dataset: List[Tuple[float, float]]) -> float:
    errors = [(w * x - y) ** 2 for x, y in dataset]
    return float(np.mean(errors))


def finite_difference_grad(w: float, dataset: List[Tuple[float, float]], eps: float = EPS) -> float:
    return (calculate_loss(w + eps, dataset) - calculate_loss(w - eps, dataset)) / (2.0 * eps)


def main():
    dataset = create_dataset(6)
    rng = np.random.default_rng(SEED)
    w = float(rng.uniform(0.0, 10.0))

    initial_loss = calculate_loss(w, dataset)
    grad = finite_difference_grad(w, dataset)

    w_no_lr = w - grad
    w_with_lr = w - LEARNING_RATE * grad

    loss_no_lr = calculate_loss(w_no_lr, dataset)
    loss_with_lr = calculate_loss(w_with_lr, dataset)

    print(f"Initial w: {w:.10f}")
    print(f"Loss before update: {initial_loss:.10f}")
    print(f"Gradient approximation: {grad:.10f}")
    print(f"Loss after update (no learning rate): {loss_no_lr:.10f}")
    print(f"Loss after update (learning rate={LEARNING_RATE}): {loss_with_lr:.10f}")

    print("\nTraining for 10 epochs with finite differences + learning rate")
    for epoch in range(1, EPOCHS + 1):
        grad = finite_difference_grad(w, dataset)
        w -= LEARNING_RATE * grad
        loss = calculate_loss(w, dataset)
        print(f"Epoch {epoch:02d} | w={w:.10f} | MSE={loss:.10f}")


if __name__ == "__main__":
    main()

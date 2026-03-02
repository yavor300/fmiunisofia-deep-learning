#!/usr/bin/env python3
"""Task 02: one-parameter model + MSE + perturbation experiment."""

import numpy as np
from typing import List, Tuple

SEED = 42


def create_dataset(n: int):
    return [(x, 2 * x) for x in range(n)]


def initialize_model(seed: int = SEED) -> float:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(0.0, 10.0))


def calculate_loss(w: float, dataset: List[Tuple[float, float]]) -> float:
    errors = [(w * x - y) ** 2 for x, y in dataset]
    return float(np.mean(errors))


def main():
    dataset = create_dataset(6)
    w = initialize_model()

    loss = calculate_loss(w, dataset)
    print(f"Initial weight w: {w:.10f}")
    print(f"MSE: {loss}")

    deltas = [0.001 * 2, 0.001, -0.001, -0.001 * 2]
    for delta in deltas:
        trial_w = w + delta
        trial_loss = calculate_loss(trial_w, dataset)
        print(f"w + ({delta:+.6f}) -> MSE: {trial_loss:.10f}")

    # With this initialization, w > 2, so increasing w raises loss and
    # decreasing w lowers loss because the true optimal parameter is w=2.


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task 01: dataset creation and weight initialization."""

import numpy as np
from typing import Optional


# Model form: y_hat = w * x
# The model has 1 trainable parameter: w.
def create_dataset(n: int):
    return [(x, 2 * x) for x in range(n)]


def initialize_weights(x: float, y: float, seed: Optional[int] = None) -> float:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(x, y))


def main():
    print(f"create_dataset(4) = {create_dataset(4)}")
    print(f"initialize_weights(0, 100) = {initialize_weights(0, 100)}")
    print(f"initialize_weights(0, 10) = {initialize_weights(0, 10)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Task 07: sigmoid function plot."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main():
    x = np.linspace(-10.0, 10.0, 400)
    y = sigmoid(x)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.title("Sigmoid Function")
    plt.xlabel("x")
    plt.ylabel("sigmoid(x)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()

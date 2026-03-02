#!/usr/bin/env python3
"""Task 06: add bias to OR/AND models."""

import numpy as np
from typing import List, Tuple

EPOCHS = 100_000
LEARNING_RATE = 0.01

OR_DATASET = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
AND_DATASET = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def predict_linear_with_bias(weights: np.ndarray, bias: float, x1: float, x2: float) -> float:
    return float(weights[0] * x1 + weights[1] * x2 + bias)


def calculate_loss(weights: np.ndarray, bias: float, dataset: List[Tuple[float, float, float]]) -> float:
    errors = []
    for x1, x2, y in dataset:
        y_hat = predict_linear_with_bias(weights, bias, x1, x2)
        errors.append((y_hat - y) ** 2)
    return float(np.mean(errors))


def train_model(dataset: List[Tuple[float, float, float]], epochs: int, learning_rate: float):
    rng = np.random.default_rng()
    weights = rng.uniform(-1.0, 1.0, size=2)
    bias = float(rng.uniform(-1.0, 1.0))

    for epoch in range(1, epochs + 1):
        grad_w = np.zeros_like(weights)
        grad_b = 0.0

        for x1, x2, y in dataset:
            x = np.array([x1, x2])
            y_hat = float(weights @ x + bias)
            err = y_hat - y
            grad_w += 2.0 * err * x
            grad_b += 2.0 * err

        grad_w /= len(dataset)
        grad_b /= len(dataset)

        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

        if epoch % 20_000 == 0 or epoch == 1 or epoch == epochs:
            loss = calculate_loss(weights, bias, dataset)
            print(f"Epoch {epoch:6d} | w={weights} b={bias:.6f} | loss={loss:.10f}")

    return weights, bias


def print_predictions(name: str, weights: np.ndarray, bias: float, dataset: List[Tuple[float, float, float]]):
    print(f"\n{name} predictions:")
    for x1, x2, y in dataset:
        y_hat = predict_linear_with_bias(weights, bias, x1, x2)
        print(f"({int(x1)}, {int(x2)}) -> pred={y_hat:.4f}, target={y:.1f}")


def main():
    or_w, or_b = train_model(OR_DATASET, EPOCHS, LEARNING_RATE)
    and_w, and_b = train_model(AND_DATASET, EPOCHS, LEARNING_RATE)

    print(f"\nFinal OR params: w={or_w}, b={or_b:.6f}, loss={calculate_loss(or_w, or_b, OR_DATASET):.10f}")
    print(f"Final AND params: w={and_w}, b={and_b:.6f}, loss={calculate_loss(and_w, and_b, AND_DATASET):.10f}")

    print_predictions("OR", or_w, or_b, OR_DATASET)
    print_predictions("AND", and_w, and_b, AND_DATASET)

    # Compared to Task 05, bias shifts predictions and usually lowers loss,
    # but linear models still cannot perfectly represent both gates as hard 0/1 mappings.


if __name__ == "__main__":
    main()

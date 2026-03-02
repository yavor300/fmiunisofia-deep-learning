#!/usr/bin/env python3
"""Task 05: train OR and AND with two weights (no bias)."""

import numpy as np
from typing import List, Tuple

EPOCHS = 100_000
LEARNING_RATE = 0.01
PRINT_EACH_EPOCH = False

# Model forms:
# OR model:  y_hat = w1_or * x1 + w2_or * x2
# AND model: y_hat = w1_and * x1 + w2_and * x2
# Each model has 2 trainable parameters.
OR_DATASET = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]
AND_DATASET = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]


def predict_linear(weights: np.ndarray, x1: float, x2: float) -> float:
    return float(weights[0] * x1 + weights[1] * x2)


def calculate_loss(weights: np.ndarray, dataset: List[Tuple[float, float, float]]) -> float:
    errors = []
    for x1, x2, y in dataset:
        y_hat = predict_linear(weights, x1, x2)
        errors.append((y_hat - y) ** 2)
    return float(np.mean(errors))


def train_model(
    dataset: List[Tuple[float, float, float]],
    epochs: int,
    learning_rate: float,
    print_each_epoch: bool = PRINT_EACH_EPOCH,
):
    # Random init without seed, as required.
    rng = np.random.default_rng()
    weights = rng.uniform(-1.0, 1.0, size=2)

    for epoch in range(1, epochs + 1):
        grad = np.zeros_like(weights)

        for x1, x2, y in dataset:
            x = np.array([x1, x2])
            y_hat = float(weights @ x)
            grad += 2.0 * (y_hat - y) * x

        grad /= len(dataset)
        weights -= learning_rate * grad

        loss = calculate_loss(weights, dataset)
        if print_each_epoch:
            print(f"Epoch {epoch} | w={weights} | loss={loss:.10f}")
        elif epoch % 20_000 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:6d} | w={weights} | loss={loss:.10f}")

    return weights


def print_predictions(name: str, weights: np.ndarray, dataset: List[Tuple[float, float, float]]):
    print(f"\n{name} predictions:")
    for x1, x2, y in dataset:
        y_hat = predict_linear(weights, x1, x2)
        print(f"({int(x1)}, {int(x2)}) -> pred={y_hat:.4f}, target={y:.1f}")


def main():
    or_weights = train_model(OR_DATASET, EPOCHS, LEARNING_RATE)
    and_weights = train_model(AND_DATASET, EPOCHS, LEARNING_RATE)

    print(f"\nFinal OR weights: {or_weights}, loss={calculate_loss(or_weights, OR_DATASET):.10f}")
    print(f"Final AND weights: {and_weights}, loss={calculate_loss(and_weights, AND_DATASET):.10f}")

    print_predictions("OR", or_weights, OR_DATASET)
    print_predictions("AND", and_weights, AND_DATASET)

    # Without non-linearity and bias, confidence is limited:
    # predictions often stay in-between 0 and 1 instead of very close to the targets.


if __name__ == "__main__":
    main()

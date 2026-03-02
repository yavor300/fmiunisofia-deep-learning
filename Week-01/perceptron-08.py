#!/usr/bin/env python3
"""Task 08: compare training with and without sigmoid and plot loss."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCHS = 100_000
LEARNING_RATE = 0.05

OR_DATASET = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 1.0)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_linear_with_bias(dataset, epochs, learning_rate):
    rng = np.random.default_rng(7)
    w = rng.uniform(-1.0, 1.0, size=2)
    b = float(rng.uniform(-1.0, 1.0))
    losses = []

    for _ in range(epochs):
        grad_w = np.zeros(2)
        grad_b = 0.0
        mse_terms = []

        for x1, x2, y in dataset:
            x = np.array([x1, x2])
            y_hat = float(w @ x + b)
            err = y_hat - y
            mse_terms.append(err ** 2)
            grad_w += 2.0 * err * x
            grad_b += 2.0 * err

        grad_w /= len(dataset)
        grad_b /= len(dataset)
        losses.append(float(np.mean(mse_terms)))

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, np.array(losses)


def train_sigmoid_with_bias(dataset, epochs, learning_rate):
    rng = np.random.default_rng(7)
    w = rng.uniform(-1.0, 1.0, size=2)
    b = float(rng.uniform(-1.0, 1.0))
    losses = []

    for _ in range(epochs):
        grad_w = np.zeros(2)
        grad_b = 0.0
        bce_terms = []

        for x1, x2, y in dataset:
            x = np.array([x1, x2])
            z = float(w @ x + b)
            y_hat = float(sigmoid(z))
            y_hat_clipped = float(np.clip(y_hat, 1e-9, 1.0 - 1e-9))
            bce_terms.append(-(y * np.log(y_hat_clipped) + (1.0 - y) * np.log(1.0 - y_hat_clipped)))

            # BCE with sigmoid: derivative wrt z is (y_hat - y)
            dz = y_hat - y
            grad_w += dz * x
            grad_b += dz

        grad_w /= len(dataset)
        grad_b /= len(dataset)
        losses.append(float(np.mean(bce_terms)))

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, np.array(losses)


def main():
    _, _, linear_losses = train_linear_with_bias(OR_DATASET, EPOCHS, LEARNING_RATE)
    _, _, sigmoid_losses = train_sigmoid_with_bias(OR_DATASET, EPOCHS, LEARNING_RATE)

    print(f"Final linear loss (MSE): {linear_losses[-1]:.10f}")
    print(f"Final sigmoid loss (BCE): {sigmoid_losses[-1]:.10f}")

    plt.figure(figsize=(9, 5))
    plt.plot(linear_losses, label="Without sigmoid (MSE)")
    plt.plot(sigmoid_losses, label="With sigmoid (BCE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over 100,000 Epochs")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"), dpi=120)
    plt.close()

    # Comparison:
    # With sigmoid the model outputs stay in [0, 1], training is better suited
    # for classification, and the confidence on class-like targets improves.


if __name__ == "__main__":
    main()

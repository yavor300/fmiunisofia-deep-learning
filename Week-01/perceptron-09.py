#!/usr/bin/env python3
"""Task 09: NAND gate using the same reusable perceptron training logic."""

import numpy as np

EPOCHS = 100_000
LEARNING_RATE = 0.05

NAND_DATASET = [(0.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def train_sigmoid_with_bias(dataset, epochs, learning_rate, seed=21):
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, size=2)
    b = float(rng.uniform(-1.0, 1.0))

    for epoch in range(1, epochs + 1):
        grad_w = np.zeros(2)
        grad_b = 0.0

        for x1, x2, y in dataset:
            x = np.array([x1, x2])
            z = float(w @ x + b)
            y_hat = float(sigmoid(z))
            dz = y_hat - y
            grad_w += dz * x
            grad_b += dz

        grad_w /= len(dataset)
        grad_b /= len(dataset)
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch % 20_000 == 0 or epoch == 1 or epoch == epochs:
            preds = [float(sigmoid(w[0] * x1 + w[1] * x2 + b)) for x1, x2, _ in dataset]
            loss = float(
                np.mean(
                    [
                        -(
                            y * np.log(np.clip(p, 1e-9, 1.0 - 1e-9))
                            + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-9, 1.0 - 1e-9))
                        )
                        for p, (_, _, y) in zip(preds, dataset)
                    ]
                )
            )
            print(f"Epoch {epoch:6d} | loss={loss:.10f}")

    return w, b


def main():
    w, b = train_sigmoid_with_bias(NAND_DATASET, EPOCHS, LEARNING_RATE)

    print(f"\nLearned NAND parameters: w={w}, b={b:.6f}")
    print("NAND predictions:")
    for x1, x2, y in NAND_DATASET:
        proba = float(sigmoid(w[0] * x1 + w[1] * x2 + b))
        pred = int(proba >= 0.5)
        print(f"({int(x1)}, {int(x2)}) -> pred={pred}, proba={proba:.4f}, target={int(y)}")

    # Yes, we can reuse the same model/training code from AND/OR and only swap dataset.


if __name__ == "__main__":
    main()

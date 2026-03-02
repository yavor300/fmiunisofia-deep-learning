#!/usr/bin/env python3
"""Task 10: XOR model class with trainable parameters and biases."""

import numpy as np

X_XOR = np.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ]
)
Y_XOR = np.array([[0.0], [1.0], [1.0], [0.0]])

EPOCHS = 20_000
LEARNING_RATE = 1.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Xor:
    """Simple 2-2-1 MLP for XOR."""

    def __init__(self, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.5, size=(2, 2))
        self.b1 = np.zeros((1, 2))
        self.w2 = rng.normal(0.0, 0.5, size=(2, 1))
        self.b2 = np.zeros((1, 1))

    def _forward_batch(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h1 = sigmoid(z1)
        z2 = h1 @ self.w2 + self.b2
        y_hat = sigmoid(z2)
        return h1, y_hat

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = EPOCHS, lr: float = LEARNING_RATE):
        m = x.shape[0]
        for epoch in range(1, epochs + 1):
            h1, y_hat = self._forward_batch(x)

            dz2 = y_hat - y
            dw2 = (h1.T @ dz2) / m
            db2 = np.sum(dz2, axis=0, keepdims=True) / m

            dh1 = dz2 @ self.w2.T
            dz1 = dh1 * h1 * (1.0 - h1)
            dw1 = (x.T @ dz1) / m
            db1 = np.sum(dz1, axis=0, keepdims=True) / m

            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w1 -= lr * dw1
            self.b1 -= lr * db1

            if epoch % 5000 == 0 or epoch == 1 or epoch == epochs:
                eps = 1e-9
                y_clip = np.clip(y_hat, eps, 1.0 - eps)
                bce = -np.mean(y * np.log(y_clip) + (1.0 - y) * np.log(1.0 - y_clip))
                print(f"Epoch {epoch:6d} | BCE={bce:.10f}")

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._forward_batch(x)[1]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(int)


def forward(model: Xor, x1: float, x2: float) -> float:
    """Required standalone forward function taking model + two inputs."""
    point = np.array([[x1, x2]], dtype=float)
    return float(model.predict_proba(point)[0, 0])


def main():
    model = Xor(seed=7)
    model.train(X_XOR, Y_XOR)

    print("\nXOR predictions:")
    for x1, x2 in X_XOR:
        proba = forward(model, float(x1), float(x2))
        pred = int(proba >= 0.5)
        target = int((x1 + x2) % 2)
        print(f"({int(x1)}, {int(x2)}) -> pred={pred}, proba={proba:.4f}, target={target}")


if __name__ == "__main__":
    main()

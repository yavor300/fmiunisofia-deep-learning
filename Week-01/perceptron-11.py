#!/usr/bin/env python3
"""Task 11: MLP for square(x) with built-in unit tests."""

import unittest
import sys

import numpy as np


class SquareMLP:
    def __init__(self, hidden_size: int = 16, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0.0, 0.4, size=(1, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = rng.normal(0.0, 0.4, size=(hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    @staticmethod
    def _tanh(x):
        return np.tanh(x)

    @staticmethod
    def _tanh_derivative(x):
        t = np.tanh(x)
        return 1.0 - t * t

    def forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        h1 = self._tanh(z1)
        y = h1 @ self.w2 + self.b2
        return z1, h1, y

    def train(self, x: np.ndarray, y_true: np.ndarray, epochs: int = 5000, lr: float = 0.01):
        n = x.shape[0]
        losses = []

        for _ in range(epochs):
            z1, h1, y_pred = self.forward(x)
            err = y_pred - y_true
            loss = float(np.mean(err ** 2))
            losses.append(loss)

            dloss_dy = (2.0 / n) * err
            dw2 = h1.T @ dloss_dy
            db2 = np.sum(dloss_dy, axis=0, keepdims=True)

            dloss_dh = dloss_dy @ self.w2.T
            dloss_dz1 = dloss_dh * self._tanh_derivative(z1)
            dw1 = x.T @ dloss_dz1
            db1 = np.sum(dloss_dz1, axis=0, keepdims=True)

            self.w2 -= lr * dw2
            self.b2 -= lr * db2
            self.w1 -= lr * dw1
            self.b1 -= lr * db1

        return np.array(losses)

    def predict(self, x_values):
        x = np.array(x_values, dtype=float).reshape(-1, 1)
        return self.forward(x)[2].flatten()


def build_square_dataset(n: int = 200):
    x = np.linspace(-10.0, 10.0, n).reshape(-1, 1)
    y = x ** 2
    return x, y


def main():
    x_train, y_train = build_square_dataset(200)
    model = SquareMLP(hidden_size=16, seed=123)
    losses = model.train(x_train, y_train, epochs=5000, lr=0.01)

    print(f"Final training MSE: {losses[-1]:.6f}")
    for value in [2.0, 7.0, -3.0, 0.5]:
        pred = model.predict([value])[0]
        print(f"model({value}) = {pred:.4f} (target {value ** 2:.4f})")


class TestSquareMLP(unittest.TestCase):
    def test_output_shape(self):
        model = SquareMLP(hidden_size=8, seed=1)
        preds = model.predict([2.0, 3.0, 4.0])
        self.assertEqual(preds.shape, (3,))

    def test_training_reduces_loss(self):
        x_train, y_train = build_square_dataset(120)
        model = SquareMLP(hidden_size=16, seed=123)
        losses = model.train(x_train, y_train, epochs=3000, lr=0.01)
        self.assertLess(losses[-1], losses[0])

    def test_reasonable_generalization(self):
        x_train, y_train = build_square_dataset(200)
        model = SquareMLP(hidden_size=16, seed=123)
        model.train(x_train, y_train, epochs=5000, lr=0.01)

        x_test = np.array([-9.0, -4.0, -1.5, 0.0, 2.0, 5.5, 8.0])
        y_test = x_test ** 2
        y_pred = model.predict(x_test)
        mse = float(np.mean((y_pred - y_test) ** 2))
        self.assertLess(mse, 20.0)


if __name__ == "__main__":
    if "--test" in sys.argv:
        unittest.main(argv=[sys.argv[0]])
    else:
        main()

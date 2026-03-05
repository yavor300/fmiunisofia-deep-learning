import numpy as np

from neuron_lib import MLP, zero_grad


def main() -> None:
    np.random.seed(42)
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    n = MLP(in_channels=3, hidden_channels=[4, 4, 1])
    y_preds = [n(x) for x in xs]
    loss = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ys, y_preds))
    print(f"Before gradient step: {loss}")

    zero_grad(n.parameters())
    loss.backward()

    for parameter in n.parameters():
        parameter.data += -0.1 * parameter.grad

    y_preds = [n(x) for x in xs]
    loss = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ys, y_preds))
    print(f"After gradient step: {loss}")


if __name__ == "__main__":
    main()

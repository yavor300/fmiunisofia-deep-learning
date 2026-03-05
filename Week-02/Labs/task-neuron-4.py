import numpy as np

from neuron_lib import MLP


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
    print(f"{y_preds=}")

    loss = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ys, y_preds))
    print(f"Loss = {loss}")

    loss.backward()

    print(
        "Gradient of the first weight in the first neuron in the first layer: "
        f"{n.layers[0].neurons[0].w[0].grad}."
    )

    print(f"Parameters: {n.parameters()}")


if __name__ == "__main__":
    main()

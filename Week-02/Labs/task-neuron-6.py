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

    # Notice: this gradient is usually not 0.0 after the first epoch.
    # Problem: we're reusing stale gradients from the previous step.
    for _ in range(10):
        print(
            "Gradient of the first weight in the first neuron in the first layer: "
            f"{n.layers[0].neurons[0].w[0].grad}."
        )

        y_preds = [n(x) for x in xs]
        loss = sum((y_pred - y_true) ** 2 for y_true, y_pred in zip(ys, y_preds))
        print(f"Loss={loss.data}")

        loss.backward()

        for parameter in n.parameters():
            parameter.data += -0.1 * parameter.grad

    print("Predictions:")
    print([n(x).data for x in xs])


if __name__ == "__main__":
    main()

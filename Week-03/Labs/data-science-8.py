from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def function(x):
    return x**4 + x**3 - 5 * x**2


def optimize_and_collect(lr: float, momentum: float) -> tuple[float, float, list[tuple[torch.Tensor, torch.Tensor]]]:
    if lr > 0.05:
        raise ValueError("choose a learning <= 0.05")

    x = torch.tensor(2.0, requires_grad=True)
    buffer = torch.zeros_like(x.data)
    values: list[tuple[torch.Tensor, torch.Tensor]] = []

    for _ in range(20):
        y = function(x)
        values.append((x.detach().clone(), y.detach().clone()))
        y.backward()

        d_p = x.grad.data
        if momentum != 0:
            buffer.mul_(momentum).add_(d_p)
            d_p = buffer

        x.data.add_(d_p, alpha=-lr)
        x.grad.zero_()

    return x.item(), function(x).item(), values


def optimize_and_plot(lr: float, momentum: float) -> tuple[float, float]:
    final_x, final_y, values = optimize_and_collect(lr, momentum)

    x_plot = np.arange(-3, 2, 0.001)
    y_plot = function(x_plot)

    plt.figure(figsize=(10, 5))
    plt.plot([v[0].item() for v in values], [v[1].item() for v in values], "r-X", linewidth=2, markersize=7)
    for i in range(20):
        plt.text(values[i][0].item() + 0.1, values[i][1].item(), f"step {i}", fontdict={"color": "r"})
    plt.plot(x_plot, y_plot, linewidth=2)
    plt.grid()
    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.legend(["Optimizer steps", "Square function"])
    plt.tight_layout()
    plt.savefig(Path(__file__).with_name("data-science-8-optimization.png"))

    return final_x, final_y


def find_good_hyperparameters() -> tuple[float, float, float, float]:
    xs = np.linspace(-3, 2, 10000)
    ys = function(xs)
    global_x = float(xs[np.argmin(ys)])
    global_y = float(np.min(ys))

    best = None
    for lr in np.linspace(0.001, 0.05, 120):
        for momentum in np.linspace(0.0, 0.99, 120):
            final_x, final_y, _ = optimize_and_collect(float(lr), float(momentum))
            score = abs(final_x - global_x) + abs(final_y - global_y)
            if best is None or score < best[0]:
                best = (score, float(lr), float(momentum), final_x, final_y)

    assert best is not None
    return best[1], best[2], best[3], best[4]


def main() -> None:
    lr, momentum, approx_x, approx_y = find_good_hyperparameters()
    final_x, final_y = optimize_and_plot(lr, momentum)

    print(f"chosen learning rate: {lr}")
    print(f"chosen momentum: {momentum}")
    print(f"approximate global minimum reached near x={approx_x}, y={approx_y}")
    print(f"final point after plotting run: x={final_x}, y={final_y}")


if __name__ == "__main__":
    main()

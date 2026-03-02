#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 123
N_STEPS = 100


def take_step(step: int, rng: np.random.Generator):
    dice = rng.integers(1, 7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step += 1
    else:
        step += rng.integers(1, 7)

    return step


def main():
    rng = np.random.default_rng(SEED)
    step = 0
    random_walk = [step]

    for _ in range(N_STEPS):
        step = take_step(step, rng)
        random_walk.append(step)

    plt.figure(figsize=(9, 5))
    plt.plot(random_walk)
    plt.xlabel("Throw")
    plt.ylabel("Step")
    plt.title("Random Walk")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()

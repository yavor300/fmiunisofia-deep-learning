#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 123
N_STEPS = 100
N_WALKS = 5


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
    all_walks = []

    for _ in range(N_WALKS):
        step = 0
        random_walk = [step]

        for _ in range(N_STEPS):
            step = take_step(step, rng)
            random_walk.append(step)

        all_walks.append(random_walk)

    np_all_walks = np.array(all_walks)
    np_all_walks_t = np.transpose(np_all_walks)

    plt.figure(figsize=(9, 5))
    plt.plot(np_all_walks_t)
    plt.xlabel("Throw")
    plt.ylabel("Step")
    plt.title("Five Random Walks")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()

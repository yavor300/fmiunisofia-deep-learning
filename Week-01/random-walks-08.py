#!/usr/bin/env python3
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 123
N_STEPS = 100
N_WALKS = 500
CLUMSY_PROBABILITY = 0.005
TARGET_STEP = 60


def take_step(step: int, rng: np.random.Generator):
    dice = rng.integers(1, 7)

    if dice <= 2:
        step = max(0, step - 1)
    elif dice <= 5:
        step += 1
    else:
        step += rng.integers(1, 7)

    if rng.random() <= CLUMSY_PROBABILITY:
        step = 0

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
    end_points = np_all_walks[:, -1]

    odds = np.mean(end_points >= TARGET_STEP)
    # With seed 123 and the given setup, the estimated odds are 0.552 (55.2%).
    print(f"Odds of reaching {TARGET_STEP} steps: {odds:.3f}")

    plt.figure(figsize=(9, 5))
    plt.hist(end_points, bins=20)
    plt.xlabel("Final Step After 100 Throws")
    plt.ylabel("Frequency")
    plt.title("Distribution of Random Walk End Points")
    plt.tight_layout()
    plt.savefig(Path(__file__).with_suffix(".png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()

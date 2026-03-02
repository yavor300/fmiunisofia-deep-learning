#!/usr/bin/env python3
import numpy as np

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

    print(all_walks)


if __name__ == "__main__":
    main()

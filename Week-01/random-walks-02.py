#!/usr/bin/env python3
import numpy as np

SEED = 123
N_STEPS = 100


def take_step(step: int, rng: np.random.Generator):
    dice = rng.integers(1, 7)

    if dice <= 2:
        # Bug behavior from task: the step can become negative
        step -= 1
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

    print(random_walk)


if __name__ == "__main__":
    main()

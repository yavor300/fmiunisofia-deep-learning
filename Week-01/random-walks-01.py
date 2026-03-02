#!/usr/bin/env python3
import numpy as np

SEED = 123


def take_step(step: int, rng: np.random.Generator):
    dice = rng.integers(1, 7)

    if dice <= 2:
        step -= 1
    elif dice <= 5:
        step += 1
    else:
        step += rng.integers(1, 7)

    return step, int(dice)


def main():
    rng = np.random.default_rng(SEED)

    random_float = rng.random()
    random_integer_1 = rng.integers(1, 7)
    random_integer_2 = rng.integers(1, 7)

    print(f"Random float: {random_float}")
    print(f"Random integer 1: {random_integer_1}")
    print(f"Random integer 2: {random_integer_2}")

    step = 50
    print(f"Before throw step = {step}")
    step, dice = take_step(step, rng)
    print(f"After throw dice = {dice}")
    print(f"After throw step = {step}")


if __name__ == "__main__":
    main()

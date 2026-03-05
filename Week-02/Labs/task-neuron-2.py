import numpy as np

from neuron_lib import Linear


def main() -> None:
    np.random.seed(42)
    n = Linear(in_features=2, out_features=3)
    x = [2.0, 3.0]
    print(n(x))


if __name__ == "__main__":
    main()

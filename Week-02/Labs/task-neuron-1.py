import numpy as np

from neuron_lib import Neuron


def main() -> None:
    np.random.seed(42)
    n = Neuron(2)
    x = [2.0, 3.0]
    print(n(x))


if __name__ == "__main__":
    main()

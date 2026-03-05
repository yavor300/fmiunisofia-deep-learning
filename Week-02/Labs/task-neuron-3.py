import numpy as np

from neuron_lib import MLP
from value import draw_dot


def main() -> None:
    np.random.seed(42)
    x = [2.0, 3.0, -1.0]

    n1 = MLP(in_channels=3, hidden_channels=[4, 4, 1])
    y1 = n1(x)
    print(y1)

    n2 = MLP(in_channels=3, hidden_channels=[4, 4, 2])
    print(n2(x))

    draw_dot(y1, filename="neuron_03_result").render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

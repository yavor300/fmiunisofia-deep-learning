import numpy as np

from value import Value


def main() -> None:
    x = Value(2.0, label="x")

    expected = Value(4.0)

    actuals = {
        "actual_sum_l": x + 2.0,
        "actual_sum_r": 2.0 + x,
        "actual_mul_l": x * 2.0,
        "actual_mul_r": 2.0 * x,
        "actual_div_r": (x + 6.0) / 2.0,
        "actual_pow_l": x**2,
    }

    assert x.exp().data == np.exp(2), (
        f"Mismatch for exponentiating Euler's number: expected {np.exp(2)}, but got {x.exp().data}."
    )

    for actual_name, actual_value in actuals.items():
        assert actual_value.data == expected.data, (
            f"Mismatch for {actual_name}: expected {expected.data}, but got {actual_value.data}."
        )

    print("All tests passed!")


if __name__ == "__main__":
    main()

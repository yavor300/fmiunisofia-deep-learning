from dl_lib import nn


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("install torch to run tensor tests")
        return

    module = nn.Sigmoid()
    x = torch.tensor([-2.0, 0.0, 2.0])
    expected = 1.0 / (1.0 + torch.exp(-x))
    actual = module(x)

    assert torch.allclose(actual, expected)
    print("task 2 passed")


if __name__ == "__main__":
    main()

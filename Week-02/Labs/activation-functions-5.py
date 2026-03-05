from dl_lib import nn


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("install torch to run tensor tests")
        return

    module = nn.LeakyReLU(negative_slope=0.1)
    x = torch.tensor([-3.0, -1.0, 0.0, 2.0])
    expected = torch.maximum(x, torch.zeros_like(x)) + 0.1 * torch.minimum(x, torch.zeros_like(x))
    actual = module(x)

    assert torch.allclose(actual, expected)
    print("task 5 passed")


if __name__ == "__main__":
    main()

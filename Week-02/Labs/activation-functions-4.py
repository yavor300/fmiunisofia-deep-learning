from dl_lib import nn


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("install torch to run tensor tests")
        return

    module = nn.ReLU()
    x = torch.tensor([-2.0, 0.0, 2.0])
    expected = torch.relu(x)
    actual = module(x)

    assert torch.allclose(actual, expected)
    print("task 4 passed")


if __name__ == "__main__":
    main()

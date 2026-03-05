from dl_lib import nn


def main() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        print("install torch to run tensor tests")
        return

    model = nn.Sequential(nn.ReLU(), nn.Tanh(), nn.Sigmoid())
    x = torch.tensor([-2.0, -0.5, 0.0, 1.0])
    expected = torch.sigmoid(torch.tanh(torch.relu(x)))
    actual = model(x)
    assert torch.allclose(actual, expected)

    seq = nn.Sequential(nn.ReLU())
    seq.append(nn.Tanh())
    assert len(seq.modules) == 2

    seq2 = nn.Sequential(nn.Sigmoid())
    seq.extend(seq2)
    assert len(seq.modules) == 3

    seq.insert(1, nn.LeakyReLU())
    assert isinstance(seq.modules[1], nn.LeakyReLU)

    y = seq(torch.tensor([-1.0, 2.0]))
    assert isinstance(y, torch.Tensor)

    print("task 6 passed")


if __name__ == "__main__":
    main()

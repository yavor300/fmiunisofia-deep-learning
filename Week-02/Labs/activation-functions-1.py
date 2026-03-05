from dl_lib.nn import Module


class Double(Module):
    def forward(self, input_tensor):
        return input_tensor * 2


def main() -> None:
    module = Double()
    assert module.forward(3) == 6
    assert module(3) == 6
    print("Task 1")


if __name__ == "__main__":
    main()

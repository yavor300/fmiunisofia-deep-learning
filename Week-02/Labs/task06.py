from value import Value, trace


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    nodes, edges = trace(x)
    print("x")
    print(f"{nodes=}")
    print(f"{edges=}")

    nodes, edges = trace(y)
    print("y")
    print(f"{nodes=}")
    print(f"{edges=}")

    nodes, edges = trace(z)
    print("z")
    print(f"{nodes=}")
    print(f"{edges=}")

    nodes, edges = trace(result)
    print("result")
    print(f"{nodes=}")
    print(f"{edges=}")


if __name__ == "__main__":
    main()

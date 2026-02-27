from value import Value, draw_dot


def main() -> None:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    e = a * b
    e.label = "e"

    d = e + c
    d.label = "d"

    f = Value(-2.0, label="f")

    result = d * f
    result.label = "L"

    draw_dot(result, filename="02_result").render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

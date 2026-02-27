from value import Value, draw_dot


def main() -> None:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")
    f = Value(5.0, label="f")

    e = a * b
    e.label = "e"
    d = e + c
    d.label = "d"
    l = d * f
    l.label = "L"

    # Gradient of the output w.r.t. itself.
    l.grad = 1.0

    draw_dot(l, filename="03_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

from value import Value, draw_dot


def main() -> None:
    a = Value(2.0, label="a")
    b = Value(-3.0, label="b")
    c = Value(10.0, label="c")

    d = a + b
    d.label = "d"

    l = d + c
    l.label = "L"

    l.backward()

    draw_dot(l, filename="08_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

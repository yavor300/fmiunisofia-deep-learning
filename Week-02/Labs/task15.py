from value import Value, draw_dot


def main() -> None:
    x = Value(5.0, label="x")
    y = Value(10.0, label="y")

    z = x + y
    z.label = "z"

    z.backward()

    draw_dot(z, filename="08_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

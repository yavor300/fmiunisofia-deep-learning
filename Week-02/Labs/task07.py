from value import Value, draw_dot


def main() -> None:
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(10.0)
    result = x * y + z

    draw_dot(result).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

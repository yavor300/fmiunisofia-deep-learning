from value import Value, draw_dot


def main() -> None:
    x = Value(3.0, label="x")
    y = x + x
    y.label = "y"

    y.backward()

    print(f"x.grad = {x.grad}")
    draw_dot(y, filename="10_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

from value import Value, draw_dot


def main() -> None:
    x1 = Value(2.0, label="x1")
    x2 = Value(0.0, label="x2")
    w1 = Value(-3.0, label="w1")
    w2 = Value(1.0, label="w2")
    b = Value(6.8813735870195432, label="b")

    x1w1 = x1 * w1
    x1w1.label = "x1*w1"

    x2w2 = x2 * w2
    x2w2.label = "x2*w2"

    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2.label = "x1*w1 + x2*w2"

    n = x1w1x2w2 + b
    n.label = "n"

    o = n.tanh()
    o.label = "o"

    # do/dо = 1
    o.grad = 1.0

    # do/dn = 1 - tanh(n)^2 = 1 - o^2
    n.grad = (1 - o.data**2) * o.grad

    # dn/d(x1w1 + x2w2) = 1 and dn/db = 1
    x1w1x2w2.grad = 1.0 * n.grad
    b.grad = 1.0 * n.grad

    # d(x1w1 + x2w2)/d(x1w1) = 1 and d(x1w1 + x2w2)/d(x2w2) = 1
    x1w1.grad = 1.0 * x1w1x2w2.grad
    x2w2.grad = 1.0 * x1w1x2w2.grad

    # d(x1*w1)/dx1 = w1 and d(x1*w1)/dw1 = x1
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad

    # d(x2*w2)/dx2 = w2 and d(x2*w2)/dw2 = x2
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad

    draw_dot(o, filename="07_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

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

    logit = x1w1x2w2 + b
    logit.label = "logit"

    l = logit.tanh()
    l.label = "L"

    # dl/dl = 1
    l.grad = 1.0

    # dl/dlogit = 1 - tanh(n)^2 = 1 - o^2
    logit.grad = (1 - l.data**2) * l.grad

    # dlogit/d(x1w1 + x2w2) = 1 ; dlogit/db = 1
    x1w1x2w2.grad = 1.0 * logit.grad
    b.grad = 1.0 * logit.grad

    # d(x1w1 + x2w2)/d(x1w1) = 1 ; d(x1w1 + x2w2)/d(x2w2) = 1
    x1w1.grad = 1.0 * x1w1x2w2.grad
    x2w2.grad = 1.0 * x1w1x2w2.grad

    # d(x1*w1)/dx1 = w1 ; d(x1*w1)/dw1 = x1
    x1.grad = w1.data * x1w1.grad
    w1.grad = x1.data * x1w1.grad

    # d(x2*w2)/dx2 = w2 ; d(x2*w2)/dw2 = x2
    x2.grad = w2.data * x2w2.grad
    w2.grad = x2.data * x2w2.grad

    draw_dot(l, filename="07_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

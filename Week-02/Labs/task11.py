from value import Value


def build(a_data: float, b_data: float, c_data: float, f_data: float):
    a = Value(a_data, label="a")
    b = Value(b_data, label="b")
    c = Value(c_data, label="c")
    f = Value(f_data, label="f")

    e = a * b
    d = e + c
    l = d * f

    return a, b, c, f, l


def main() -> None:
    a, b, c, f, l = build(2.0, -3.0, 10.0, 5.0)
    l.backward()

    print(f"Old L = {l.data}")

    learning_rate = 0.01
    for node in [a, b, c, f]:
        node.data += -learning_rate * node.grad

    _, _, _, _, new_l = build(a.data, b.data, c.data, f.data)
    print(f"New L = {new_l.data}")


if __name__ == "__main__":
    main()

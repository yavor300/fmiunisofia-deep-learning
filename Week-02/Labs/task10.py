from __future__ import annotations

from typing import Dict

from value import Value, draw_dot


EPSILON = 0.001


def build_graph(vals: Dict[str, float]):
    a = Value(vals["a"], label="a")
    b = Value(vals["b"], label="b")
    c = Value(vals["c"], label="c")
    f = Value(vals["f"], label="f")

    e = a * b
    e.label = "e"

    d = e + c
    d.label = "d"

    l = d * f
    l.label = "L"

    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f, "L": l}


def manual_der(var_name: str, vals: Dict[str, float], epsilon: float = EPSILON) -> float:
    vals_plus = vals.copy()
    vals_plus[var_name] += epsilon
    l_plus = build_graph(vals_plus)["L"].data

    vals_minus = vals.copy()
    vals_minus[var_name] -= epsilon
    l_minus = build_graph(vals_minus)["L"].data

    return (l_plus - l_minus) / (2 * epsilon)


def main() -> None:
    vals = {"a": 2.0, "b": -3.0, "c": 10.0, "f": 5.0}
    nodes = build_graph(vals)

    a = nodes["a"]
    b = nodes["b"]
    c = nodes["c"]
    d = nodes["d"]
    e = nodes["e"]
    f = nodes["f"]
    l = nodes["L"]

    # dL/dL = 1
    l.grad = 1.0

    # dL/dd = f and dL/df = d
    d.grad = f.data * l.grad
    f.grad = d.data * l.grad

    # dL/de = dL/dd * dd/de and dL/dc = dL/dd * dd/dc
    e.grad = 1.0 * d.grad
    c.grad = 1.0 * d.grad

    # dL/da = dL/de * de/da and dL/db = dL/de * de/db
    a.grad = b.data * e.grad
    b.grad = a.data * e.grad

    for name in ["a", "b", "c", "f"]:
        approx = manual_der(name, vals)
        node = nodes[name]
        assert abs(node.grad - approx) < 1e-3, (
            f"Gradient mismatch for {name}: manual={node.grad}, finite_diff={approx}"
        )

    print("Manual gradients match finite-difference checks.")
    draw_dot(l, filename="04_result", show_grad=True).render(directory="./graphviz_output", view=False)


if __name__ == "__main__":
    main()

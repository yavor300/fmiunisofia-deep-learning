from __future__ import annotations

import math
from numbers import Real
from typing import Callable, Iterable, List, Set, Tuple


class Value:
    def __init__(
        self,
        data: float,
        _children: Tuple["Value", ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = float(data)
        self._prev: Set["Value"] = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    @staticmethod
    def _coerce(other: Real | "Value") -> "Value":
        if isinstance(other, Value):
            return other
        if isinstance(other, Real):
            return Value(float(other))
        return NotImplemented

    def __add__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented

        out = Value(self.data + other_value.data, (self, other_value), "+")

        def _backward() -> None:
            self.grad += out.grad
            other_value.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other: Real | "Value") -> "Value":
        return self + other

    def __mul__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented

        out = Value(self.data * other_value.data, (self, other_value), "*")

        def _backward() -> None:
            self.grad += other_value.data * out.grad
            other_value.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other: Real | "Value") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1

    def __sub__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self + (-other_value)

    def __rsub__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return other_value + (-self)

    def __truediv__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return self * (other_value**-1)

    def __rtruediv__(self, other: Real | "Value") -> "Value":
        other_value = self._coerce(other)
        if other_value is NotImplemented:
            return NotImplemented
        return other_value / self

    def __pow__(self, exponent: Real) -> "Value":
        if not isinstance(exponent, Real):
            return NotImplemented

        out = Value(self.data**float(exponent), (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += float(exponent) * (self.data ** (float(exponent) - 1.0)) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Value":
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Value":
        tanh_value = math.tanh(self.data)
        out = Value(tanh_value, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - tanh_value**2) * out.grad

        out._backward = _backward
        return out

    def backward(self) -> None:
        topo = top_sort([self])
        for node in topo:
            node.grad = 0.0

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


def trace(root: Value) -> Tuple[Set[Value], Set[Tuple[Value, Value]]]:
    nodes: Set[Value] = set()
    edges: Set[Tuple[Value, Value]] = set()

    def build(node: Value) -> None:
        if node in nodes:
            return

        nodes.add(node)
        for child in node._prev:
            edges.add((child, node))
            build(child)

    build(root)
    return nodes, edges


def top_sort(values: Iterable[Value]) -> List[Value]:
    visited: Set[Value] = set()
    order: List[Value] = []

    def build(node: Value) -> None:
        if node in visited:
            return

        visited.add(node)
        for child in node._prev:
            build(child)
        order.append(node)

    for value in values:
        build(value)

    return order


def draw_dot(root: Value, filename: str = "01_result", show_grad: bool = False):
    try:
        import graphviz
    except ModuleNotFoundError as exc:
        message = (
            "The 'graphviz' Python package is required for draw_dot(). "
            "Install it with: pip install graphviz"
        )
        raise ModuleNotFoundError(message) from exc

    dot = graphviz.Digraph(
        filename=filename,
        format="svg",
        graph_attr={"rankdir": "LR"},
    )

    nodes, edges = trace(root)
    for node in nodes:
        uid = str(id(node))

        parts = []
        if node.label:
            parts.append(node.label)

        if show_grad:
            parts.append(f"data: {node.data}")
            parts.append(f"grad: {node.grad}")
        else:
            parts.append(f"data: {node.data}")

        dot.node(name=uid, label="{ " + " | ".join(parts) + " }", shape="record")

        if node._op:
            dot.node(name=uid + node._op, label=node._op)
            dot.edge(uid + node._op, uid)

    for left, right in edges:
        dot.edge(str(id(left)), str(id(right)) + right._op)

    return dot

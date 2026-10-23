"""Native MathML equations with literal labels and navigable field identities."""

from __future__ import annotations

import re
from html import unescape
from html.entities import html5
from typing import Any, Mapping
from xml.dom.minidom import Document, Text

from sympy import Basic, Eq, S, Symbol
from sympy.printing.conventions import requires_partial
from sympy.printing.mathml import MathMLPresentationPrinter

from modello.vectors import Vector, _VectorElement, _VectorFormula


class EquationPrinter(MathMLPresentationPrinter):
    """Print symbols without substitution, which could change the algebra."""

    def __init__(
        self,
        labels: Mapping[Symbol, tuple[str, str]],
        vectors: Mapping[Vector, tuple[str, str]] | None = None,
        coordinates: Mapping[Symbol, tuple[str, str, int]] | None = None,
    ) -> None:
        super().__init__({"mul_symbol": "dot"})
        # SymPy installs unescaped text nodes for its encoded operator glyphs.
        # Decode those references and serialize normal XML text instead, so
        # comparison signs and literal function names remain well-formed.
        self.dom.createTextNode = lambda text: Document.createTextNode(self.dom, unescape(text))
        self.labels = labels
        self.vectors = vectors or {}
        self.coordinates = coordinates or {}

    def _node(self, tag: str, *children: Any, text: str | None = None) -> Any:
        node = self.dom.createElement(tag)
        if text is not None:
            literal = Text()
            literal.data, literal.ownerDocument = text, self.dom
            node.appendChild(literal)
        for child in children:
            node.appendChild(child)
        return node

    def _link(self, node: Any, path: str) -> Any:
        if path:
            node.setAttribute("data-path", path)
            node.setAttribute("tabindex", "0")
            node.setAttribute("role", "button")
            node.setAttribute("aria-label", f"Inspect {path}")
        return node

    def _print_Symbol(self, expression: Symbol) -> Any:
        path, label = self.labels.get(expression, ("", expression.name))
        if expression in self.coordinates:
            path, label, index = self.coordinates[expression]
            node = self._node("msub", self._node("mi", text=label), self._node("mn", text=str(index)))
        else:
            node = self._node("mi", text=label)
        return self._link(node, path)

    _print_Dummy = _print_Symbol
    _print_InstanceDummy = _print_Symbol
    _print_BoundInstanceDummy = _print_Symbol
    _print_Index = _print_Symbol

    def _print_Vector(self, expression: Vector) -> Any:
        if expression in self.vectors:
            path, label = self.vectors[expression]
            return self._vector_symbol(path, label)
        return self.vector_values(expression)

    def _vector_symbol(self, path: str, label: str) -> Any:
        arrow = self._node("mo", text="→")
        arrow.setAttribute("stretchy", "true")
        result = self._node("mover", self._node("mi", text=label), arrow)
        result.setAttribute("accent", "true")
        return self._link(result, path)

    def vector_symbol(self, path: str, label: str) -> str:
        return self._markup(self._vector_symbol(path, label).toxml())

    def vector_values(self, expression: Vector) -> Any:
        """Display values as a column, independently of their vector identity."""
        table = self._node("mtable", *(self._node("mtr", self._node("mtd", self._print(v))) for v in expression))
        return self._node("mrow", self._node("mo", text="["), table, self._node("mo", text="]"))

    def _print__VectorElement(self, expression: _VectorElement) -> Any:
        if expression.vector in self.vectors:
            path, label = self.vectors[expression.vector]
            return self._link(self._node("msub", self._node("mi", text=label), self._print(expression.index)), path)
        if isinstance(expression.vector, _VectorFormula):
            function = expression.vector.function
            variable, first, _ = expression.vector.limits
            selected = function.subs(variable, first + expression.index)
            body = self._print(selected)
            if selected.is_Add:
                return self._node("mrow", self._node("mo", text="("), body, self._node("mo", text=")"))
            return body
        return self._node("msub", self._print(expression.vector), self._print(expression.index))

    def _print__VectorFormula(self, expression: _VectorFormula) -> Any:
        i, first, last = expression.limits
        return self._node(
            "mrow",
            self._node("mo", text="["),
            self._print(expression.function),
            self._iteration(i, first, last),
            self._node("mo", text="]"),
        )

    def _iteration(self, variable: Basic, first: Basic, last: Basic) -> Any:
        # Whitespace at mtext edges is normalized by some MathML renderers.
        before, after = self._node("mspace"), self._node("mspace")
        before.setAttribute("width", "0.8em")
        after.setAttribute("width", "0.3em")
        return self._node(
            "mrow",
            before,
            self._node("mtext", text="for"),
            after,
            self._print(variable),
            self._node("mo", text="∈"),
            self._range(first, last),
        )

    def _range(self, first: Basic, last: Basic) -> Any:
        if last < first:
            return self._node("mo", text="∅")
        if last - first <= 2:
            entries: list[Any] = []
            for value in range(int(first), int(last) + 1):
                if entries:
                    entries.append(self._node("mo", text=","))
                entries.append(self._node("mn", text=str(value)))
            return self._node("mrow", self._node("mo", text="{"), *entries, self._node("mo", text="}"))
        return self._node(
            "mrow",
            self._node("mo", text="{"),
            self._print(first),
            self._node("mo", text=","),
            self._node("mo", text="…"),
            self._node("mo", text=","),
            self._print(last),
            self._node("mo", text="}"),
        )

    def _discrete_operator(self, expression: Any, glyph: str) -> Any:
        operators = []
        for variable, lower, upper in reversed(expression.limits):
            symbol = self._node("mo", text=glyph)
            symbol.setAttribute("largeop", "true")
            lower_row = self._node("mrow", self._print(variable), self._node("mo", text="="), self._print(lower))
            operators.append(self._node("munderover", symbol, lower_row, self._print(upper)))
        # Parentheses make the complete summand or factor unambiguous.
        body = self._node(
            "mrow", self._node("mo", text="("), self._print(expression.function), self._node("mo", text=")")
        )
        return self._node("mrow", *operators, body)

    def _print_Sum(self, expression: Any) -> Any:
        return self._discrete_operator(expression, "∑")

    def _print_Product(self, expression: Any) -> Any:
        return self._discrete_operator(expression, "∏")

    def _print_Integral(self, expression: Any) -> Any:
        operators = []
        for limit in reversed(expression.limits):
            operator = self._node("mo", text="∫")
            operator.setAttribute("largeop", "true")
            if len(limit) == 3:
                operator = self._node("msubsup", operator, self._print(limit[1]), self._print(limit[2]))
            elif len(limit) == 2:
                operator = self._node("msup", operator, self._print(limit[1]))
            operators.append(operator)
        body = self._node(
            "mrow", self._node("mo", text="("), self._print(expression.function), self._node("mo", text=")")
        )
        differentials: list[Any] = []
        for limit in expression.limits:
            differential = self._node("mi", text="d")
            differential.setAttribute("mathvariant", "normal")
            differentials.extend((differential, self._print(limit[0])))
        return self._node("mrow", *operators, body, *differentials)

    def _print_Derivative(self, expression: Any) -> Any:
        glyph = "∂" if requires_partial(expression.expr) else "d"
        order = sum(count for _, count in expression.variable_count)
        differential = self._node("mo", text=glyph)
        if order != 1:
            differential = self._node("msup", differential, self._print(order))
        numerator = self._node("mrow", differential, self._print(expression.expr))
        factors: list[Any] = []
        for variable, count in expression.variable_count:
            symbol = self._print(variable)
            if count != 1:
                symbol = self._node("msup", symbol, self._print(count))
            factors.extend((self._node("mo", text=glyph), symbol))
        return self._node("mfrac", numerator, self._node("mrow", *factors))

    def _print_Limit(self, expression: Any) -> Any:
        function, variable, point, direction = expression.args
        target = self._print(point)
        if str(direction) in {"+", "-"} and point.is_finite is not False:
            target = self._node("msup", target, self._node("mo", text=str(direction)))
        approach = self._node("mrow", self._print(variable), self._node("mo", text="→"), target)
        operator = self._node("munder", self._node("mo", text="lim"), approach)
        return self._node(
            "mrow", operator, self._node("mo", text="("), self._print(function), self._node("mo", text=")")
        )

    def _print_Piecewise(self, expression: Any) -> Any:
        rows = []
        for value, condition in expression.args:
            if condition is S.true:
                qualifier = self._node("mtext", text="otherwise")
            else:
                space = self._node("mspace")
                space.setAttribute("width", "0.3em")
                qualifier = self._node("mrow", self._node("mtext", text="if"), space, self._print(condition))
            rows.append(self._node("mtr", self._node("mtd", self._print(value)), self._node("mtd", qualifier)))
        table = self._node("mtable", *rows)
        table.setAttribute("columnalign", "left left")
        table.setAttribute("columnspacing", "0.8em")
        brace = self._node("mo", text="{")
        brace.setAttribute("stretchy", "true")
        brace.setAttribute("fence", "true")
        return self._node("mrow", brace, table)

    def _print_IndexedBase(self, expression: Any) -> Any:
        label = self._print(expression.label)
        if expression.shape is not None and len(expression.shape) == 1:
            result = self._node("mover", label, self._node("mo", text="→"))
            result.setAttribute("accent", "true")
            return result
        return label

    def _print_Indexed(self, expression: Any) -> Any:
        # An indexed coordinate is scalar even when its base is a vector.
        base = expression.base.label if hasattr(expression.base, "label") else expression.base
        subscript = (
            self._print(expression.indices[0]) if len(expression.indices) == 1 else self._print(expression.indices)
        )
        return self._node("msub", self._print(base), subscript)

    def vector_equation(self, left: Vector, right: Vector, *, path: str = "", label: str = "") -> str:
        """Render a vector definition with its explicit iteration domain."""
        if isinstance(right, _VectorFormula) and len(right):
            variable, first, last = right.limits
            lhs = (
                self._link(self._node("msub", self._node("mi", text=label), self._print(variable - first)), path)
                if path
                else self._print__VectorElement(_VectorElement(left, variable - first))
            )
            row = self._node(
                "mrow",
                lhs,
                self._node("mo", text="="),
                self._print(right.function),
                self._iteration(variable, first, last),
            )
            return self._markup(row.toxml(), display=True)
        lhs = self._vector_symbol(path, label) if path else self._print(left)
        rhs = self._print(right) if len(right) else self.vector_values(Vector())
        return self._markup(self._node("mrow", lhs, self._node("mo", text="="), rhs).toxml(), display=True)

    def values(self, expression: Vector) -> str:
        return self._markup(self.vector_values(expression).toxml())

    def expression(self, expression: Basic, *, display: bool = False, parenthesize: bool = False) -> str:
        """Render a value or formula as standalone, XML-compatible MathML."""
        markup = self.doprint(expression)
        if parenthesize:
            markup = f"<mrow><mo>(</mo>{markup}<mo>)</mo></mrow>"
        return self._markup(markup, display=display)

    def _markup(self, markup: str, *, display: bool = False) -> str:
        markup = re.sub(r"&([A-Za-z][A-Za-z0-9]+);", _xml_entity, markup)
        mode = "block" if display else "inline"
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="{mode}">{markup}</math>'

    def equation(self, left: Basic, right: Basic) -> str:
        """Keep both original sides, including equations that could simplify."""
        return self.expression(Eq(left, right, evaluate=False), display=True)


def _xml_entity(match: re.Match[str]) -> str:
    if match[1] in {"amp", "lt", "gt", "quot", "apos"}:
        return match[0]
    return "".join(f"&#{ord(char)};" for char in html5[match[1] + ";"])

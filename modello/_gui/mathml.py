"""Native MathML equations with literal labels and navigable field identities."""

from __future__ import annotations

import re
from html.entities import html5
from typing import Any, Mapping
from xml.dom.minidom import Text

from sympy import Basic, Eq, Symbol
from sympy.printing.mathml import MathMLPresentationPrinter


class EquationPrinter(MathMLPresentationPrinter):
    """Print symbols without substitution, which could change the algebra."""

    def __init__(self, labels: Mapping[Symbol, tuple[str, str]]) -> None:
        super().__init__()
        self.labels = labels

    def _print_Symbol(self, expression: Symbol) -> Any:
        node = self.dom.createElement("mi")
        path, label = self.labels.get(expression, ("", expression.name))
        # SymPy replaces createTextNode with an unescaped writer. Display labels
        # require the standard DOM node, including on older Python versions.
        text = Text()
        text.data = label
        text.ownerDocument = self.dom
        node.appendChild(text)
        if path:
            node.setAttribute("data-path", path)
            node.setAttribute("tabindex", "0")
        return node

    _print_Dummy = _print_Symbol
    _print_InstanceDummy = _print_Symbol
    _print_BoundInstanceDummy = _print_Symbol

    def equation(self, left: Basic, right: Basic) -> str:
        """Return a standalone equation with XML-compatible entity references."""
        markup = self.doprint(Eq(left, right, evaluate=False))
        markup = re.sub(r"&([A-Za-z][A-Za-z0-9]+);", _xml_entity, markup)
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{markup}</math>'


def _xml_entity(match: re.Match[str]) -> str:
    if match[1] in {"amp", "lt", "gt", "quot", "apos"}:
        return match[0]
    return "".join(f"&#{ord(char)};" for char in html5[match[1] + ";"])

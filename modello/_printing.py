"""LaTeX export with the same scoped field identities as text inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from sympy import Basic, Derivative, Limit, Symbol
from sympy.printing.latex import LatexPrinter

from modello._symbols import Index
from modello.vectors import Vector

if TYPE_CHECKING:
    from modello._problem import _FieldPrinter


def _label(text: str) -> str:
    """Render literal paths, with numeric coordinate suffixes as subscripts."""
    base, bracket, index = text.rpartition("[")
    if bracket and index.endswith("]") and index[:-1].isdecimal():
        return rf"{_label(base)}_{{{index[:-1]}}}"
    if len(text) == 1 and text.isalpha():
        return str(LatexPrinter().doprint(Symbol(text)))
    escapes = {
        "\\": r"\backslash ",
        "{": r"\{",
        "}": r"\}",
        "%": r"\%",
        "&": r"\&",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "^": r"\hat{}",
        "~": r"\sim ",
        " ": r"\ ",
    }
    return r"\mathrm{" + "".join(escapes.get(character, character) for character in text) + "}"


class _FormulaLatexPrinter(LatexPrinter):
    def __init__(
        self,
        text: _FieldPrinter,
        vectors: Mapping[Vector, str],
        symbol_names: Mapping[Symbol, str],
        vector_names: Mapping[Vector, str],
    ) -> None:
        super().__init__({"mul_symbol": "dot", "symbol_names": dict(symbol_names)})
        self.labels = {symbol: _label(label) for symbol, label in text.labels.items()}
        self.vector_labels = {vector: _label(label) for vector, label in vectors.items()}
        coordinate_names: dict[Symbol, str] = {}
        for vector, name in vector_names.items():
            # A family's public and chart vectors share one display path.
            path = vectors.get(vector)
            aliases = [other for other, alias in vectors.items() if path is not None and alias == path]
            for values in dict.fromkeys((vector, *aliases)):
                self.vector_labels[values] = name
                for index, symbol in enumerate(values):
                    if isinstance(symbol, Symbol):
                        coordinate_names.setdefault(symbol, rf"{name}_{{{index}}}")
        self.labels.update(coordinate_names)
        self.labels.update(symbol_names)
        self._field_symbols = text._field_symbols
        self._bound_symbols: frozenset[Symbol] = frozenset()

    def _print(self, expression: Any, **kwargs: Any) -> str:
        previous = self._bound_symbols
        self._bound_symbols |= frozenset(getattr(expression, "bound_symbols", ()))
        if isinstance(expression, Derivative):
            self._bound_symbols |= frozenset(expression.variables)
        elif isinstance(expression, Limit):
            self._bound_symbols |= {expression.args[1]}
        try:
            return str(super()._print(expression, **kwargs))
        finally:
            self._bound_symbols = previous

    def _print_Symbol(self, expression: Symbol, style: str = "plain") -> str:
        if expression in self._bound_symbols and expression not in self._field_symbols:
            return str(super()._print_Symbol(expression, style=style))
        return self.labels.get(expression, str(super()._print_Symbol(expression, style=style)))

    _print_Dummy = _print_Symbol

    def _print_Index(self, expression: Index) -> str:
        return str(super()._print_Symbol(expression))


def _latex(
    expression: Basic,
    text: _FieldPrinter,
    vectors: Mapping[Vector, str],
    symbol_names: Mapping[Symbol, str] | None,
    vector_names: Mapping[Vector, str] | None,
) -> str:
    return str(_FormulaLatexPrinter(text, vectors, symbol_names or {}, vector_names or {}).doprint(expression))

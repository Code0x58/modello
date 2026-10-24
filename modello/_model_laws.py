"""Indexed model equations, kept separate from their finite compilation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from sympy import Basic, Expr, Symbol, Tuple

from modello._collections import _Location
from modello._operators import _DeclarationSymbol, _SymbolicTemplate
from modello._paths import FieldPath
from modello.vectors import Vector


class ModelRecurrence(_SymbolicTemplate):
    """The scoped equations returned by an indexed ``model_recurrence`` declaration.

    Declare a law over ``states[n]`` with named field expressions. Members before
    ``start`` are boundary unknowns; observations anywhere in the finite window
    participate in the joint solve. This template retains the source equations.
    """

    def __new__(
        cls, equations: Tuple, limits: Tuple, _declaration: _DeclarationSymbol | None = None
    ) -> ModelRecurrence:
        declaration = _DeclarationSymbol("model_recurrence") if _declaration is None else _declaration
        return Basic.__new__(cls, equations, limits, declaration)

    @property
    def equations(self) -> tuple[Basic, ...]:
        return tuple(self.args[0])

    @property
    def index(self) -> Symbol:
        return self.args[1][0]

    @property
    def start(self) -> Basic:
        return self.args[1][1]

    @property
    def stop(self) -> Basic:
        return self.args[1][2]

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return (self.index,)

    @property
    def _binding_signature(self) -> tuple[type, tuple[bool, ...]]:
        return type(self), (True,)

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset({0})

    @property
    def free_symbols(self) -> set[Symbol]:
        return (self.args[0].free_symbols | Tuple(self.start, self.stop).free_symbols) - {self.index}

    def _sympystr(self, printer: Any) -> str:
        equations = "; ".join(printer.doprint(eq) for eq in self.equations)
        first, last = printer.doprint(self.start), printer.doprint(self.stop)
        return f"{equations} for {printer.doprint(self.index)} in {{{first}, ..., {last}}}"

    def _latex(self, printer: Any) -> str:
        equations = r" \\ ".join(printer._print(eq) for eq in self.equations)
        domain = (
            rf"{printer._print(self.index)}\in\{{{printer._print(self.start)},\ldots,{printer._print(self.stop)}\}}"
        )
        return rf"\left\{{\begin{{aligned}}{equations}\end{{aligned}}\right.\quad {domain}"


@dataclass(frozen=True)
class _ModelLawSpec:
    target: _Location
    columns: Mapping[FieldPath, Vector]
    template: ModelRecurrence
    updates: tuple[tuple[FieldPath, Expr], ...]
    compile: Callable[[Any, _ModelLawSpec], tuple[tuple[FieldPath, Symbol, Basic], ...]]

    def substitute(self, bindings: Mapping[Basic, Basic]) -> _ModelLawSpec:
        """Rebind updates while preserving distinct source columns, even at N=0."""
        scalar_bindings = {key: value for key, value in bindings.items() if not isinstance(key, Vector)}
        return replace(
            self,
            target=self.target.substitute(bindings),
            columns={path: vector.xreplace(bindings) for path, vector in self.columns.items()},
            template=self.template.xreplace(scalar_bindings),
            updates=tuple((path, expression.xreplace(bindings)) for path, expression in self.updates),
        )

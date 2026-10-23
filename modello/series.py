"""Horizon-independent indexed quantities and causal recurrence laws."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import cached_property
from typing import Any

from sympy import (
    And,
    Basic,
    Dummy,
    Eq,
    Expr,
    Function,
    Integer,
    Piecewise,
    S,
    Symbol,
    Tuple,
    diff,
    rsolve,
    simplify,
    sympify,
)

from modello._operators import _FunctionSymbol, _ScopedOperator, _SymbolicTemplate, _value_condition
from modello._symbols import Index, _minimal_assumptions
from modello._values import ScalarValue, SymbolicScalar
from modello.errors import ModelInputError
from modello.vectors import Vector, _invalid_index, _VectorElement, vector

__all__ = ["Recurrence", "Series"]


class Series(_SymbolicTemplate):
    """A symbolic scalar family with no chosen sample window.

    ``Series('T', n, real=True)[n-1]`` is a relative term, not Python negative
    indexing. An integer coordinate denotes a discrete series; a real coordinate
    can denote a continuous process. Finite observations remain ordinary model
    fields or vectors, and a ``Recurrence`` supplies a discrete evolution law.
    """

    def __new__(cls, name: str | _FunctionSymbol, index: Symbol | None = None, **assumptions: Any) -> Series:
        if index is None:
            index = Index("n")
        if not isinstance(index, Symbol) or index.is_real is False:
            raise ModelInputError("A series needs a real or integer symbolic coordinate")
        if isinstance(name, _FunctionSymbol):
            symbol = name
        elif isinstance(name, str) and name:
            symbol = _FunctionSymbol(name, **assumptions)
        else:
            raise ModelInputError("A series needs a nonempty name")
        if symbol.is_commutative is not True:
            raise ModelInputError("Series currently represent commutative scalar quantities")
        return Basic.__new__(cls, symbol, index)

    @property
    def symbol(self) -> _FunctionSymbol:
        return self.args[0]

    @property
    def index(self) -> Symbol:
        return self.args[1]

    @property
    def discrete(self) -> bool:
        return self.index.is_integer is True

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return (self.index,)

    @property
    def free_symbols(self) -> set[Symbol]:
        return set()

    @cached_property
    def _requirements(self) -> dict[str, bool]:
        return _minimal_assumptions(self.symbol)

    def __getitem__(self, index: ScalarValue) -> Expr:
        return _SeriesTerm(self, index)

    def _sympystr(self, printer: Any) -> str:
        return str(getattr(printer, "template_labels", {}).get(self, self.symbol.name))

    def _latex(self, printer: Any) -> str:
        return getattr(printer, "template_labels", {}).get(self, printer._print(Symbol(self.symbol.name)))


class _SeriesTerm(_ScopedOperator):
    _diff_wrt = True
    is_commutative = True

    def __new__(cls, series: Series, index: ScalarValue) -> _SeriesTerm:
        try:
            point = sympify(index)
        except (TypeError, ValueError) as error:
            raise ModelInputError("A series coordinate must be scalar") from error
        if not isinstance(series, Series) or not isinstance(point, Expr) or point.is_Matrix:
            raise ModelInputError("A series coordinate must be scalar")
        if series.discrete and _invalid_index(point):
            raise ModelInputError("A discrete series coordinate must be an integer")
        if not series.discrete and point.is_real is False:
            raise ModelInputError("A continuous series coordinate must be real")
        return Expr.__new__(cls, series, point)

    @property
    def series(self) -> Series:
        return self.args[0]

    @property
    def index(self) -> Expr:
        return self.args[1]

    @property
    def free_symbols(self) -> set[Symbol]:
        return self.index.free_symbols

    @property
    def assumptions0(self) -> dict[str, bool]:
        return self.series.symbol.assumptions0

    def _eval_is_real(self) -> bool | None:
        valid = self.index.is_integer is True if self.series.discrete else self.index.is_real is True
        return True if valid and self.series.symbol.is_real is True else None

    def _sympystr(self, printer: Any) -> str:
        return f"{printer.doprint(self.series)}[{printer.doprint(self.index)}]"

    def _latex(self, printer: Any) -> str:
        return rf"{printer._print(self.series)}_{{{printer._print(self.index)}}}"


class Recurrence(_SymbolicTemplate):
    """A causal relative law and an explicit contiguous initial history.

    ``Recurrence(T[n], a*T[n-1], initial={0: opening})`` holds from ``n=1``.
    ``.at(k)`` is a scalar formula; ``.window(first, last)`` is an inclusive
    finite vector view. Neither view changes the law's coordinate system.
    """

    def __new__(
        cls,
        target: Expr,
        expression: ScalarValue,
        initial: Mapping[int, ScalarValue] | Tuple,
        start: int | Integer | None = None,
    ) -> Recurrence:
        body = sympify(expression)
        if not isinstance(target, _SeriesTerm) or not target.series.discrete or not isinstance(target.index, Symbol):
            raise ModelInputError("Use Recurrence(T[n], expression, initial={0: value}) with an integer index")
        if not isinstance(body, Expr) or body.is_Matrix:
            raise ModelInputError("A recurrence expression must be scalar")
        entries = (
            Tuple(*(Tuple(sympify(key), sympify(value)) for key, value in initial.items()))
            if isinstance(initial, Mapping)
            else initial
        )
        if (
            not isinstance(entries, Tuple)
            or not entries
            or any(
                not isinstance(pair, Tuple)
                or len(pair) != 2
                or not isinstance(pair[0], Integer)
                or not isinstance(pair[1], Expr)
                or pair[1].is_Matrix
                or target.index in pair[1].free_symbols
                for pair in entries
            )
        ):
            raise ModelInputError(
                "Initial history needs integer positions and scalar values independent of the local index"
            )
        entries = Tuple(*sorted(entries, key=lambda pair: pair[0]))
        if any(term.series == target.series for _, value in entries for term in value.atoms(_SeriesTerm)):
            raise ModelInputError("Initial history must use scalar boundary quantities, not the same series")
        first, last = entries[0][0], entries[-1][0]
        start = last + 1 if start is None else sympify(start)
        if (
            not isinstance(start, Integer)
            or start != last + 1
            or tuple(pair[0] for pair in entries) != tuple(range(int(first), int(start)))
        ):
            raise ModelInputError(
                "Initial history must contain every position immediately before the recurrence starts"
            )
        for term in body.atoms(_SeriesTerm):
            if term.series == target.series:
                delay = simplify(target.index - term.index)
                if not isinstance(delay, Integer) or not 1 <= delay <= len(entries):
                    raise ModelInputError("A causal recurrence uses fixed positive lags covered by its initial history")
        return Basic.__new__(cls, target, body, entries, start)

    @property
    def target(self) -> _SeriesTerm:
        return self.args[0]

    @property
    def function(self) -> Expr:
        return self.args[1]

    @property
    def initial(self) -> Mapping[int, Expr]:
        from types import MappingProxyType

        return MappingProxyType({int(point): value for point, value in self.args[2]})

    @property
    def start(self) -> Integer:
        return self.args[3]

    @property
    def origin(self) -> int:
        return int(self.args[2][0][0])

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return (self.target.index,)

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset(range(4))

    @property
    def free_symbols(self) -> set[Symbol]:
        return (self.function.free_symbols | self.args[2].free_symbols) - {self.target.index}

    @property
    def equations(self) -> tuple[Basic, ...]:
        return (
            Eq(self.target, self.function, evaluate=False),
            *(Eq(self.target.series[point], value, evaluate=False) for point, value in self.initial.items()),
        )

    def at(self, index: ScalarValue) -> Expr:
        return _RecurrenceValue(self, index)

    def window(self, first: int | SymbolicScalar, last: int | SymbolicScalar) -> Vector:
        i = Index(self.target.index.name)
        return vector(self.at(i), (i, first, last))

    @cached_property
    def _terms(self) -> dict[int, Expr]:
        return dict(self.initial)

    @cached_property
    def _conditions(self) -> dict[int, Basic]:
        condition = And(*(_series_condition(self.target.series, value) for value in self.initial.values()))
        return {point: condition for point in self.initial}

    @cached_property
    def _invariant(self) -> bool:
        """Prove coordinate requirements by induction when scalar assumptions suffice."""
        if any(_series_condition(self.target.series, value) is not S.true for value in self.initial.values()):
            return False
        body = self.function.xreplace(
            {
                term: Dummy("previous", **self.target.series.symbol.assumptions0)
                for term in self.function.atoms(_SeriesTerm)
                if term.series == self.target.series
            }
        )
        position = Index("position", nonnegative=True)
        return _series_condition(self.target.series, body.subs(self.target.index, position + self.start)) is S.true

    def _first_order(self) -> Expr | None:
        """A universal affine formula, retaining the unit-coefficient branch."""
        if len(self.initial) != 1 or self.function.has(_VectorElement):
            return None
        n = self.target.index
        previous = Dummy("previous")
        body = self.function.xreplace({self.target.series[n - 1]: previous})
        if body.has(_SeriesTerm):
            return None
        a = diff(body, previous)
        b = simplify(body - a * previous)
        if (a.free_symbols | b.free_symbols) & {n, previous}:
            return None
        opening = self.initial[self.origin]
        count = n - self.origin
        if b == 0:
            return opening * a**count
        return Piecewise((opening + b * count, Eq(a, 1)), (opening * a**count + b * (a**count - 1) / (a - 1), True))

    def _at(self, point: int) -> Expr:
        if point < self.origin:
            return S.NaN
        conditions = self._conditions
        for position in range(max(self._terms) + 1, point + 1):
            expression = self.function.subs(self.target.index, position)
            replacements = {
                term: self._terms[int(term.index)]
                for term in expression.atoms(_SeriesTerm)
                if term.series == self.target.series
            }
            self._terms[position] = expression.xreplace(replacements)
            conditions[position] = And(
                conditions[position - 1], _series_condition(self.target.series, self._terms[position])
            )
        return Piecewise((self._terms[point], conditions[point]), (S.NaN, True))

    @cached_property
    def _closed_form(self) -> Expr | None:
        n = self.target.index
        result = self._first_order()
        if result is None:
            # Generic higher-order coefficients can conceal singular cases.
            if self.function.free_symbols - {n} or self.function.has(_VectorElement):
                return None
            if any(term.series != self.target.series for term in self.function.atoms(_SeriesTerm)):
                return None
            position = Index("position", nonnegative=True)
            denominator = self.function.as_numer_denom()[1]
            if denominator.subs(n, position + self.start).is_zero is not False:
                return None
            function = Function(f"_recurrence_{self.target.series.symbol.dummy_index}")
            native = self.function.replace(
                lambda term: isinstance(term, _SeriesTerm), lambda term: function(term.index)
            )
            try:
                result = rsolve(function(n) - native, function(n), dict(self.initial))
            except (NotImplementedError, ValueError):
                return None
        if result is None or result.free_symbols - ({n} | self.args[2].free_symbols | self.function.free_symbols):
            return None
        position = Index("position", nonnegative=True)
        if (
            not self._invariant
            and _series_condition(self.target.series, result.subs(n, position + self.origin)) is not S.true
        ):
            return None
        residual = self.function.replace(
            lambda term: isinstance(term, _SeriesTerm), lambda term: result.subs(n, term.index)
        )
        if simplify((result - residual).subs(n, position + self.start)) != 0 or any(
            simplify(result.subs(n, point) - value) != 0 for point, value in self.initial.items()
        ):
            return None
        return result

    def _sympystr(self, printer: Any) -> str:
        return (
            f"{printer.doprint(self.target)} = {printer.doprint(self.function)} "
            f"for {printer.doprint(self.target.index)} >= {self.start}"
        )

    def _latex(self, printer: Any) -> str:
        rows = [
            rf"{printer._print(self.target)} = {printer._print(self.function)}, & "
            rf"{printer._print(self.target.index)}\ge {self.start}"
        ]
        rows.extend(
            rf"{printer._print(self.target.series[point])} = {printer._print(value)}"
            for point, value in self.initial.items()
        )
        return r"\begin{cases}" + r"\\ ".join(rows) + r"\end{cases}"


class _RecurrenceValue(_ScopedOperator):
    is_commutative = True

    def __new__(cls, recurrence: Recurrence, index: ScalarValue) -> _RecurrenceValue:
        point = sympify(index)
        if not isinstance(point, Expr) or _invalid_index(point):
            raise ModelInputError("A recurrence position must be an integer")
        return Expr.__new__(cls, recurrence, point)

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return self.args[0].bound_symbols

    @property
    def assumptions0(self) -> dict[str, bool]:
        return self.args[0].target.series.symbol.assumptions0

    def doit(self, **hints: Any) -> Expr:
        recurrence, point = self.args
        if isinstance(point, Integer):
            if point < recurrence.origin:
                return S.NaN
            if point >= recurrence.start and recurrence._closed_form is not None:
                return simplify(recurrence._closed_form.subs(recurrence.target.index, point))
            return recurrence._at(int(point))
        result = recurrence._closed_form
        if result is None:
            return self
        value = result.subs(recurrence.target.index, point)
        return Piecewise((value, point >= recurrence.origin), (S.NaN, True))

    def _sympystr(self, printer: Any) -> str:
        return printer.doprint(self.args[0].target.series[self.args[1]])

    def _latex(self, printer: Any) -> str:
        return printer._print(self.args[0].target.series[self.args[1]])


def _series_condition(series: Series, value: Expr) -> Basic:
    """Retain requirements on the full evaluated history, including hidden lags."""
    return _value_condition(series._requirements, value)


def _project_series(expression: Basic, templates: Iterable[_SymbolicTemplate]) -> Basic:
    """Interpret point queries through a declared law, leaving laws themselves intact."""
    if not expression.has(_SeriesTerm):
        return expression
    laws: dict[Series, Recurrence] = {}
    ambiguous: set[Series] = set()
    for template in templates:
        if isinstance(template, Recurrence):
            series = template.target.series
            if series in laws and laws[series] != template:
                ambiguous.add(series)
            laws[series] = template

    def project(value: Basic) -> Basic:
        if isinstance(value, _SymbolicTemplate):
            return value
        if isinstance(value, _SeriesTerm) and value.series in laws:
            if value.series in ambiguous:
                raise ModelInputError("The series has multiple recurrence laws; select one law explicitly with .at()")
            return laws[value.series].at(value.index)
        args = tuple(project(arg) for arg in value.args)
        return value.func(*args) if args != value.args else value

    return project(expression)

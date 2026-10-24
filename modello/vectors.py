"""Finite symbolic vectors and indexed formulas built from SymPy scalars.

Vector coordinates participate in the ordinary joint scalar solve. Formulas
retain their indices until evaluation, so inspection can show the mathematics
that was declared instead of an expanded list of scalar operations.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Any, TypeAlias, overload

from sympy import Basic, Expr, Integer, Product, S, Sum, Symbol, Tuple, simplify, sympify

from modello._operators import _FunctionSymbol
from modello._symbols import Index, InstanceDummy
from modello._values import ScalarValue, SymbolicScalar
from modello.errors import ModelInputError

__all__ = ["Vector", "VectorValue", "vector"]


class Vector(Basic):
    """An immutable, ordered, finite vector of scalar expressions.

    ``Vector('x', size=3, real=True)`` declares three independent quantities.
    ``Vector(1, 2, 3)`` or ``Vector([1, 2, 3])`` constructs explicit values.
    Integer indexing and slicing follow Python; a symbolic index represents
    an indexed scalar. Arithmetic is elementwise; ``dot`` or ``@`` contracts
    two vectors. Shapes must agree and scalars are never broadcast by addition.
    """

    def __new__(
        cls, *values: ScalarValue | Sequence[ScalarValue], size: int | SymbolicScalar | None = None, **assumptions: Any
    ) -> Vector:
        if size is not None:
            if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
                raise ModelInputError("A declared vector needs a name and size")
            if isinstance(size, Expr) and not isinstance(size, Integer):
                if size.is_integer is not True or size.is_nonnegative is False:
                    raise ModelInputError("Vector size must be a nonnegative integer expression")
                return _VectorSymbol(_FunctionSymbol(values[0], **assumptions), size)
            if isinstance(size, bool) or not isinstance(size, (int, Integer)) or size < 0:
                raise ModelInputError("Vector size must be a nonnegative integer")
            entries = tuple(InstanceDummy(f"{values[0]}[{i}]", **assumptions) for i in range(size))
        else:
            if assumptions:
                raise ModelInputError("Assumptions apply to declared vectors; supply a name and size")
            if (
                len(values) == 1
                and isinstance(values[0], (Vector, Sequence, Tuple))
                and not isinstance(values[0], (str, bytes))
            ):
                values = tuple(values[0])
            try:
                entries = tuple(sympify(value) for value in values)
            except (TypeError, ValueError) as error:
                raise ModelInputError("Vector entries must be SymPy-compatible scalars") from error
            if any(not isinstance(value, Expr) or value.is_Matrix for value in entries):
                raise ModelInputError("Vector entries must be SymPy-compatible scalars")
        return Basic.__new__(cls, *entries)

    @property
    def elements(self) -> tuple[Expr, ...]:
        """Return the ordered scalar expressions, without evaluating formulas."""
        return self.args

    def __len__(self) -> int:
        return len(self.elements)

    @property
    def shape(self) -> tuple[int | Expr]:
        """Return the single finite axis, including ``(0,)`` for an empty vector."""
        return (len(self),)

    @property
    def _finite(self) -> bool:
        return isinstance(self.shape[0], (int, Integer))

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.elements)

    def doit(self, **hints: Any) -> Vector:
        """Materialize a comprehension and evaluate its scalar entries."""
        return Vector(*(entry.doit(**hints) for entry in self))

    @overload
    def __getitem__(self, index: slice) -> Vector: ...

    @overload
    def __getitem__(self, index: object) -> Expr: ...

    def __getitem__(self, index: object) -> Expr | Vector:
        if isinstance(index, slice):
            return Vector(*(self._at(i) for i in range(len(self))[index]))
        if isinstance(index, bool):
            raise ModelInputError("A vector index must be an integer")
        try:
            position = sympify(index)
        except (TypeError, ValueError) as error:
            raise ModelInputError("A vector index must be an integer") from error
        if not isinstance(position, Expr) or _invalid_index(position):
            raise ModelInputError("A vector index must be an integer")
        if isinstance(position, Integer):
            return self._at(int(position))
        return _VectorElement(self, position)

    def _at(self, position: int) -> Expr:
        return self.elements[position]

    @cached_property
    def _coordinate_symbols(self) -> frozenset[Symbol]:
        return frozenset().union(*(entry.free_symbols for entry in self))

    def sum(self, index: Symbol | None = None) -> Expr:
        """Return an unevaluated SymPy sum over zero-based coordinates."""
        i = _local_index(self) if index is None else index
        return Sum(self[i], (i, 0, self.shape[0] - 1))

    def product(self, index: Symbol | None = None) -> Expr:
        """Return an unevaluated SymPy product over zero-based coordinates."""
        i = _local_index(self) if index is None else index
        return Product(self[i], (i, 0, self.shape[0] - 1))

    def dot(self, other: Vector, index: Symbol | None = None) -> Expr:
        """Return the bilinear dot product as a SymPy sum (no conjugation)."""
        self._same_shape(other)
        i = _local_index(self, other) if index is None else index
        return Sum(self[i] * other[i], (i, 0, self.shape[0] - 1))

    def __matmul__(self, other: Vector) -> Expr:
        return self.dot(other)

    def _same_shape(self, other: object) -> None:
        if not isinstance(other, Vector) or simplify(self.shape[0] - other.shape[0]) != 0:
            raise ModelInputError("Vector operations require vectors of the same size")

    def __add__(self, other: Vector) -> Vector:
        self._same_shape(other)
        i = _local_index(self, other)
        return vector(self[i] + other[i], (i, 0, self.shape[0] - 1))

    def __sub__(self, other: Vector) -> Vector:
        self._same_shape(other)
        i = _local_index(self, other)
        return vector(self[i] - other[i], (i, 0, self.shape[0] - 1))

    def __neg__(self) -> Vector:
        return self * -1

    def __mul__(self, scalar: ScalarValue) -> Vector:
        return self._scale(_factor(scalar))

    def _scale(self, value: Expr, *, left: bool = False) -> Vector:
        i = _local_index(self)
        return vector(value * self[i] if left else self[i] * value, (i, 0, self.shape[0] - 1))

    def __rmul__(self, scalar: ScalarValue) -> Vector:
        return self._scale(_factor(scalar), left=True)

    def __truediv__(self, scalar: ScalarValue) -> Vector:
        return self._scale(1 / _factor(scalar))

    def _sympystr(self, printer: Any) -> str:
        label = getattr(printer, "vector_labels", {}).get(self)
        if label is not None:
            return label
        return "[" + ", ".join(printer.doprint(value) for value in self) + "]"

    def _latex(self, printer: Any) -> str:
        label = getattr(printer, "vector_labels", {}).get(self)
        if label is not None:
            return rf"\vec{{{label}}}"
        if not len(self):
            return r"\left[\right]"
        return r"\begin{bmatrix}" + r"\\ ".join(printer._print(value) for value in self) + r"\end{bmatrix}"


class _VectorCoordinate(InstanceDummy):
    """A reproducible coordinate identity within one symbolic vector declaration."""


class _VectorSymbol(Vector):
    """Independent coordinates whose finite count is determined by observations."""

    def __new__(cls, symbol: _FunctionSymbol, size: Expr) -> Vector:
        if isinstance(size, Integer):
            if size < 0:
                raise ModelInputError("Vector size must be a nonnegative integer")
            return Vector(
                *(
                    _VectorCoordinate(f"{symbol.name}[{i}]", dummy_index=symbol.dummy_index, **symbol.assumptions0)
                    for i in range(size)
                )
            )
        return Basic.__new__(cls, symbol, size)

    @property
    def shape(self) -> tuple[Expr]:
        return (self.args[1],)

    @property
    def elements(self) -> tuple[Expr, ...]:
        raise TypeError("This vector has a symbolic length; supply observations or its size first")

    @property
    def free_symbols(self) -> set[Symbol]:
        return self.args[1].free_symbols

    @cached_property
    def _coordinate_symbols(self) -> frozenset[Symbol]:
        return frozenset(self.free_symbols)

    def _at(self, position: int) -> Expr:
        return Expr.__new__(_VectorElement, self, Integer(position))

    def _sympystr(self, printer: Any) -> str:
        return getattr(printer, "vector_labels", {}).get(self, self.args[0].name)

    def _latex(self, printer: Any) -> str:
        label = getattr(printer, "vector_labels", {}).get(self, printer._print(Symbol(self.args[0].name)))
        return rf"\vec{{{label}}}"


class _VectorElement(Expr):
    """A scalar selection whose symbols are its coordinates and its index."""

    def __new__(cls, values: Vector, index: Expr) -> Expr:
        if isinstance(index, Integer) and values._finite:
            try:
                return values[int(index)]
            except IndexError:
                return S.NaN
        if _invalid_index(index):
            return S.NaN
        return Expr.__new__(cls, values, index)

    @property
    def vector(self) -> Vector:
        return self.args[0]

    @property
    def index(self) -> Expr:
        return self.args[1]

    def _eval_is_real(self) -> bool | None:
        if not self.vector._finite:
            return self.vector.args[0].is_real if isinstance(self.vector, _VectorSymbol) else None
        valid = (
            self.index.is_integer is True
            and (self.index + len(self.vector)).is_nonnegative is True
            and (len(self.vector) - 1 - self.index).is_nonnegative is True
        )
        return True if valid and all(value.is_real is True for value in self.vector) else None

    def _eval_is_commutative(self) -> bool | None:
        if not self.vector._finite:
            return self.vector.args[0].is_commutative if isinstance(self.vector, _VectorSymbol) else True
        assumptions = {value.is_commutative for value in self.vector}
        return next(iter(assumptions)) if len(assumptions) == 1 else None

    def _eval_subs(self, old: Basic, new: Basic) -> Expr | None:
        # A local index normally does not occur in the coordinates themselves.
        # Avoid visiting the full vector for each term of a finite reduction.
        if isinstance(old, Symbol) and old not in self.vector._coordinate_symbols:
            return _VectorElement(self.vector, self.index.subs(old, new))
        return None

    def _sympystr(self, printer: Any) -> str:
        return f"{printer.doprint(self.vector)}[{printer.doprint(self.index)}]"

    def _latex(self, printer: Any) -> str:
        label = getattr(printer, "vector_labels", {}).get(self.vector)
        if label is not None:
            return rf"{label}_{{{printer._print(self.index)}}}"
        if isinstance(self.vector, _VectorFormula):
            variable, first, _ = self.vector.limits
            selected = self.vector.function.subs(variable, first + self.index)
            body = printer._print(selected)
            return rf"\left({body}\right)" if selected.is_Add else str(body)
        return rf"{{{printer._print(self.vector)}}}_{{{printer._print(self.index)}}}"


class _VectorFormula(Vector):
    """A finite comprehension, retaining its binder independently of its values."""

    def __new__(
        cls,
        expression: Expr,
        limits: tuple[Symbol, int | Expr, int | Expr] | Tuple,
        _integer_bounds: Basic | None = None,
    ) -> _VectorFormula:
        try:
            expression = sympify(expression)
            limits = Tuple(*limits)
        except (TypeError, ValueError) as error:
            raise ModelInputError("Use vector(expression, (index, first, last))") from error
        if (
            not isinstance(expression, Expr)
            or expression.is_Matrix
            or len(limits) != 3
            or not isinstance(limits[0], Symbol)
        ):
            raise ModelInputError("Use vector(expression, (index, first, last))")
        # Native alpha-renaming temporarily removes coordinate assumptions.
        # Preserve the original integer-bound premise in canonical arguments,
        # while rejecting replacements that are provably noninteger.
        if (
            _integer_bounds not in (None, S.true)
            or any(
                not isinstance(bound, Expr)
                or _invalid_index(bound)
                or (_integer_bounds is None and bound.is_integer is not True)
                for bound in limits[1:]
            )
            or (limits[2] - limits[1] + 1).is_negative
        ):
            raise ModelInputError("Vector formula bounds must describe a finite increasing integer range")
        return Basic.__new__(cls, expression, limits, S.true)

    @property
    def function(self) -> Expr:
        return self.args[0]

    @property
    def limits(self) -> Tuple:
        return self.args[1]

    @property
    def bound_symbols(self) -> list[Symbol]:
        return [self.limits[0]]

    @property
    def free_symbols(self) -> set[Symbol]:
        return (
            (self.function.free_symbols - {self.limits[0]}) | self.limits[1].free_symbols | self.limits[2].free_symbols
        )

    @property
    def shape(self) -> tuple[int | Expr]:
        count = self.limits[2] - self.limits[1] + 1
        return (int(count) if isinstance(count, Integer) else count,)

    @property
    def elements(self) -> tuple[Expr, ...]:
        return self._elements

    @cached_property
    def _elements(self) -> tuple[Expr, ...]:
        return tuple(self._at(position) for position in range(len(self)))

    def _at(self, position: int) -> Expr:
        if not self._finite:
            return Expr.__new__(_VectorElement, self, Integer(position))
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("Vector index out of range")
        variable, first, _ = self.limits
        return self.function.subs(variable, first + position)

    @cached_property
    def _coordinate_symbols(self) -> frozenset[Symbol]:
        return frozenset(self.free_symbols)

    def __len__(self) -> int:
        if not self._finite:
            raise TypeError("This vector has a symbolic length; supply observations or its size first")
        return int(self.shape[0])

    def _eval_subs(self, old: Basic, new: Basic) -> _VectorFormula:
        variable, first, last = self.limits
        body = self.function if old == variable else self.function.subs(old, new)
        return _VectorFormula(body, (variable, first.subs(old, new), last.subs(old, new)), self.args[2])

    def _sympystr(self, printer: Any) -> str:
        if hasattr(printer, "_print_VectorFormula"):
            return printer._print_VectorFormula(self)
        return f"vector({printer.doprint(self.function)}, {printer.doprint(self.limits)})"

    def _latex(self, printer: Any) -> str:
        if self in getattr(printer, "vector_labels", {}):
            return super()._latex(printer)
        variable, first, last = self.limits
        domain = (
            r"\varnothing"
            if self.shape[0] == 0
            else r"\{" + ", ".join(printer._print(first + offset) for offset in range(len(self))) + r"\}"
            if self._finite and len(self) <= 3
            else rf"\{{{printer._print(first)}, \ldots, {printer._print(last)}\}}"
        )
        return (
            rf"\left[{printer._print(self.function)}\;\middle|\;"
            rf"{printer._print(variable)}\in {domain}\right]"
        )


def vector(expression: ScalarValue, limits: tuple[Symbol, int | SymbolicScalar, int | SymbolicScalar]) -> Vector:
    """Build a finite indexed vector, including both endpoints of ``limits``.

    For example, ``vector(x[i] - x[i-1], (i, 1, len(x)-1))`` describes adjacent
    changes. The result is zero-based: its first entry corresponds to ``i=1``.
    Like SymPy Sum, this retains the original function and bound variable.
    """
    return _VectorFormula(expression, limits)


VectorValue: TypeAlias = Vector | Sequence[ScalarValue | None]


def _invalid_index(index: Expr) -> bool:
    return (
        bool(index.is_Matrix) or index.is_integer is False or (index.is_number is True and index.is_integer is not True)
    )


def _factor(scalar: ScalarValue) -> Expr:
    try:
        value = sympify(scalar)
    except (TypeError, ValueError) as error:
        raise ModelInputError("Scale a vector by a scalar; use dot for a contraction") from error
    if not isinstance(value, Expr) or value.is_Matrix:
        raise ModelInputError("Scale a vector by a scalar; use dot for a contraction")
    return value


def _local_index(*vectors: Vector) -> Index:
    """Choose a fresh display name as well as a fresh symbolic identity."""
    names = {symbol.name for values in vectors for symbol in values.atoms(Symbol)}
    for name in ("i", "j", "k", "l", "m"):
        if name not in names:
            return Index(name)
    suffix = 1
    while f"i{suffix}" in names:
        suffix += 1
    return Index(f"i{suffix}")


def _vector_input(value: object, size: int | None, path: str) -> tuple[object, ...]:
    """Read finite observations, optionally checking their resolved shape."""
    if not isinstance(value, (Vector, Sequence, Tuple)) or isinstance(value, (str, bytes)):
        count = "" if size is None else f"{size} "
        raise ModelInputError(f"{path} must be a vector or a sequence of {count}scalar values")
    if isinstance(value, Vector) and not value._finite:
        raise ModelInputError(f"{path} observations must have a known finite length")
    if size is not None and len(value) != size:
        raise ModelInputError(f"{path} needs {size} entries; received {len(value)}")
    return tuple(value)

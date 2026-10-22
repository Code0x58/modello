"""Finite symbolic vectors and indexed formulas built from SymPy scalars.

Vector coordinates participate in the ordinary joint scalar solve. Formulas
retain their indices until evaluation, so inspection can show the mathematics
that was declared instead of an expanded list of scalar operations.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from functools import cached_property
from typing import Any, TypeAlias, overload

from sympy import Basic, Expr, Integer, Product, Sum, Symbol, Tuple, sympify

from modello._symbols import Index, InstanceDummy
from modello._values import ScalarValue
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
        cls, *values: ScalarValue | Sequence[ScalarValue], size: int | None = None, **assumptions: Any
    ) -> Vector:
        if size is not None:
            if isinstance(size, bool) or not isinstance(size, (int, Integer)) or size < 0:
                raise ModelInputError("Vector size must be a nonnegative integer")
            if len(values) != 1 or not isinstance(values[0], str) or not values[0]:
                raise ModelInputError("A declared vector needs a name and size")
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
        if not isinstance(position, Expr) or position.is_integer is False:
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
        return Sum(self[i], (i, 0, len(self) - 1))

    def product(self, index: Symbol | None = None) -> Expr:
        """Return an unevaluated SymPy product over zero-based coordinates."""
        i = _local_index(self) if index is None else index
        return Product(self[i], (i, 0, len(self) - 1))

    def dot(self, other: Vector, index: Symbol | None = None) -> Expr:
        """Return the bilinear dot product as a SymPy sum (no conjugation)."""
        self._same_shape(other)
        i = _local_index(self, other) if index is None else index
        return Sum(self[i] * other[i], (i, 0, len(self) - 1))

    def __matmul__(self, other: Vector) -> Expr:
        return self.dot(other)

    def _same_shape(self, other: object) -> None:
        if not isinstance(other, Vector) or len(self) != len(other):
            raise ModelInputError("Vector operations require vectors of the same size")

    def __add__(self, other: Vector) -> Vector:
        self._same_shape(other)
        i = _local_index(self, other)
        return vector(self[i] + other[i], (i, 0, len(self) - 1))

    def __sub__(self, other: Vector) -> Vector:
        self._same_shape(other)
        i = _local_index(self, other)
        return vector(self[i] - other[i], (i, 0, len(self) - 1))

    def __neg__(self) -> Vector:
        return self * -1

    def __mul__(self, scalar: ScalarValue) -> Vector:
        value = sympify(scalar)
        if not isinstance(value, Expr) or value.is_Matrix:
            raise ModelInputError("Scale a vector by a scalar; use dot for a contraction")
        i = _local_index(self)
        return vector(self[i] * value, (i, 0, len(self) - 1))

    def __rmul__(self, scalar: ScalarValue) -> Vector:
        return self * scalar

    def __truediv__(self, scalar: ScalarValue) -> Vector:
        return self * (1 / sympify(scalar))

    def _sympystr(self, printer: Any) -> str:
        label = getattr(printer, "vector_labels", {}).get(self)
        if label is not None:
            return label
        return "[" + ", ".join(printer.doprint(value) for value in self) + "]"


class _VectorElement(Expr):
    """A scalar selection whose symbols are its coordinates and its index."""

    def __new__(cls, values: Vector, index: Expr) -> Expr:
        if isinstance(index, Integer):
            return values[int(index)]
        return Expr.__new__(cls, values, index)

    @property
    def vector(self) -> Vector:
        return self.args[0]

    @property
    def index(self) -> Expr:
        return self.args[1]

    def _eval_is_real(self) -> bool | None:
        return True if all(value.is_real is True for value in self.vector) else None

    def _eval_is_commutative(self) -> bool | None:
        assumptions = {value.is_commutative for value in self.vector}
        return False if False in assumptions else True if None not in assumptions else None

    def _eval_subs(self, old: Basic, new: Basic) -> Expr | None:
        # A local index normally does not occur in the coordinates themselves.
        # Avoid visiting the full vector for each term of a finite reduction.
        if isinstance(old, Symbol) and old not in self.vector._coordinate_symbols:
            return _VectorElement(self.vector, self.index.subs(old, new))
        return None

    def _sympystr(self, printer: Any) -> str:
        return f"{printer.doprint(self.vector)}[{printer.doprint(self.index)}]"


class _VectorFormula(Vector):
    """A finite comprehension, retaining its binder independently of its values."""

    def __new__(cls, expression: Expr, limits: tuple[Symbol, int, int] | Tuple) -> _VectorFormula:
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
        if any(not isinstance(bound, Integer) for bound in limits[1:]) or limits[2] < limits[1] - 1:
            raise ModelInputError("Vector formula bounds must describe a finite increasing integer range")
        return Basic.__new__(cls, expression, limits)

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
        return self.function.free_symbols - {self.limits[0]}

    @cached_property
    def elements(self) -> tuple[Expr, ...]:
        i, first, last = self.limits
        return tuple(self.function.subs(i, value) for value in range(int(first), int(last) + 1))

    def _at(self, position: int) -> Expr:
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
        return int(self.limits[2] - self.limits[1] + 1)

    def _eval_subs(self, old: Basic, new: Basic) -> _VectorFormula:
        variable, first, last = self.limits
        if old == variable:
            return self
        return _VectorFormula(self.function.subs(old, new), (variable, first.subs(old, new), last.subs(old, new)))

    def _sympystr(self, printer: Any) -> str:
        if hasattr(printer, "_print_VectorFormula"):
            return printer._print_VectorFormula(self)
        return f"vector({printer.doprint(self.function)}, {printer.doprint(self.limits)})"


def vector(expression: ScalarValue, limits: tuple[Symbol, int, int]) -> Vector:
    """Build a finite indexed vector, including both endpoints of ``limits``.

    For example, ``vector(x[i] - x[i-1], (i, 1, len(x)-1))`` describes adjacent
    changes. The result is zero-based: its first entry corresponds to ``i=1``.
    Like SymPy Sum, this retains the original function and bound variable.
    """
    return _VectorFormula(expression, limits)


VectorValue: TypeAlias = Vector | Sequence[ScalarValue | None]


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


def _vector_input(value: object, size: int, path: str) -> tuple[object, ...]:
    """Validate a complete shape; None marks an unobserved coordinate."""
    if not isinstance(value, (Vector, Sequence, Tuple)) or isinstance(value, (str, bytes)):
        raise ModelInputError(f"{path} must be a vector or a sequence of {size} scalar values")
    if len(value) != size:
        raise ModelInputError(f"{path} needs {size} entries; received {len(value)}")
    return tuple(value)

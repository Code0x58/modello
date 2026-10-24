"""Minimal upstream-only reproducers, kept separate from passing library contracts."""

import pytest
import sympy
from sympy import Array, Function, Subs, Sum, Symbol, diff, srepr
from sympy.series.sequences import RecursiveSeq


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="SymPy finite-Sum differentiation introduces a spurious zero singularity; diagnostics in docstring",
)
def test_differentiating_a_finite_polynomial_sum_preserves_its_value_at_zero():
    """SymPy 1.14.0 creates (6*x**3 + 2*x**2)/x**2 instead of 6*x + 2.

    Standalone reproducer, with no Modello imports::

        x = Symbol("x", real=True)
        i = Symbol("i", integer=True)
        source = Sum(x**(i + 1), (i, 0, 2))
        diff(source, x, 2).doit().subs(x, 0)  # nan; expected 2
        diff(source.doit(), x, 2).subs(x, 0)  # 2

    The finite source is x + x**2 + x**3, an everywhere-defined polynomial.
    Differentiation under the sum followed by summation introduces a removable
    x**2 denominator; direct substitution loses the valid zero point. Inspect
    Sum._eval_derivative and the symbolic-power summation path in
    sympy/concrete/summations.py. Expanding the finite sum before differentiation
    avoids it. A fix should retain that point, not only prove generic equality
    after cancellation. No upstream issue number has been established here.
    This strict xfail must be removed if a SymPy upgrade repairs the result.
    """
    x, i = Symbol("x", real=True), Symbol("i", integer=True)
    source = Sum(x ** (i + 1), (i, 0, 2))
    derivative = diff(source, x, 2).doit()
    expected = diff(source.doit(), x, 2).subs(x, 0)
    actual = derivative.subs(x, 0)
    assert actual == expected == 2, (
        f"SymPy {sympy.__version__}: source={source!r}; derivative={derivative!r}; "
        f"srepr={srepr(derivative)}; actual={actual}; expected={expected}"
    )


@pytest.mark.parametrize(
    "start",
    [
        0,
        pytest.param(
            10,
            marks=pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason="SymPy RecursiveSeq uses an absolute position as a cache offset; diagnostics in docstring",
            ),
        ),
        pytest.param(
            -3,
            marks=pytest.mark.xfail(
                strict=True,
                raises=UnboundLocalError,
                reason="SymPy RecursiveSeq leaves its shifted-origin loop variable unbound; diagnostics in docstring",
            ),
        ),
    ],
)
def test_recursive_sequence_coefficients_respect_a_shifted_origin(start):
    """SymPy 1.14.0 misaddresses cached values when RecursiveSeq starts away from zero.

    Standalone reproducer, with no Modello imports::

        n = Symbol("n", integer=True)
        y = Function("y")
        seq = RecursiveSeq(y(n - 1) + 1, y(n), n, [0], start=10)
        seq.coeff(12)  # 12; expected 2 (already stored as seq.cache[y(12)])

    With start=-3, coeff(-1) raises UnboundLocalError for ``current``; start=0
    passes. RecursiveSeq._eval_coeff in sympy/series/sequences.py iterates
    range(len(cache), index + 1), then adds self.start to its loop variable.
    The endpoint should use an offset from start. Positive origins compute
    extra terms and return the wrong one; negative origins can skip the loop.
    A workaround is a zero-based sequence with an external coordinate shift.
    No upstream issue number has been established here. The strict xfails pin
    the two distinct failures and must be removed once an upgrade repairs them.
    """
    n = Symbol("n", integer=True)
    y = Function("y")
    sequence = RecursiveSeq(y(n - 1) + 1, y(n), n, [0], start=start)
    point = start + 2
    actual = sequence.coeff(point)
    assert actual == 2, (
        f"SymPy {sympy.__version__}: start={start}; point={point}; actual={actual}; "
        f"expected=2; cache={sequence.cache!r}; recurrence={srepr(sequence.recurrence)}"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="SymPy Subs equality conflates ordered overlapping replacements; diagnostics in docstring",
)
def test_ordered_compound_substitutions_with_different_values_have_distinct_identities():
    """SymPy 1.14.0 equates substitutions with different ordered evaluations.

    Standalone reproducer, with no Modello imports::

        n, j, k = symbols("n j k")
        f = Function("f")
        first = Subs(f(j), (j, j+k), (j+k, n-2))
        second = Subs(f(j), (j+k, j), (n-2, j+k))
        first == second  # True; expected False
        first.doit(), second.doit()  # (f(n-2), f(j+k))

    Subs.doit applies replacements in order, but Subs._hashable_content in
    sympy/core/function.py uses a normalized expression that loses that order.
    Equal hashes and equality can then let enclosing cached expressions reuse
    the wrong substitution, before a downstream library receives the input.
    Use nonoverlapping symbolic binders, or evaluate substitutions before building
    enclosing expressions when safe. No upstream issue number is established.
    Remove this strict xfail once equality distinguishes these two operations.
    """
    n, j, k = (Symbol(name) for name in ("n", "j", "k"))
    f = Function("f")
    first = Subs(f(j), (j, j + k), (j + k, n - 2))
    second = Subs(f(j), (j + k, j), (n - 2, j + k))
    assert first != second, (
        f"SymPy {sympy.__version__}: first={srepr(first)}; second={srepr(second)}; "
        f"values={(first.doit(), second.doit())}; hashes={(hash(first), hash(second))}"
    )


@pytest.mark.parametrize(
    "entries",
    [
        (1,),
        pytest.param(
            (),
            marks=pytest.mark.xfail(
                strict=True,
                raises=TypeError,
                reason="SymPy Array.applyfunc cannot reconstruct an empty array from map; diagnostics in docstring",
            ),
        ),
    ],
)
def test_array_applyfunc_preserves_empty_and_nonempty_shapes(entries):
    """SymPy 1.14.0 raises TypeError for Array([]).applyfunc, before calling f.

    Standalone reproducer, with no Modello imports::

        from sympy import Array
        Array([]).applyfunc(lambda value: value**2 + 1)
        # TypeError: object of type 'map' has no len()
        # expected an empty array with shape (0,)

    NDimArray.applyfunc in sympy/tensor/array/ndim_array.py passes a map object
    to the array constructor. The empty-array reconstruction path requires a
    length and rejects that iterator. Materializing the transformed entries
    before construction avoids the error; the nonempty control passes.
    No upstream issue number is established. Remove the strict xfail after
    an upgrade repairs empty-array reconstruction.
    """
    source = Array(entries)
    result = source.applyfunc(lambda value: value**2 + 1)
    assert result.shape == source.shape
    assert list(result) == [value**2 + 1 for value in entries], f"SymPy {sympy.__version__}: {srepr(source)}"

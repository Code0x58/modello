"""Minimal upstream-only reproducers, kept separate from passing library contracts."""

import pytest
import sympy
from sympy import Sum, Symbol, diff, srepr


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

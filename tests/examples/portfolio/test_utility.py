"""Symbolic cash utility, derivative ratios, and evaluation domains."""

from datetime import date

import pytest
from sympy import Rational, Symbol, exp, log

from examples.compute.series import Series
from examples.portfolio.utility import CashUtility, cara, crra


def test_utility_derivatives_and_affine_invariance():
    exponential = cara(0.001)
    wealth = exponential.wealth
    assert exponential.expressions["marginal_utility"] == exp(-wealth / 1000)
    assert exponential.expressions["absolute_risk_aversion"] == Rational(1, 1000)
    power = crra(2)
    assert power.expressions["absolute_risk_aversion"] == 2 / power.wealth
    assert power.expressions["relative_risk_aversion"] == 2
    logarithmic = crra(1)
    assert logarithmic.expression == log(logarithmic.wealth)
    transformed = CashUtility("Scaled log", logarithmic.wealth, 5 * logarithmic.expression + 7)
    for field in ("absolute_risk_aversion", "relative_risk_aversion"):
        assert transformed.expressions[field] == logarithmic.expressions[field]
    for utility in (cara(0), crra(0)):
        assert utility.expressions["absolute_risk_aversion"] == 0
        assert utility.expressions["relative_risk_aversion"] == 0


def test_utility_reports_wealth_preferences_including_the_deposit(reporting):
    assert reporting.utilities["cara"].values["absolute_risk_aversion"].values == (0.001,) * 4
    assert reporting.utilities["crra"].values["relative_risk_aversion"].values == (2,) * 4
    assert reporting.utilities["crra"].values["absolute_risk_aversion"].values == pytest.approx(
        tuple(2 / value for value in reporting.path.nav.values)
    )


@pytest.mark.parametrize("coefficient", [-1, float("inf"), float("nan")])
def test_invalid_risk_aversion_coefficients(coefficient):
    for constructor in (cara, crra):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            constructor(coefficient)


def test_utility_domains_and_undefined_ratios():
    times = (date(2025, 1, 1), date(2026, 1, 1))
    with pytest.raises(ValueError, match="positive wealth"):
        crra(2).over(Series(times, (0, 100)))
    w = Symbol("wealth", positive=True)
    with pytest.raises(ValueError, match="only free symbol"):
        CashUtility("Unbound", w, w * Symbol("alpha"))
    with pytest.raises(ValueError, match="vary with wealth"):
        CashUtility("Constant", w, Rational(1))
    with pytest.raises(ValueError, match="increasing"):
        CashUtility("Decreasing", w, -w).over(Series(times, (100, 200)))
    with pytest.raises(ValueError, match="finite"):
        CashUtility("Singular", w, log(w - 100)).over(Series(times, (100, 200)))

"""A reconciliation alternative must retain trusted facts and all other inputs."""

import pytest
from sympy import FiniteSet, Rational, S

from examples.reconciliation import CDSMark, CDSPnL, CDSTerms, REDData, sample_problem
from modello import FieldValue, ModelSolveError, UnresolvedSolutionError, inspect_model


def test_recorded_pnl_and_break_are_exact():
    problem = sample_problem()
    recorded = problem.without("pnl").solutions().one()
    assert isinstance(recorded, CDSPnL)
    assert isinstance(recorded.closing, CDSMark)
    assert (recorded.opening.value, recorded.closing.value, recorded.pnl) == (2200, 8300, -3900)
    assert recorded.pnl - 51_000 == -54_900
    assert problem.analyse().status == "inconsistent"


@pytest.mark.parametrize(
    ("released", "expected"),
    [("signed_notional", 10_000_000), ("coupon_bps", 55_000), ("cashflow", 44_900)],
)
def test_each_alternative_keeps_the_other_observations(released, expected):
    problem = sample_problem()
    trial = problem.without(released)
    family = trial.solutions()
    assert family.values(problem.field(released)) == FiniteSet(expected)
    case = family.one()  # Materialisation audits the complete retained problem.
    assert getattr(case, released) == expected
    assert case.pnl == 51_000
    for name, original in {
        "signed_notional": 1_000_000,
        "coupon_bps": 100,
        "recovery": Rational(2, 5),
        "cashflow": -10_000,
    }.items():
        if name != released:
            assert getattr(case, name) == original
    assert case.opening.terms.signed_notional == case.closing.terms.signed_notional == case.signed_notional
    assert case.opening.terms.coupon_bps == case.closing.terms.coupon_bps == case.coupon_bps
    assert case.opening.red.recovery == case.closing.red.recovery == case.recovery
    assert case.opening.red.survival == Rational(98, 100)
    assert case.closing.red.survival == Rational(97, 100)
    assert all(c in trial.constraints for c in problem.constraints if c.input != released)
    assert any(c.input == released for c in problem.constraints)  # The source was not edited.


def test_recovery_alone_cannot_explain_the_break_in_its_domain():
    family = sample_problem().without("recovery").solutions()
    assert family.is_empty is True
    assert family.cases() == ()


@pytest.mark.parametrize("released", ["signed_notional", "coupon_bps", "cashflow"])
def test_independent_opening_valuation_discriminates_between_explanations(released):
    problem = sample_problem()
    alternatives = problem.without(released).solutions()
    checked = alternatives.observe(FieldValue(problem.field("opening", "value"), 22_000))
    if released == "signed_notional":
        case = checked.one()
        assert (case.signed_notional, case.opening.value, case.closing.value) == (10_000_000, 22_000, 83_000)
    else:
        assert checked.is_empty is True


def test_releasing_several_inputs_can_leave_a_family():
    family = sample_problem().without("signed_notional", "cashflow").solutions()
    assert family.parameters
    with pytest.raises(UnresolvedSolutionError):
        family.one()


@pytest.mark.parametrize("field", ["recovery", "survival"])
def test_probability_and_recovery_bounds_are_real_constraints(field):
    with pytest.raises(ModelSolveError):
        REDData("outside domain", **{field: Rational(11, 10)})


def test_formula_is_the_same_for_both_protection_sides():
    data = REDData("credit inputs", recovery=Rational(2, 5), survival=Rational(97, 100))
    buyer = CDSMark(
        "buyer",
        red=data,
        discount=1,
        year_fraction=1,
        terms=CDSTerms("buyer terms", signed_notional=10_000_000, coupon_bps=100),
    )
    seller = CDSMark(
        "seller",
        red=data,
        discount=1,
        year_fraction=1,
        terms=CDSTerms("seller terms", signed_notional=-10_000_000, coupon_bps=100),
    )
    assert buyer.value == -seller.value == 83_000
    p = inspect_model(CDSMark, "formula")

    def field(*path):
        return p.field(*path).symbol

    expected = (
        field("terms", "signed_notional")
        * field("discount")
        * (
            (1 - field("red", "recovery")) * (1 - field("red", "survival"))
            - field("terms", "coupon_bps") / 10000 * field("year_fraction") * field("red", "survival")
        )
    )
    assert (p.solutions().expression(p.field("value")) - expected).simplify() == S.Zero

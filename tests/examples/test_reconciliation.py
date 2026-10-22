"""CDS reconciliation retains trusted facts, trade terms, and reference data."""

import pytest
from sympy import Eq, FiniteSet, Rational, S

from examples.reconcile import Reconciliation, Suspect
from examples.reconciliation import CDSMark, CDSPnL, CDSTerms, CreditParameters, sample_problem
from modello import ModelProblem, ModelSolveError, UnresolvedSolutionError, inspect_model


def pnl_problem(data: CreditParameters) -> ModelProblem[CDSPnL]:
    return inspect_model(
        CDSPnL,
        "RED comparison",
        opening=CDSMark(
            "opening", credit=CreditParameters("opening", survival=Rational(98, 100)), discount=1, year_fraction=1
        ),
        closing=CDSMark("closing", credit=data, discount=1, year_fraction=1),
        signed_notional=10_000_000,
        coupon_bps=100,
        recovery=Rational(2, 5),
        cashflow=-10_000,
        pnl=51_000,
    )


def test_swapping_credit_component_explains_break_without_changing_trade_terms():
    previous = CreditParameters("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    current = CreditParameters("current", recovery=Rational(2, 5), survival=Rational(97, 100))
    stale_problem = pnl_problem(previous)
    stale = stale_problem.without("pnl").solutions().one()
    assert (stale.opening.value, stale.closing.value, stale.pnl) == (22_000, 52_500, 20_500)
    assert stale_problem.analyse().status == "inconsistent"

    refreshed_problem = pnl_problem(current)
    assert refreshed_problem.analyse().status == "consistent"
    refreshed = refreshed_problem.solutions().one()
    assert (refreshed.opening.value, refreshed.closing.value, refreshed.pnl) == (22_000, 83_000, 51_000)
    for name in ("signed_notional", "coupon_bps", "recovery", "cashflow"):
        assert getattr(refreshed, name) == getattr(stale, name)
    assert previous.survival == stale.closing.credit.survival == Rational(39, 40)
    assert current.survival == refreshed.closing.credit.survival == Rational(97, 100)
    assert stale_problem.without("pnl").solutions().one().pnl == 20_500


def test_component_values_and_domains_survive_copying():
    data = CreditParameters("source", survival=Rational(97, 100))
    result = pnl_problem(data).solutions().one()
    assert isinstance(result.closing.credit, CreditParameters)
    assert result.closing.credit is not data
    assert result.closing.credit.recovery == Rational(2, 5)
    assert result.closing.credit.loss_fraction == Rational(3, 5)
    assert result.closing.credit.default_probability == Rational(3, 100)
    assert data.recovery.free_symbols  # Completing the parent leaves the source symbolic.


def test_inverse_notional_could_fit_pnl_while_hiding_bad_reference_data():
    previous = CreditParameters("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    problem = pnl_problem(previous)
    notional = problem.field("signed_notional")
    hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
    assert hypothetical.signed_notional == 20_000_000
    assert hypothetical.pnl == 51_000


def test_swapping_credit_component_uses_both_recovery_and_survival():
    previous = CreditParameters("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    current = CreditParameters("current", recovery=Rational(3, 10), survival=Rational(97, 100))
    terms = CDSTerms("trade", signed_notional=10_000_000, coupon_bps=100)
    old_mark = CDSMark("old", terms=terms, credit=previous, discount=1, year_fraction=1)
    new_mark = CDSMark("new", terms=terms, credit=current, discount=1, year_fraction=1)
    assert (old_mark.protection, old_mark.premium, old_mark.value) == (150_000, 97_500, 52_500)
    assert (new_mark.protection, new_mark.premium, new_mark.value) == (210_000, 97_000, 113_000)


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
    assert family.values(problem.field(released).symbol) == FiniteSet(expected)
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
    assert case.opening.credit.recovery == case.closing.credit.recovery == case.recovery
    assert case.opening.credit.survival == Rational(98, 100)
    assert case.closing.credit.survival == Rational(97, 100)
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
    checked = alternatives.where(Eq(problem.symbols.opening.value, 22_000))
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
        CreditParameters("outside domain", **{field: Rational(11, 10)})


def test_formula_is_the_same_for_both_protection_sides():
    data = CreditParameters("credit inputs", recovery=Rational(2, 5), survival=Rational(97, 100))
    buyer = CDSMark(
        "buyer",
        credit=data,
        discount=1,
        year_fraction=1,
        terms=CDSTerms("buyer terms", signed_notional=10_000_000, coupon_bps=100),
    )
    seller = CDSMark(
        "seller",
        credit=data,
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
            (1 - field("credit", "recovery")) * (1 - field("credit", "survival"))
            - field("terms", "coupon_bps") / 10000 * field("year_fraction") * field("credit", "survival")
        )
    )
    assert (p.solutions().expression(p.symbols.value) - expected).simplify() == S.Zero

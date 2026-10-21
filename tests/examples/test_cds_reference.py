"""Compare inline RED parameter components with trusted PnL."""

from sympy import Rational

from examples.reconcile import Reconciliation, Suspect
from examples.reconciliation import CDSMark, CDSPnL, CDSTerms, REDData
from modello import ModelProblem, inspect_model


def pnl_problem(data: REDData) -> ModelProblem[CDSPnL]:
    return inspect_model(
        CDSPnL,
        "RED comparison",
        opening=CDSMark("opening", red=REDData("opening", survival=Rational(98, 100)), discount=1, year_fraction=1),
        closing=CDSMark("closing", red=data, discount=1, year_fraction=1),
        signed_notional=10_000_000,
        coupon_bps=100,
        recovery=Rational(2, 5),
        cashflow=-10_000,
        pnl=51_000,
    )


def test_swapping_red_component_explains_break_without_changing_trade_terms():
    previous = REDData("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    current = REDData("current", recovery=Rational(2, 5), survival=Rational(97, 100))
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
    assert previous.survival == stale.closing.red.survival == Rational(39, 40)
    assert current.survival == refreshed.closing.red.survival == Rational(97, 100)
    assert stale_problem.without("pnl").solutions().one().pnl == 20_500


def test_component_values_and_domains_survive_copying():
    data = REDData("source", survival=Rational(97, 100))
    result = pnl_problem(data).solutions().one()
    assert isinstance(result.closing.red, REDData)
    assert result.closing.red is not data
    assert result.closing.red.recovery == Rational(2, 5)
    assert result.closing.red.loss_fraction == Rational(3, 5)
    assert result.closing.red.default_probability == Rational(3, 100)
    assert data.recovery.free_symbols  # Completing the parent leaves the source symbolic.


def test_relabelling_old_inputs_does_not_change_the_price():
    for label in ("DEMO:PAIR:OLD", "DEMO:PAIR:CURRENT"):
        data = REDData(label, recovery=Rational(2, 5), survival=Rational(39, 40))
        problem = pnl_problem(data)
        assert problem.analyse().status == "inconsistent"
        assert problem.without("pnl").solutions().one().pnl == 20_500


def test_inverse_notional_could_fit_pnl_while_hiding_bad_reference_data():
    previous = REDData("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    problem = pnl_problem(previous)
    notional = problem.field("signed_notional")
    hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
    assert hypothetical.signed_notional == 20_000_000
    assert hypothetical.pnl == 51_000


def test_swapping_red_component_uses_both_recovery_and_survival():
    previous = REDData("previous", recovery=Rational(2, 5), survival=Rational(39, 40))
    current = REDData("current", recovery=Rational(3, 10), survival=Rational(97, 100))
    terms = CDSTerms("trade", signed_notional=10_000_000, coupon_bps=100)
    old_mark = CDSMark("old", terms=terms, red=previous, discount=1, year_fraction=1)
    new_mark = CDSMark("new", terms=terms, red=current, discount=1, year_fraction=1)
    assert (old_mark.protection, old_mark.premium, old_mark.value) == (150_000, 97_500, 52_500)
    assert (new_mark.protection, new_mark.premium, new_mark.value) == (210_000, 97_000, 113_000)

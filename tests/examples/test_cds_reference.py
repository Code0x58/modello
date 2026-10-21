"""Compare source components with trusted PnL while retaining trade terms."""

from sympy import Rational

from examples.reconcile import Reconciliation, Suspect
from examples.reconciliation import CurrentREDMarket, PreviousREDMarket, reference_problem


def test_swapping_source_component_explains_break_without_changing_trade_terms():
    stale_problem = reference_problem(PreviousREDMarket)
    stale = stale_problem.without("pnl").solutions().one()
    assert (stale.opening.value, stale.closing.value, stale.pnl) == (22_000, 52_500, 20_500)
    assert stale_problem.analyse().status == "inconsistent"

    current_problem = reference_problem(CurrentREDMarket)
    assert current_problem.analyse().status == "consistent"
    refreshed = current_problem.solutions().one()
    assert (refreshed.opening.value, refreshed.closing.value, refreshed.pnl) == (22_000, 83_000, 51_000)
    for name in ("signed_notional", "coupon_bps", "recovery", "cashflow"):
        assert getattr(refreshed, name) == getattr(stale, name)
    assert stale.closing.market.survival == Rational(39, 40)
    assert refreshed.closing.market.survival == Rational(97, 100)
    assert stale_problem.without("pnl").solutions().one().pnl == 20_500


def test_source_type_and_provenance_survive_composition():
    stale = reference_problem(PreviousREDMarket).without("pnl").solutions().one().closing.market
    current = reference_problem(CurrentREDMarket).solutions().one().closing.market
    assert type(stale) is PreviousREDMarket
    assert type(current) is CurrentREDMarket
    assert stale.entity_code == current.entity_code == "DEMO:ENTITY"
    assert (stale.pair_code, current.pair_code) == ("DEMO:PAIR:OLD", "DEMO:PAIR:CURRENT")
    assert stale.as_of < current.as_of


def test_relabelling_old_inputs_does_not_change_the_price():
    class RelabelledMarket(PreviousREDMarket):
        pair_code = CurrentREDMarket.pair_code
        as_of = CurrentREDMarket.as_of

    problem = reference_problem(RelabelledMarket)
    assert problem.analyse().status == "inconsistent"
    assert problem.without("pnl").solutions().one().pnl == 20_500


def test_same_codes_and_date_can_still_have_wrong_parameters():
    class WrongParameters(CurrentREDMarket):
        survival = Rational(39, 40)

    assert WrongParameters.pair_code == CurrentREDMarket.pair_code
    assert WrongParameters.as_of == CurrentREDMarket.as_of
    assert reference_problem(WrongParameters).analyse().status == "inconsistent"


def test_contract_definitions_are_metadata_not_a_model_of_legal_payoffs():
    class LegacyMarket(CurrentREDMarket):
        definitions = "2003"

    legacy = reference_problem(LegacyMarket).solutions().one()
    current = reference_problem(CurrentREDMarket).solutions().one()
    assert legacy.closing.market.definitions == "2003"
    assert current.closing.market.definitions == "2014"
    assert legacy.pnl == current.pnl  # No payoff difference is modelled here.


def test_inverse_notional_could_fit_pnl_while_hiding_bad_reference_data():
    problem = reference_problem(PreviousREDMarket)
    notional = problem.field("signed_notional")
    hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
    assert hypothetical.signed_notional == 20_000_000
    assert hypothetical.pnl == 51_000
    assert reference_problem(CurrentREDMarket).solutions().one().signed_notional == 10_000_000


def test_swapping_component_uses_all_of_its_pricing_assumptions():
    class OtherMarket(CurrentREDMarket):
        discount = Rational(99, 100)
        year_fraction = Rational(9, 10)

    repriced = reference_problem(OtherMarket).without("pnl").solutions().one()
    assert repriced.closing.market.discount == Rational(99, 100)
    assert repriced.closing.market.year_fraction == Rational(9, 10)
    assert repriced.closing.market.survival == Rational(97, 100)
    assert repriced.pnl == 59_773
    assert repriced.signed_notional == 10_000_000


def test_different_parameter_components_can_both_fit_the_same_pnl():
    class AnotherFit(CurrentREDMarket):
        survival = Rational(969, 1000)
        discount = Rational(830, 891)

    for market in (CurrentREDMarket, AnotherFit):
        assert reference_problem(market).analyse().status == "consistent"
        assert reference_problem(market).solutions().one().pnl == 51_000

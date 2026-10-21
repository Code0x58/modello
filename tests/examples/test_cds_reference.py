"""Reference selection follows confirmed identity and date, never matching PnL."""

from dataclasses import replace
from datetime import date

import pytest
from sympy import Rational

from examples.reconcile import Reconciliation, Suspect
from examples.reconciliation import stale_reference_case


def test_stale_pair_bundle_explains_the_break_without_changing_trade_terms():
    case = stale_reference_case()
    assert case.loaded.key.entity_code == case.confirmed.entity_code
    assert case.mismatches == ("pair_code", "as_of")
    stale_problem = case.problem(case.loaded)
    stale = stale_problem.without("pnl").solutions().one()
    assert (stale.opening.value, stale.closing.value, stale.pnl) == (22_000, 52_500, 20_500)
    assert stale_problem.analyse().status == "inconsistent"
    selected = case.select()
    refreshed = case.problem(selected).solutions().one()
    assert (refreshed.opening.value, refreshed.closing.value, refreshed.pnl) == (22_000, 83_000, 51_000)
    for name in ("signed_notional", "coupon_bps", "recovery", "cashflow"):
        assert getattr(refreshed, name) == getattr(stale, name)
    assert case.loaded.survival == Rational(39, 40)
    assert selected.survival == Rational(97, 100)


def test_relabelling_old_inputs_does_not_change_the_price():
    case = stale_reference_case()
    relabelled = replace(case.loaded, key=case.confirmed, as_of=case.as_of)
    assert not replace(case, loaded=relabelled).mismatches
    assert case.problem(relabelled).without("pnl").solutions().one().pnl == 20_500


def test_inverse_notional_could_fit_pnl_while_hiding_bad_reference_data():
    case = stale_reference_case()
    problem = case.problem(case.loaded)
    notional = problem.field("signed_notional")
    hypothetical = Reconciliation(problem, [Suspect(notional)]).infer(notional).one()
    assert hypothetical.signed_notional == 20_000_000
    assert hypothetical.pnl == 51_000
    assert case.problem(case.select()).solutions().one().signed_notional == 10_000_000


def test_same_codes_can_still_have_stale_market_inputs():
    case = stale_reference_case()
    same_codes = replace(case, loaded=replace(case.loaded, key=case.confirmed))
    assert same_codes.mismatches == ("as_of",)
    assert same_codes.select().as_of == case.as_of


def test_same_codes_do_not_select_between_2003_and_2014_contracts():
    case = stale_reference_case()
    current = case.select()
    legacy = replace(current, key=replace(current.key, definitions="2003"))
    wrong_definitions = replace(case, loaded=legacy, available=(legacy, current))
    assert wrong_definitions.mismatches == ("definitions",)
    assert wrong_definitions.select() is current
    # A confirmed legacy contract is not automatically upgraded to 2014 terms.
    confirmed_legacy = replace(wrong_definitions, confirmed=legacy.key)
    assert confirmed_legacy.select() is legacy


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entity_code", "DEMO:OTHER"),
        ("pair_code", "DEMO:PAIR:OTHER"),
        ("currency", "EUR"),
        ("definitions", "2003"),
        ("seniority", "subordinated"),
        ("restructuring", "MM"),
        ("maturity", date(2028, 1, 5)),
    ],
)
def test_every_contract_coordinate_is_part_of_selection(field, value):
    case = stale_reference_case()
    wrong = replace(case.select(), key=replace(case.confirmed, **{field: value}))
    with pytest.raises(ValueError, match="found 0"):
        replace(case, available=(wrong,)).select()


def test_requested_date_preserves_historical_mappings():
    case = stale_reference_case()
    historical = replace(case, confirmed=case.loaded.key, as_of=case.loaded.as_of)
    assert historical.select() is case.loaded
    wrong_date = replace(case.select(), as_of=case.loaded.as_of)
    with pytest.raises(ValueError, match="found 0"):
        replace(case, available=(wrong_date,)).select()


def test_duplicate_sources_are_not_resolved_by_choosing_the_pnl_match():
    case = stale_reference_case()
    current = case.select()
    conflicting = replace(current, survival=Rational(98, 100))
    assert case.problem(current).analyse().status == "consistent"
    assert case.problem(conflicting).analyse().status == "inconsistent"
    with pytest.raises(ValueError, match="found 2"):
        replace(case, available=(current, conflicting)).select()


def test_repricing_uses_the_whole_selected_bundle():
    case = stale_reference_case()
    changed = replace(case.select(), discount=Rational(99, 100), year_fraction=Rational(9, 10))
    repriced = case.problem(changed).without("pnl").solutions().one()
    assert repriced.closing.market.discount == Rational(99, 100)
    assert repriced.closing.market.year_fraction == Rational(9, 10)
    assert repriced.closing.market.survival == changed.survival
    assert repriced.signed_notional == 10_000_000

"""Posting models reuse balance equations in forward and inverse views."""

from examples.portfolio.postings import CapitalFlow, FillPosting


def test_a_fill_can_be_valued_or_its_execution_price_inferred_from_cash():
    posted = FillPosting("fill", opening_quantity=3, opening_cash=1000, quantity=2, price=100, fee=1)
    inferred = FillPosting("fit", opening_quantity=3, opening_cash=1000, quantity=2, closing_cash=799, fee=1)
    assert posted.closing_quantity == 5
    assert posted.closing_cash == 799
    assert inferred.price == 100
    assert inferred.closing_quantity == posted.closing_quantity


def test_external_capital_preserves_the_pre_flow_unit_price_in_both_directions():
    flow = CapitalFlow("deposit", opening_cash=1000, opening_nav=2500, opening_units=25, amount=500)
    inferred = CapitalFlow("fit", opening_cash=1000, opening_nav=2500, opening_units=25, closing_units=30)
    assert flow.unit_price == 100
    assert flow.closing_cash == 1500
    assert flow.closing_units == 30
    assert inferred.amount == 500
    assert inferred.closing_cash == flow.closing_cash

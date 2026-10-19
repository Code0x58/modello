"""Shared symbolic profit models, sensitivities, and inverse reconciliation."""

from sympy import Rational, diff, simplify

from examples.portfolio.pnl import AccountPnL, InstrumentPnL


def test_reporting_models_support_symbolic_sensitivities_and_inverse_reconciliation():
    instrument = InstrumentPnL("instrument")
    assert (
        simplify(
            diff(instrument.net, instrument.opening_quantity) - (instrument.closing_price - instrument.opening_price)
        )
        == 0
    )
    assert (
        simplify(
            diff(instrument.net, instrument.closing_price) - instrument.opening_quantity - instrument.fill_quantity
        )
        == 0
    )
    assert diff(instrument.net, instrument.fees) == -1
    account = AccountPnL(
        "account",
        opening_nav=2500,
        external_flows=500,
        holding=Rational(57),
        trading=Rational(17, 5),
        income=Rational(17, 2),
        fees=2,
        residual=0,
    )
    assert account.net == Rational(669, 10)
    assert account.closing_nav == Rational(30669, 10)


def test_profit_model_compiles_a_fitted_fill_notional_view():
    from sympy import Symbol

    from examples.compute.named_map import NamedMap

    observed = Symbol("observed_net", real=True)
    fitted = InstrumentPnL(
        "fit",
        opening_quantity=10,
        opening_price=100,
        closing_price=105,
        fill_quantity=2,
        income=0,
        fees=1,
        net=observed,
    )
    plan = fitted.compile(inputs=("net",), outputs=("fill_notional",))
    assert plan.evaluate(NamedMap({"net": 59}))["fill_notional"] == 200

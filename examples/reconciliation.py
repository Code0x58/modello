"""Investigate a synthetic CDS PnL break by releasing individual observations.

Run ``python -m examples.reconciliation``. The pricing model has one remaining
period: premium on survival and protection on default, both paid at its end.
It omits accrual on default, schedules, curve calibration, and settlement
conventions. See the documentation for the precise scope and sign conventions.
"""

from sympy import Rational, S

from modello import FieldValue, InstanceDummy, Modello, ModelProblem, component, inspect_model


class CDSTerms(Modello):
    """Signed notional: positive buys protection; coupon in annual basis points."""

    signed_notional = InstanceDummy("signed_notional", real=True, nonzero=True)
    coupon_bps = InstanceDummy("coupon_bps", nonnegative=True)
    recovery = InstanceDummy("recovery", nonnegative=True)
    loss_fraction = InstanceDummy("loss_fraction", nonnegative=True)
    loss_fraction = S.One - recovery


class CreditPeriod(Modello):
    """Trusted one-period market inputs, with survival constrained to [0, 1]."""

    discount = InstanceDummy("discount", positive=True)
    survival = InstanceDummy("survival", nonnegative=True)
    default_probability = InstanceDummy("default_probability", nonnegative=True)
    default_probability = S.One - survival
    year_fraction = InstanceDummy("year_fraction", positive=True)
    risky_annuity = discount * survival * year_fraction
    default_factor = discount * default_probability


class CDSMark(Modello):
    """Signed contract PV in notional currency under the one-period model."""

    terms = component(CDSTerms)
    market = component(CreditPeriod)
    protection = terms.signed_notional * terms.loss_fraction * market.default_factor
    premium = terms.signed_notional * terms.coupon_bps / 10000 * market.risky_annuity
    value = protection - premium


class CDSPnL(Modello):
    """Unchanged trade at two marks, plus net cash received over the interval.

    Top-level aliases expose the original trade observations individually.
    The marks are connected by equations, rather than by object identity.
    """

    opening = component(CDSMark)
    closing = component(CDSMark)
    signed_notional = closing.terms.signed_notional
    signed_notional = opening.terms.signed_notional
    coupon_bps = closing.terms.coupon_bps
    coupon_bps = opening.terms.coupon_bps
    recovery = closing.terms.recovery
    recovery = opening.terms.recovery
    cashflow = InstanceDummy("cashflow", real=True)
    pnl = closing.value - opening.value + cashflow


def sample_problem() -> ModelProblem[CDSPnL]:
    """Keep raw trade observations separate from trusted markets and PnL.

    The recorded notional is 1m; the synthetic reference uses 10m. No payments
    occur between these marks except the separately recorded net cashflow.
    Both marks use a one-year remaining period to isolate a market-data change.
    """
    return inspect_model(
        CDSPnL,
        "PnL break",
        opening=CDSMark(
            "opening", market=CreditPeriod("opening market", discount=1, survival=Rational(98, 100), year_fraction=1)
        ),
        closing=CDSMark(
            "closing", market=CreditPeriod("closing market", discount=1, survival=Rational(97, 100), year_fraction=1)
        ),
        signed_notional=1_000_000,
        coupon_bps=100,
        recovery=Rational(2, 5),
        cashflow=-10_000,
        pnl=51_000,
    )


def main() -> None:
    """Show the formula, the break, and individually implied alternatives."""
    pricing = inspect_model(CDSMark, "pricing")
    expression = pricing.solutions().expression(pricing.field("value"))
    print("Pricing formula:", pricing.format(expression))
    problem = sample_problem()
    recorded = problem.without("pnl").solutions().one()
    print("Recorded PnL:", recorded.pnl)
    print("Trusted PnL:", 51_000)
    print("Break (recorded - trusted):", recorded.pnl - 51_000)
    print("One observation released at a time; all other observations stay fixed:")
    for name in ("signed_notional", "coupon_bps", "recovery", "cashflow"):
        family = problem.without(name).solutions()
        print(f"  {name}: {family.values(problem.field(name))}")
        # cases() checks every state against the retained equations and domains.
        for case in family.cases():
            print(f"    opening PV={case.opening.value}, closing PV={case.closing.value}, PnL={case.pnl}")
    print("Also requiring independently observed opening PV=22000:")
    opening_value = FieldValue(problem.field("opening", "value"), 22_000)
    for name in ("signed_notional", "coupon_bps", "cashflow"):
        family = problem.without(name).solutions().observe(opening_value)
        print(f"  {name}: {family.values(problem.field(name))}")
    print("Alternatives explain the number; confirm any correction against independent records.")


if __name__ == "__main__":
    main()

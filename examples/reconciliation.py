"""Investigate synthetic CDS breaks from entry errors and stale reference data.

Run ``python -m examples.reconciliation``. The pricing model has one remaining
period: premium on survival and protection on default, both paid at its end.
It omits accrual on default, schedules, curve calibration, and settlement
conventions. See the documentation for the precise scope and sign conventions.
"""

from sympy import Eq, Rational, S

from examples.reconcile import Reconciliation, Suspect
from modello import InstanceDummy, Modello, ModelProblem, component, inspect_model


class CDSTerms(Modello):
    """Signed notional: positive buys protection; coupon in annual basis points."""

    signed_notional = InstanceDummy("signed_notional", real=True, nonzero=True)
    coupon_bps = InstanceDummy("coupon_bps", nonnegative=True)


class CreditParameters(Modello):
    """Recovery and one-period survival assumptions supplied by a pricing source."""

    recovery = InstanceDummy("recovery", nonnegative=True)
    loss_fraction = InstanceDummy("loss_fraction", nonnegative=True)
    loss_fraction = S.One - recovery
    survival = InstanceDummy("survival", nonnegative=True)
    default_probability = InstanceDummy("default_probability", nonnegative=True)
    default_probability = S.One - survival


class CDSMark(Modello):
    """Signed present value (PV) in notional currency for one remaining period."""

    terms = component(CDSTerms)
    credit = component(CreditParameters)
    discount = InstanceDummy("discount", positive=True)
    year_fraction = InstanceDummy("year_fraction", positive=True)
    risky_annuity = discount * credit.survival * year_fraction
    protection = terms.signed_notional * credit.loss_fraction * discount * credit.default_probability
    premium = terms.signed_notional * terms.coupon_bps / 10000 * risky_annuity
    value = protection - premium


class CDSPnL(Modello):
    """Unchanged trade at two marks, plus net cash received over the interval.

    Top-level aliases expose the original observations individually.
    The marks are connected by equations, rather than by object identity.
    """

    opening = component(CDSMark)
    closing = component(CDSMark)
    signed_notional = closing.terms.signed_notional
    signed_notional = opening.terms.signed_notional
    coupon_bps = closing.terms.coupon_bps
    coupon_bps = opening.terms.coupon_bps
    recovery = closing.credit.recovery
    recovery = opening.credit.recovery
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
            "opening", credit=CreditParameters("opening", survival=Rational(98, 100)), discount=1, year_fraction=1
        ),
        closing=CDSMark(
            "closing", credit=CreditParameters("closing", survival=Rational(97, 100)), discount=1, year_fraction=1
        ),
        signed_notional=1_000_000,
        coupon_bps=100,
        recovery=Rational(2, 5),
        cashflow=-10_000,
        pnl=51_000,
    )


def main() -> None:
    """Try common errors first; expose inverse inference as a separate stage."""
    pricing = inspect_model(CDSMark, "pricing")
    expression = pricing.solutions().expression(pricing.symbols.value)
    print("Pricing formula:", pricing.format(expression))
    problem = sample_problem()
    calculated = problem.without("pnl").solutions().one()
    print("Calculated PnL from recorded inputs:", calculated.pnl)
    print("Trusted PnL:", 51_000)
    print("Break (calculated - trusted):", calculated.pnl - 51_000)
    study = Reconciliation(problem, tuple(Suspect(problem.field(name)) for name in ("signed_notional", "coupon_bps")))
    report = study.scan()
    print("Common entry mistakes (one field per trial):")
    for match in report.matches:
        print(f"  {'.'.join(match.field.path)}: {match.original} -> {match.proposed}")
        print("    Possible explanations:", "; ".join(mutation.label for mutation in match.mutations))
    for trial in report.unresolved:
        print(f"  Undecided: {'.'.join(trial.field.path)} -> {trial.proposed}: {trial.analysis.reason}")
    if report.baseline.status != "consistent" and not report.matches:
        print("No checked match from common mistakes; infer each suspect independently:")
        for suspect in study.suspects:
            family = study.infer(suspect.field)
            print(f"  {'.'.join(suspect.field.path)}: {family.values(suspect.field.symbol)}")
    print("Add independently observed opening value=22000:")
    opening_observation = Eq(problem.symbols.opening.value, 22_000)
    for match in report.matches:
        checked = match.problem.solutions().where(opening_observation)
        print(f"  {'.'.join(match.field.path)}: {checked.values(match.field.symbol)}")
    print("Each match is a possible explanation to check against the trade record.")

    # Synthetic credit parameters associated with previous/current RED pairs.
    print("Compare source assumptions (synthetic RED parameters):")
    for label, data in (
        ("previous", CreditParameters("DEMO:PAIR:OLD", recovery=Rational(2, 5), survival=Rational(39, 40))),
        ("current", CreditParameters("DEMO:PAIR:CURRENT", recovery=Rational(2, 5), survival=Rational(97, 100))),
    ):
        source_trial = inspect_model(
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
        calculated = source_trial.without("pnl").solutions().one()
        print(f"  {label}: PnL={calculated.pnl}, {source_trial.analyse().status}")
    print("  Only the credit parameters changed; trade terms and trusted PnL stayed fixed.")


if __name__ == "__main__":
    main()

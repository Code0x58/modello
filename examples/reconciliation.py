"""Investigate synthetic CDS breaks from entry errors and stale reference data.

Run ``python -m examples.reconciliation``. The pricing model has one remaining
period: premium on survival and protection on default, both paid at its end.
It omits accrual on default, schedules, curve calibration, and settlement
conventions. See the documentation for the precise scope and sign conventions.
"""

from dataclasses import dataclass, fields, replace
from datetime import date

from sympy import Rational, S

from examples.reconcile import Reconciliation, Suspect
from modello import FieldValue, InstanceDummy, Modello, ModelProblem, ScalarValue, component, inspect_model


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


@dataclass(frozen=True)
class CDSKey:
    """Example market-data key; RED identifiers alone are not a full contract."""

    entity_code: str
    pair_code: str
    currency: str
    definitions: str
    seniority: str
    restructuring: str
    maturity: date


@dataclass(frozen=True)
class CDSParameters:
    """One dated source bundle; identifiers stay outside the symbolic algebra."""

    key: CDSKey
    as_of: date
    survival: ScalarValue
    discount: ScalarValue = 1
    year_fraction: ScalarValue = 1

    def market(self) -> CreditPeriod:
        """Build fresh pricing inputs from this entire bundle."""
        return CreditPeriod(
            "source market", survival=self.survival, discount=self.discount, year_fraction=self.year_fraction
        )


@dataclass(frozen=True)
class ReferenceCase:
    """Confirmed lookup coordinates, loaded inputs, and available source rows."""

    confirmed: CDSKey
    as_of: date
    loaded: CDSParameters
    available: tuple[CDSParameters, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "available", tuple(self.available))

    @property
    def mismatches(self) -> tuple[str, ...]:
        """Compare provenance with independently confirmed lookup coordinates."""
        names = tuple(
            field.name
            for field in fields(CDSKey)
            if getattr(self.loaded.key, field.name) != getattr(self.confirmed, field.name)
        )
        return (*names, "as_of") if self.loaded.as_of != self.as_of else names

    def select(self) -> CDSParameters:
        """Require one complete-key and date match; never select by matching PnL."""
        matches = tuple(row for row in self.available if row.key == self.confirmed and row.as_of == self.as_of)
        if len(matches) != 1:
            raise ValueError(f"Expected one parameter bundle for the confirmed key and date, found {len(matches)}")
        return matches[0]

    def problem(self, parameters: CDSParameters) -> ModelProblem[CDSPnL]:
        """Reprice this fixture with fixed trade terms and a chosen source bundle."""
        return inspect_model(
            CDSPnL,
            "reference-data break",
            opening=CDSMark(
                "opening",
                market=CreditPeriod("opening market", discount=1, survival=Rational(98, 100), year_fraction=1),
            ),
            closing=CDSMark("closing", market=parameters.market()),
            signed_notional=10_000_000,
            coupon_bps=100,
            recovery=Rational(2, 5),
            cashflow=-10_000,
            pnl=51_000,
        )


def stale_reference_case() -> ReferenceCase:
    """Synthetic sovereign: same entity, different pair, and an old cached row.

    DEMO identifiers are placeholders, not licensed RED codes. The new pair is
    supplied by the confirmed reference mapping, not inferred from the PnL.
    A pair change itself has no pricing effect; the stale numerical inputs do.
    """
    confirmed = CDSKey("DEMO:ENTITY", "DEMO:PAIR:CURRENT", "USD", "2014", "senior", "CR", date(2027, 1, 5))
    loaded = CDSParameters(replace(confirmed, pair_code="DEMO:PAIR:OLD"), date(2026, 1, 2), Rational(39, 40))
    current = CDSParameters(confirmed, date(2026, 1, 5), Rational(97, 100))
    return ReferenceCase(confirmed, current.as_of, loaded, (loaded, current))


def main() -> None:
    """Try common errors first; expose inverse inference as a separate stage."""
    pricing = inspect_model(CDSMark, "pricing")
    expression = pricing.solutions().expression(pricing.field("value"))
    print("Pricing formula:", pricing.format(expression))
    problem = sample_problem()
    recorded = problem.without("pnl").solutions().one()
    print("Recorded PnL:", recorded.pnl)
    print("Trusted PnL:", 51_000)
    print("Break (recorded - trusted):", recorded.pnl - 51_000)
    study = Reconciliation(problem, tuple(Suspect(problem.field(name)) for name in ("signed_notional", "coupon_bps")))
    report = study.scan()
    print("Common entry mistakes, one field at a time:")
    for match in report.matches:
        print(f"  {'.'.join(match.field.path)}: {match.original} -> {match.proposed}")
        print("    Possible explanations:", "; ".join(mutation.label for mutation in match.mutations))
    for trial in report.unresolved:
        print(f"  Undecided: {'.'.join(trial.field.path)} -> {trial.proposed}: {trial.analysis.reason}")
    if report.baseline.status != "consistent" and not report.matches:
        print("No confirmed common mutation; infer each suspect independently:")
        for suspect in study.suspects:
            family = study.infer(suspect.field)
            print(f"  {'.'.join(suspect.field.path)}: {family.values(suspect.field)}")
    print("Also requiring independently observed opening PV=22000:")
    opening_value = FieldValue(problem.field("opening", "value"), 22_000)
    for match in report.matches:
        checked = match.problem.solutions().observe(opening_value)
        print(f"  {'.'.join(match.field.path)}: {checked.values(match.field)}")
    print("Alternatives explain the number; confirm any correction against independent records.")

    reference = stale_reference_case()
    stale = reference.problem(reference.loaded).without("pnl").solutions().one()
    selected = reference.select()
    refreshed = reference.problem(selected).solutions().one()
    print("Stale reference-data example (synthetic identifiers):")
    print("  Mismatched source fields:", ", ".join(reference.mismatches))
    print(f"  Loaded: {reference.loaded.key.pair_code}, as of {reference.loaded.as_of}, PnL={stale.pnl}")
    print(f"  Confirmed: {selected.key.pair_code}, as of {selected.as_of}, PnL={refreshed.pnl}")
    print("  Rebuilt the market-input bundle; trade terms and trusted PnL stayed fixed.")


if __name__ == "__main__":
    main()

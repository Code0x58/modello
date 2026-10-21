"""Common-error trials must preserve trust boundaries and their evidence."""

from dataclasses import replace

import pytest
from sympy import Float, Function, Rational, Symbol

from examples.reconcile import DECIMAL_SHIFTS, Reconciliation, Scale, Suspect
from examples.reconciliation import sample_problem
from modello import ConstraintKind, InstanceDummy, Modello, ModelProblem, component, inspect_model


class Terms(Modello):
    quantity = InstanceDummy("quantity", positive=True, integer=True)
    price = InstanceDummy("price", positive=True)


class Trade(Modello):
    terms = component(Terms)
    quantity = terms.quantity
    total = terms.quantity * terms.price


def trade_problem(quantity=2, price=3, total=600):
    return inspect_model(Trade, "check", terms=Terms("terms", quantity=quantity, price=price), total=total)


def test_cds_common_error_scan_does_not_use_inverse_solution_families(monkeypatch):
    problem = sample_problem()
    study = Reconciliation(problem, (Suspect(problem.field("signed_notional")), Suspect(problem.field("coupon_bps"))))

    def forbidden(*args, **kwargs):
        raise AssertionError("The common-mutation stage must not call solutions()")

    monkeypatch.setattr(ModelProblem, "solutions", forbidden)
    report = study.scan()
    assert report.baseline.status == "inconsistent"
    assert len(report.trials) == 8
    assert not report.unresolved
    (match,) = report.matches
    assert (match.field.path, match.original, match.proposed) == (("signed_notional",), 1_000_000, 10_000_000)
    assert match.analysis.witness is not None
    assert match.analysis.witness.verdict == "satisfied"
    assert match.analysis.witness.candidate.value(problem.field("pnl")) == 51_000


def test_nested_scalar_trial_keeps_sibling_observations_and_source_snapshot():
    problem = trade_problem()
    quantity = problem.field("terms", "quantity")
    study = Reconciliation(problem, (Suspect(quantity),))
    report = study.scan()
    (match,) = report.matches
    assert (match.original, match.proposed) == (2, 200)
    original = next(c for c in problem.constraints if c.kind is ConstraintKind.INPUT and c.left == quantity.symbol)
    for before, after in zip(problem.constraints, match.problem.constraints, strict=True):
        assert after == (replace(before, right=200) if before is original else before)
    assert original.right == 2
    case = match.problem.solutions().one()
    assert isinstance(case, Trade) and isinstance(case.terms, Terms)
    assert (case.terms.quantity, case.terms.price, case.total) == (200, 3, 600)
    assert study.infer(quantity).one().terms.quantity == 200


def test_equivalent_cause_labels_share_one_checked_trial():
    problem = trade_problem(quantity=10, price=1200, total=120)
    price = problem.field("terms", "price")
    currency = Scale("GBX quoted as GBP", Rational(1, 100))
    study = Reconciliation(problem, (Suspect(price, (*DECIMAL_SHIFTS, currency, currency)),))
    report = study.scan()
    assert len(report.trials) == 4
    (match,) = report.matches
    assert match.proposed == 12
    assert [mutation.label for mutation in match.mutations] == ["two extra zeros (/100)", "GBX quoted as GBP"]


def test_all_matching_fields_are_reported_without_selecting_a_cause():
    problem = trade_problem()
    fields = (problem.field("terms", "quantity"), problem.field("terms", "price"))
    report = Reconciliation(problem, tuple(Suspect(field) for field in fields)).scan()
    assert [(trial.field, trial.proposed) for trial in report.matches] == [(fields[0], 200), (fields[1], 300)]


def test_lot_size_is_a_field_specific_hypothesis():
    problem = trade_problem(total=150)
    quantity, price = problem.field("quantity"), problem.field("terms", "price")
    study = Reconciliation(
        problem, (Suspect(quantity, (*DECIMAL_SHIFTS, Scale("25 items per lot", 25))), Suspect(price))
    )
    (match,) = study.scan().matches
    assert (match.field, match.proposed) == (quantity, 50)
    assert match.mutations == (Scale("25 items per lot", 25),)


def test_inverse_inference_is_an_explicit_second_stage():
    problem = trade_problem(total=21)
    quantity = problem.field("terms", "quantity")
    study = Reconciliation(problem, (Suspect(quantity),))
    report = study.scan()
    assert not report.matches and not report.unresolved
    assert study.infer(quantity).one().terms.quantity == 7
    with pytest.raises(ValueError, match="declared suspect"):
        study.infer(problem.field("total"))


def test_already_reconciled_baseline_does_not_suggest_mutations():
    problem = trade_problem(total=6)
    report = Reconciliation(problem, (Suspect(problem.field("quantity")),)).scan()
    assert report.baseline.status == "consistent"
    assert report.trials == ()


def test_domain_violating_mutations_are_rejected():
    # Dividing quantity by ten fits the amount but violates integer units.
    problem = trade_problem(total=Rational(3, 5))
    report = Reconciliation(problem, (Suspect(problem.field("quantity")),)).scan()
    assert not report.matches
    assert all(trial.analysis.status == "inconsistent" for trial in report.trials)
    (fractional,) = (trial for trial in report.trials if trial.proposed == Rational(1, 5))
    assert any(
        check.predicate == "integer" and check.actual is False for check in fractional.analysis.deductions.assumptions
    )


def test_copied_inferred_values_are_not_silently_discarded():
    class PricedTerms(Terms):
        amount = Terms.quantity * Terms.price

    class Check(Modello):
        terms = component(PricedTerms)
        total = terms.amount

    problem = inspect_model(Check, "copied result", terms=PricedTerms("priced", quantity=10, price=1200), total=120)
    price = problem.field("terms", "price")
    study = Reconciliation(problem, (Suspect(price),))
    report = study.scan()
    assert not report.matches and not report.unresolved
    assert study.infer(price).is_empty is True
    inferred = tuple(c for c in problem.constraints if c.kind is ConstraintKind.INFERRED)
    assert inferred
    assert all(c in trial.problem.constraints for trial in report.trials for c in inferred)


def test_zero_values_do_not_produce_noop_trials():
    class Reading(Modello):
        x = InstanceDummy("x", real=True)
        value = x + 1

    problem = inspect_model(Reading, "zero", x=0, value=3)
    study = Reconciliation(problem, (Suspect(problem.field("x")),))
    report = study.scan()
    assert report.baseline.status == "inconsistent"
    assert report.trials == ()
    assert study.infer(problem.field("x")).one().x == 2


def test_sign_reversal_requires_an_explicit_hypothesis():
    class Signed(Modello):
        quantity = InstanceDummy("quantity", real=True)
        value = quantity * 3

    problem = inspect_model(Signed, "side", quantity=-2, value=6)
    quantity = problem.field("quantity")
    assert not Reconciliation(problem, (Suspect(quantity),)).scan().matches
    (match,) = Reconciliation(problem, (Suspect(quantity, (Scale("opposite side", -1),)),)).scan().matches
    assert match.proposed == 2


def test_undecided_mathematics_is_retained_as_unknown():
    opaque = Function("opaque")

    class Reading(Modello):
        x = InstanceDummy("x", real=True)
        value = opaque(x)

    problem = inspect_model(Reading, "opaque", x=1, value=2)
    report = Reconciliation(problem, (Suspect(problem.field("x"), (Scale("x10", 10),)),)).scan()
    assert report.baseline.status == "unknown"
    assert not report.matches
    assert report.unresolved == report.trials
    assert len(report.unresolved) == 1


def test_single_field_trials_do_not_combine_independent_errors():
    class Independent(Modello):
        quantity = InstanceDummy("quantity", positive=True)
        price = InstanceDummy("price", positive=True)
        weight = 2 * quantity
        amount = 3 * price

    problem = inspect_model(Independent, "two errors", quantity=2, price=3, weight=40, amount=90)
    fields = (problem.field("quantity"), problem.field("price"))
    report = Reconciliation(problem, tuple(Suspect(field) for field in fields)).scan()
    assert not report.matches and not report.unresolved


def test_foreign_fields_alias_duplicates_and_unobserved_fields_are_rejected():
    problem = trade_problem()
    other = trade_problem()
    with pytest.raises(ValueError, match="actual field"):
        Reconciliation(problem, (Suspect(other.field("quantity")),))
    with pytest.raises(ValueError, match="once"):
        Reconciliation(problem, (Suspect(problem.field("quantity")), Suspect(problem.field("terms", "quantity"))))
    symbolic = inspect_model(Trade, "no observations")
    with pytest.raises(ValueError, match="exactly one"):
        Reconciliation(symbolic, (Suspect(symbolic.field("quantity")),))


def test_ambiguous_alias_observations_are_rejected():
    problem = trade_problem()
    field = problem.field("quantity")
    original = next(c for c in problem.constraints if c.kind is ConstraintKind.INPUT and c.left == field.symbol)
    repeated = replace(problem, constraints=(*problem.constraints, replace(original, index=999)))
    with pytest.raises(ValueError, match="exactly one"):
        Reconciliation(repeated, (Suspect(field),))


@pytest.mark.parametrize("value", [Float("2.0"), Symbol("external"), "oo", "I"])
def test_suspect_observation_must_be_exact_finite_real_and_closed(value):
    class Reading(Modello):
        x = InstanceDummy("x", real=True)

    problem = inspect_model(Reading, "invalid observation", x=value)
    with pytest.raises(ValueError, match="exact, finite real"):
        Reconciliation(problem, (Suspect(problem.field("x")),))


@pytest.mark.parametrize("factor", [0, 1, 0.1, "oo", Symbol("factor"), [10]])
def test_scale_requires_an_exact_nontrivial_rational(factor):
    with pytest.raises(ValueError, match="exact rational"):
        Scale("invalid", factor)

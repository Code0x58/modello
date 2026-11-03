"""One record-review study: exact calculus, reusable premises, and exported evidence."""

import json
from pathlib import Path

import pytest
from sympy import Rational, log, simplify, sqrt

from examples.record_review import (
    CappedReview,
    EntryBudget,
    ExponentialReview,
    RationalReview,
    ReviewPlanning,
    review_context,
    review_plan,
    write_reports,
)
from modello.contrib.verification import VerificationContext, VerificationError, predicate


@pytest.fixture(scope="module")
def measured() -> VerificationContext[ReviewPlanning]:
    return review_context()


@pytest.fixture(scope="module")
def proposed(measured: VerificationContext[ReviewPlanning]) -> VerificationContext[ReviewPlanning]:
    return measured.observe(candidate=2 * log(3))


def test_definition_reports_distinguish_parameter_conditions_and_regularity():
    definition = ExponentialReview.definition_report()
    assert definition.explain("value/strictly_increasing").verdict == "conditionally true"
    assert definition.explain("value/concave").verdict == "strictly true"
    assert definition.explain("value/continuously_differentiable").verdict == "strictly true"
    context = VerificationContext.for_model(ExponentialReview)
    positive_rate = context.assume(context.symbols.rate > 0)
    assert positive_rate.explain("value/strictly_increasing").verdict == "strictly true"
    assert context.observe(rate=0).explain("value/strictly_increasing").verdict == "strictly false"
    assert context.explain("value/strictly_increasing").verdict == "conditionally true"


def test_stationary_candidate_has_a_global_certificate_and_local_curvature(proposed):
    result = proposed.run(review_plan)
    decision = result.data
    assert simplify(decision.effort - 2 * log(3)) == 0
    assert decision.net_saving == 4 - 2 * log(3)
    assert decision.minutes_per_record == 6 + 2 * log(3)
    assert decision.workload_curvature == Rational(1, 2)
    maximum = result.report.explain("global_maximum")
    assert maximum.verdict == "strictly true"
    assert {item.obligation.name for item in maximum.premises} == {
        "concavity",
        "differentiability",
        "stationarity",
        "feasibility",
    }
    assert all(item.verdict == "strictly true" for item in maximum.premises)
    # Independent values of the workload on either side of the proposed point.
    workload = proposed.expression("expected_minutes", wrt="effort")
    axis = ReviewPlanning.effort
    assert workload.subs(axis, 2) > decision.minutes_per_record
    assert workload.subs(axis, Rational(5, 2)) > decision.minutes_per_record


def test_capacity_and_staff_budget_can_be_supplied_in_either_direction(proposed):
    decision = proposed.run(review_plan).data
    day = EntryBudget("day", staff_minutes=480, minutes_per_record=decision.minutes_per_record)
    target = EntryBudget("target", records=60, minutes_per_record=decision.minutes_per_record)
    assert simplify(day.records - 480 / (6 + 2 * log(3))) == 0
    assert target.staff_minutes == 360 + 120 * log(3)


def test_a_wrong_candidate_and_an_unregistered_rule_do_not_establish_an_optimum(measured):
    wrong = measured.observe(candidate=1)
    with pytest.raises(VerificationError) as error:
        wrong.run(review_plan)
    assert error.value.context is wrong
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    assert error.value.report.explain("global_maximum").verdict == "unresolved"
    unregistered = VerificationContext.for_model(ReviewPlanning).observe(candidate=1)
    with pytest.raises(VerificationError) as error:
        unregistered.run(review_plan)
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_replacement_rechecks_the_candidate_and_preserves_the_baseline(measured, proposed):
    stale = proposed.replace("review", RationalReview)
    with pytest.raises(VerificationError) as error:
        stale.run(review_plan)
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    changed = measured.replace("review", RationalReview).observe(candidate=2 * (sqrt(3) - 1))
    decision = changed.run(review_plan).data
    assert decision.net_saving == 8 - 4 * sqrt(3)
    assert decision.minutes_per_record == 2 + 4 * sqrt(3)
    assert decision.workload_curvature == sqrt(3) / 3
    assert measured.expression("review.rate") == Rational(1, 2)
    assert proposed.run(review_plan).data.minutes_per_record == 6 + 2 * log(3)


def test_the_capped_curve_retains_its_claims_and_exposes_its_kink(measured):
    capped = measured.replace("review", CappedReview).observe(candidate=2)
    with pytest.raises(VerificationError) as error:
        capped.run(review_plan)
    regularity = error.value.report.explain("review.value/continuously_differentiable")
    assert regularity.verdict == "strictly false"
    assert dict(regularity.witness)[ReviewPlanning.effort] == 2
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_retiring_an_exploratory_saving_target_does_not_waive_the_method_contract(proposed):
    target = proposed.with_obligations(
        predicate(proposed.symbols.candidate_saving >= 2, name="saving_target", consumer="review_plan")
    )
    with pytest.raises(VerificationError) as error:
        target.run(review_plan)
    assert error.value.report.explain("saving_target").verdict == "strictly false"
    baseline = target.without_obligations("saving_target", reason="Keep the measured baseline for comparison.")
    assert baseline.run(review_plan).report.explain("global_maximum").verdict == "strictly true"
    assert target.explain("saving_target").verdict == "strictly false"
    assert "Keep the measured baseline" in baseline.changes[-1]


def test_reports_include_exact_results_and_the_blocked_replacement(tmp_path: Path):
    write_reports(tmp_path, plot=True)
    summary = json.loads((tmp_path / "results.json").read_text())
    assert summary["inputs"] == {"entry_minutes": "4", "avoidable_minutes": "6", "review.rate": "1/2"}
    assert summary["plans"]["exponential"]["minutes_per_record"]["exact"] == "2*log(3) + 6"
    assert summary["plans"]["rational"]["minutes_per_record"]["decimal"] > 8.9
    assert "strictly false" in summary["plans"]["capped"]["blocked"]
    for name in ("exponential", "rational", "capped"):
        assert json.loads((tmp_path / f"{name}.json").read_text())["evidence"]
        assert "<!doctype html>" in (tmp_path / f"{name}.html").read_text()
    assert "Changing the response curve" in (tmp_path / "curves.svg").read_text()

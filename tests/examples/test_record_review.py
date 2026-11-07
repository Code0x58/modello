"""One record-review study: exact calculus, reusable premises, and exported evidence."""

import json
from pathlib import Path

import pytest
from sympy import FiniteSet, Lambda, Matrix, Rational, S, Symbol, exp, limit, log, simplify, sqrt

from examples.record_review import (
    CappedReview,
    CheckingStep,
    ContinuousReview,
    EntryBudget,
    ExponentialReview,
    RationalReview,
    RecordWork,
    RepeatedReview,
    ReviewPlanning,
    ReviewSpecification,
    derive_checking_response,
    derive_continuous_transitions,
    review_context,
    review_plan,
    transition_recurrence,
    write_reports,
)
from modello.contrib.verification import VerificationContext, VerificationError, predicate


@pytest.fixture(scope="module")
def measured() -> VerificationContext[ReviewPlanning, ReviewSpecification]:
    return review_context()


@pytest.fixture(scope="module")
def proposed(
    measured: VerificationContext[ReviewPlanning, ReviewSpecification],
) -> VerificationContext[ReviewPlanning, ReviewSpecification]:
    return measured.observe(**{"candidate.review.response.minutes_per_record": 2 * log(3)})


def test_short_checks_derive_the_continuous_response():
    short = CheckingStep(
        "short",
        record=RecordWork(
            "record", entry_minutes_per_record=4, error_fraction=Rational(1, 10), repair_minutes_per_error=60
        ),
        minutes_per_record=Rational(1, 10),
        detection_rate=Rational(1, 2),
    )
    assert short.caught_share == Rational(1, 20)
    assert 1000 * short.remaining_error_fraction == 95
    assert short.record.repair_minutes_per_record == 6
    assert short.repair_minutes_per_record == Rational(57, 10)

    # Use the same name as the helper's bound variable to exercise its scope.
    minutes_per_record = Symbol("N", integer=True, positive=True)
    detection_rate = Symbol("k", real=True)
    response = derive_checking_response(minutes_per_record, detection_rate)
    assert response == 1 - exp(-detection_rate * minutes_per_record)
    assert response.subs(minutes_per_record, 0) == 0
    assert response.subs(detection_rate, 0) == 0
    context = VerificationContext.for_model(ExponentialReview)
    assert context.expression("caught_share", wrt="minutes_per_record") == derive_checking_response(
        context.symbols.minutes_per_record, context.symbols.detection_rate
    )
    review = ExponentialReview("observed", minutes_per_record=2, detection_rate=Rational(1, 2))
    assert review.caught_share == 1 - exp(-1)


def test_definition_reports_distinguish_parameter_conditions_and_regularity():
    definition = ExponentialReview.definition_report()
    assert definition.explain("caught_share/strictly_increasing").verdict == "conditionally true"
    assert definition.explain("caught_share/concave").verdict == "strictly true"
    assert definition.explain("caught_share/continuously_differentiable").verdict == "strictly true"
    context = VerificationContext.for_model(ExponentialReview)
    positive_detection_rate = context.assume(context.symbols.detection_rate > 0)
    assert positive_detection_rate.explain("caught_share/strictly_increasing").verdict == "strictly true"
    assert context.observe(detection_rate=0).explain("caught_share/strictly_increasing").verdict == "strictly false"
    assert context.explain("caught_share/strictly_increasing").verdict == "conditionally true"


def test_outcome_fractions_conserve_records_and_follow_response_replacements(measured):
    paths = ("correct_to_correct_fraction", "corrected_fraction", "remaining_error_fraction")
    axis = measured.symbols.candidate.review.response.minutes_per_record
    outcomes = tuple(measured.expression(f"candidate.review.{path}", wrt=axis) for path in paths)
    assert simplify(sum(outcomes)) == 1
    assert tuple(value.subs(axis, 0) for value in outcomes) == (Rational(9, 10), 0, Rational(1, 10))
    assert tuple(limit(value, axis, S.Infinity) for value in outcomes) == (Rational(9, 10), Rational(1, 10), 0)

    changed = measured.replace("candidate.review.response", RationalReview).observe(
        **{"candidate.review.response.minutes_per_record": 2}
    )
    assert tuple(changed.expression(f"candidate.review.{path}") for path in paths) == (
        Rational(9, 10),
        Rational(1, 20),
        Rational(1, 20),
    )
    assert changed.expression("candidate.minutes_per_record") == 9


def test_stationary_candidate_has_a_global_certificate_and_local_curvature(proposed):
    result = proposed.run(review_plan)
    decision = result.data
    assert simplify(decision.checking_minutes_per_record - 2 * log(3)) == 0
    assert decision.minutes_saved_per_record == 4 - 2 * log(3)
    assert decision.minutes_per_record == 6 + 2 * log(3)
    assert decision.minutes_per_record_second_derivative == Rational(1, 2)
    maximum = result.report.explain("global_maximum")
    assert maximum.verdict == "strictly true"
    premises = {item.obligation.name: item for item in maximum.premises}
    assert set(premises) == {
        "concavity",
        "differentiability",
        "stationarity",
        "feasibility",
    }
    assert all(item.verdict == "strictly true" for item in premises.values())
    assert simplify(maximum.expression - decision.minutes_saved_per_record) == 0
    # The rule checks the whole curve despite the observed checking time;
    # stationarity is checked at that observed point instead.
    axis = proposed.symbols.candidate.review.response.minutes_per_record
    for name in ("concavity", "differentiability"):
        assert premises[name].expression.free_symbols == {axis}
        assert premises[name].coordinate == axis
    assert premises["stationarity"].expression.doit() == 0
    # Independent values of the workload on either side of the proposed point.
    workload = proposed.expression(
        "candidate.minutes_per_record", wrt="candidate.review.response.minutes_per_record"
    ).doit()
    assert workload.subs(axis, 2) > decision.minutes_per_record
    assert workload.subs(axis, Rational(5, 2)) > decision.minutes_per_record


def test_capacity_and_staff_budget_can_be_supplied_in_either_direction(proposed):
    day = proposed.observe(budget=EntryBudget("day", staff_minutes=480))
    target = proposed.observe(budget=EntryBudget("target", records=60))
    assert day.run(review_plan).report.explain("global_maximum").verdict == "strictly true"
    assert target.run(review_plan).data == proposed.run(review_plan).data
    assert day.family.values(day.symbols.budget.records) == FiniteSet(simplify(480 / (6 + 2 * log(3))))
    assert target.family.values(target.symbols.budget.staff_minutes) == FiniteSet(360 + 120 * log(3))


def test_revised_inputs_recheck_the_optimum_and_its_derivative_domain(measured):
    record = RecordWork(
        "revised", entry_minutes_per_record=3, error_fraction=Rational(1, 5), repair_minutes_per_record=8
    )
    response = ExponentialReview("response", detection_rate=Rational(1, 4), error_creation_rate=0)
    revised = review_context(record, response=response)
    result = revised.observe(**{"candidate.review.response.minutes_per_record": 4 * log(2)}).run(review_plan)
    assert result.report.explain("global_maximum").verdict == "strictly true"
    assert simplify(result.data.checking_minutes_per_record - 4 * log(2)) == 0
    assert simplify(result.data.minutes_per_record - 7 - 4 * log(2)) == 0
    assert result.data.minutes_per_record_second_derivative == Rational(1, 4)
    assert measured.expression("candidate.review.record.repair_minutes_per_record") == 6
    assert revised.expression("candidate.review.record.repair_minutes_per_error") == 40

    # Formal substitution at zero does not establish a two-sided derivative
    # within the coordinate's nonnegative domain.
    endpoint_record = RecordWork(
        "endpoint", entry_minutes_per_record=3, error_fraction=Rational(1, 5), repair_minutes_per_record=4
    )
    endpoint = review_context(endpoint_record, response=response).observe(
        **{"candidate.review.response.minutes_per_record": 0}
    )
    assert endpoint.expression("candidate.minutes_saved_derivative").doit() == 0
    with pytest.raises(VerificationError) as error:
        endpoint.run(review_plan)
    assert error.value.report.explain("global_maximum/stationarity").verdict == "unresolved"


def test_a_wrong_candidate_and_an_unregistered_rule_do_not_establish_an_optimum(measured):
    wrong = measured.observe(**{"candidate.review.response.minutes_per_record": 1})
    with pytest.raises(VerificationError) as error:
        wrong.run(review_plan)
    assert error.value.context is wrong
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    assert error.value.report.explain("global_maximum").verdict == "unresolved"
    unregistered = VerificationContext.for_model(ReviewPlanning).observe(
        **{"candidate.review.response.minutes_per_record": 1}
    )
    with pytest.raises(VerificationError) as error:
        unregistered.run(review_plan)
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_replacement_rechecks_the_candidate_and_preserves_the_baseline(measured, proposed):
    stale = proposed.replace("candidate.review.response", RationalReview)
    with pytest.raises(VerificationError) as error:
        stale.run(review_plan)
    assert error.value.report.explain("global_maximum/stationarity").verdict == "strictly false"
    changed = measured.replace("candidate.review.response", RationalReview).observe(
        **{"candidate.review.response.minutes_per_record": 2 * (sqrt(3) - 1)}
    )
    decision = changed.run(review_plan).data
    assert decision.minutes_saved_per_record == 8 - 4 * sqrt(3)
    assert decision.minutes_per_record == 2 + 4 * sqrt(3)
    assert decision.minutes_per_record_second_derivative == sqrt(3) / 3
    supplied = review_context(
        response=RationalReview("response", detection_rate=Rational(1, 2), error_creation_rate=0)
    ).observe(**{"candidate.review.response.minutes_per_record": 2 * (sqrt(3) - 1)})
    assert supplied.run(review_plan).data == decision
    assert measured.expression("candidate.review.response.detection_rate") == Rational(1, 2)
    assert proposed.run(review_plan).data.minutes_per_record == 6 + 2 * log(3)


def test_the_capped_curve_retains_its_claims_and_exposes_its_kink(measured):
    capped = measured.replace("candidate.review.response", CappedReview).observe(
        **{"candidate.review.response.minutes_per_record": 2}
    )
    with pytest.raises(VerificationError) as error:
        capped.run(review_plan)
    regularity = error.value.report.explain("candidate.review.response.caught_share/continuously_differentiable")
    assert regularity.verdict == "strictly false"
    assert dict(regularity.witness)[capped.symbols.candidate.review.response.minutes_per_record] == 2
    assert error.value.report.explain("global_maximum").verdict == "unresolved"


def test_transition_recurrence_evolves_the_initial_composition():
    initial = Matrix([[Rational(9, 10), Rational(1, 10)]])
    for P in (
        Matrix.eye(2),
        Matrix([[0, 1], [1, 0]]),
        Matrix([[1, 0], [Rational(1, 2), Rational(1, 2)]]),
        Matrix([[Rational(3, 4), Rational(1, 4)], [Rational(1, 2), Rational(1, 2)]]),
    ):
        law = transition_recurrence(P, initial[0, 1])
        for count in (0, 1, 5):
            assert law.at(count).doit() == (initial * P**count)[0, 1]
    with pytest.raises(ValueError, match="two rows and two columns"):
        transition_recurrence(Matrix.eye(3), S.Zero)


@pytest.fixture(scope="module")
def continuous_transitions():
    minutes, correction, creation = (Symbol(name, nonnegative=True) for name in ("t", "k", "r"))
    return Lambda((minutes, correction, creation), derive_continuous_transitions(minutes, correction, creation))


@pytest.mark.parametrize(
    ("correction", "creation"),
    [(0, 0), (Rational(1, 2), 0), (0, Rational(1, 100)), (Rational(1, 2), Rational(1, 100))],
)
def test_continuous_matrix_matches_the_generator_including_zero_rates(continuous_transitions, correction, creation):
    generator = Matrix([[-creation, creation], [correction, -correction]])
    for minutes in (0, 2):
        P = continuous_transitions(minutes, correction, creation)
        assert (P - (minutes * generator).exp()).applyfunc(simplify) == Matrix.zeros(2)
        assert (P * Matrix.ones(2, 1)).applyfunc(simplify) == Matrix.ones(2, 1)


def test_matrix_response_reuses_workload_contracts_and_inferred_pass_counts(proposed):
    recovered = proposed.replace("candidate.review.response", ContinuousReview)
    assert recovered.run(review_plan).data == proposed.run(review_plan).data
    response = ContinuousReview("response", detection_rate=Rational(1, 2), error_creation_rate=Rational(1, 100))
    changed = review_context(response=response).observe(**{"candidate.review.response.minutes_per_record": 2})
    expected = Rational(1, 51) + (Rational(1, 10) - Rational(1, 51)) * exp(-Rational(51, 50))
    assert simplify(changed.expression("candidate.review.remaining_error_fraction") - expected) == 0
    assert (
        changed.explain("candidate.review.response.caught_share/continuously_differentiable").verdict == "strictly true"
    )
    record = RecordWork(
        "record", entry_minutes_per_record=4, error_fraction=Rational(1, 10), repair_minutes_per_error=60
    )
    original = ExponentialReview(
        "response", minutes_per_record=2, detection_rate=Rational(1, 2), error_creation_rate=Rational(1, 100)
    )
    P = Matrix(
        [
            [1 - original.introduced_error_share, original.introduced_error_share],
            [original.caught_share, 1 - original.caught_share],
        ]
    )
    for count in (0, 2):
        repeated = RepeatedReview(
            "whole passes",
            **{"review.record": record, "review.response": original},
            checking_minutes_per_record=2 * count,
        )
        assert repeated.passes == count
        expected_errors = (Matrix([[1 - record.error_fraction, record.error_fraction]]) * P**count)[0, 1]
        assert simplify(repeated.final_error_fraction - expected_errors) == 0


def test_retiring_an_exploratory_saving_target_does_not_waive_the_method_contract(proposed):
    target = proposed.with_obligations(
        predicate(
            proposed.symbols.candidate.minutes_saved_per_record >= 2, name="saving_target", consumer="review_plan"
        )
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
    assert summary["inputs"] == {
        "candidate.review.record.entry_minutes_per_record": "4",
        "candidate.review.record.error_fraction": "1/10",
        "candidate.review.record.repair_minutes_per_error": "60",
        "candidate.review.response.detection_rate": "1/2",
        "candidate.review.response.error_creation_rate": "0",
    }
    assert summary["plans"]["exponential"]["minutes_per_record"]["exact"] == "2*log(3) + 6"
    assert summary["plans"]["rational"]["minutes_per_record"]["decimal"] > 8.9
    assert "strictly false" in summary["plans"]["capped"]["blocked"]
    for name in ("exponential", "rational", "capped"):
        assert json.loads((tmp_path / f"{name}.json").read_text())["evidence"]
        assert "<!doctype html>" in (tmp_path / f"{name}.html").read_text()
    assert "Why spend time checking?" in (tmp_path / "benefit.svg").read_text()
    assert "Changing the response curve" in (tmp_path / "responses.svg").read_text()

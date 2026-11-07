"""Verify proposed checking times for digitised records and compare their workloads.

Run ``python -m examples.record_review --output artefacts/record-review``.
Add ``--plot`` with ``modello[numerics]`` installed to draw the response curves.
The worked explanation is in ``docs/examples/record-review/index.rst``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, fields
from importlib import metadata
from pathlib import Path

from sympy import (
    Derivative,
    Dummy,
    Eq,
    Expr,
    ImmutableMatrix,
    Interval,
    Lambda,
    Piecewise,
    Rational,
    S,
    lambdify,
    limit,
    log,
    simplify,
    sqrt,
)
from sympy.matrices import MatrixBase

from modello import Index, InstanceDummy, Modello, Recurrence, Series, component
from modello.contrib.verification import (
    InferenceRule,
    Method,
    Obligation,
    VerificationContext,
    VerificationError,
    VerificationModel,
    concave,
    continuously_differentiable,
    equal_to,
    nonnegative,
    positive,
    specifies,
    strictly_increasing,
)


class EntryBudget(Modello):
    """Staff minutes = records * minutes per record; supply any two quantities."""

    records = InstanceDummy("n", positive=True)
    minutes_per_record = InstanceDummy("m", positive=True)
    staff_minutes = records * minutes_per_record


class RecordWork(Modello):
    """Entry and expected repair minutes per record before checking."""

    entry_minutes_per_record = InstanceDummy("b", positive=True)
    error_fraction = InstanceDummy("p", nonnegative=True)
    repair_minutes_per_error = InstanceDummy("L", positive=True)
    repair_minutes_per_record = error_fraction * repair_minutes_per_error
    minutes_per_record = entry_minutes_per_record + repair_minutes_per_record


class CheckingStep(Modello):
    """Approximate a short check that catches a share of the errors still left.

    Physical inputs need record.error_fraction <= 1 and 0 <= detection_rate * minutes_per_record <= 1.
    These are modelling assumptions here, not automatically checked bounds.
    """

    record = component(RecordWork)
    minutes_per_record = InstanceDummy("h", nonnegative=True)
    detection_rate = InstanceDummy("k", real=True)
    caught_share = detection_rate * minutes_per_record
    remaining_share = 1 - caught_share
    remaining_error_fraction = record.error_fraction * remaining_share
    repair_minutes_per_record = remaining_error_fraction * record.repair_minutes_per_error


def derive_checking_response(minutes_per_record: Expr, detection_rate: Expr) -> Expr:
    """Derive the caught share from the checking minutes per record and detection rate."""
    steps = Dummy("N", integer=True, positive=True)
    step = CheckingStep("part", minutes_per_record=minutes_per_record / steps, detection_rate=detection_rate)
    remaining_share = limit(step.remaining_share**steps, steps, S.Infinity)
    return 1 - remaining_share


CHECKING_MINUTES_DOMAIN = Interval(0, S.Infinity)


class ExponentialReview(VerificationModel):
    """Constant-rate correction and error-introduction responses.

    ``minutes_per_record`` measures checking; ``detection_rate`` is per minute.
    The detection rate is deliberately real: increasing response needs it > 0.
    The nonnegative error-creation rate is zero in the baseline scenario.
    The curve follows from a constant detection rate for errors still left;
    that assumption has not been fitted or validated against a real team.
    """

    minutes_per_record = InstanceDummy("t", nonnegative=True)
    detection_rate = InstanceDummy("k", real=True)
    error_creation_rate = InstanceDummy("r", nonnegative=True)
    introduced_error_share = derive_checking_response(minutes_per_record, error_creation_rate)
    caught_share = specifies(
        derive_checking_response(minutes_per_record, detection_rate),
        strictly_increasing(
            "caught_share",
            wrt="minutes_per_record",
            domain=CHECKING_MINUTES_DOMAIN,
            reason="Extra checking must catch a larger fraction of the missed errors.",
        ),
        concave(
            "caught_share",
            wrt="minutes_per_record",
            domain=CHECKING_MINUTES_DOMAIN,
            reason="The marginal benefit must not increase.",
        ),
        continuously_differentiable(
            "caught_share",
            wrt="minutes_per_record",
            domain=CHECKING_MINUTES_DOMAIN,
            order=2,
            reason="The plan approximates total minutes per record near the chosen checking minutes to second order.",
        ),
    )


class ReviewModel(Modello):
    """Outcome fractions for one pass, shared by workload and repeated review."""

    record = component(RecordWork)
    response = component(ExponentialReview)
    corrected_fraction = record.error_fraction * response.caught_share
    introduced_error_fraction = (1 - record.error_fraction) * response.introduced_error_share
    correct_to_correct_fraction = 1 - record.error_fraction - introduced_error_fraction
    remaining_error_fraction = record.error_fraction - corrected_fraction + introduced_error_fraction


class ReviewWorkload(Modello):
    """Staff minutes and derivatives of the shared record outcomes."""

    review = component(ReviewModel)
    minutes_saved_per_record = (
        review.corrected_fraction - review.introduced_error_fraction
    ) * review.record.repair_minutes_per_error - review.response.minutes_per_record
    minutes_per_record = review.record.minutes_per_record - minutes_saved_per_record
    # A deferred derivative acts on the resolved curve, rather than the field's
    # placeholder symbol. It also generates a differentiability obligation.
    minutes_saved_derivative = Derivative(minutes_saved_per_record, review.response.minutes_per_record, evaluate=False)
    minutes_per_record_second_derivative = Derivative(
        minutes_per_record, review.response.minutes_per_record, 2, evaluate=False
    )


class ReviewPlanning(VerificationModel):
    """A proposed review workload linked to a staffing budget."""

    candidate = component(ReviewWorkload)
    budget = component(EntryBudget)
    minutes_per_record = budget.minutes_per_record
    minutes_per_record = candidate.minutes_per_record


GLOBAL_MAXIMUM = Obligation(
    "global_maximum",
    "candidate.minutes_saved_per_record",
    "rule",
    domain=CHECKING_MINUTES_DOMAIN,
    reason="The chosen checking minutes maximize minutes saved per record over the nonnegative checking domain.",
)

STATIONARY_MAXIMUM = InferenceRule(
    "concave_stationary_maximum",
    GLOBAL_MAXIMUM,
    (
        concave(
            "candidate.minutes_saved_per_record",
            wrt="candidate.review.response.minutes_per_record",
            domain=CHECKING_MINUTES_DOMAIN,
            name="concavity",
        ),
        continuously_differentiable(
            "candidate.minutes_saved_per_record",
            wrt="candidate.review.response.minutes_per_record",
            domain=CHECKING_MINUTES_DOMAIN,
            order=1,
            name="differentiability",
        ),
        equal_to("candidate.minutes_saved_derivative", 0, name="stationarity"),
        nonnegative("candidate.review.response.minutes_per_record", name="feasibility"),
    ),
    source="Concave supporting-line inequality; docs/examples/record-review/index.rst#why-this-is-a-global-optimum",
)


@dataclass(frozen=True)
class ReviewDecision:
    """Checking, saved, and total minutes per record, plus the total's second derivative."""

    checking_minutes_per_record: Expr
    minutes_saved_per_record: Expr
    minutes_per_record: Expr
    minutes_per_record_second_derivative: Expr


def _evaluate_plan(current: VerificationContext[ReviewPlanning]) -> ReviewDecision:
    """Evaluate the proposed plan after its Method has established the premises."""
    candidate = current.model.candidate
    values = tuple(
        simplify(value.doit())
        for value in (
            candidate.review.response.minutes_per_record,
            candidate.minutes_saved_per_record,
            candidate.minutes_per_record,
            candidate.minutes_per_record_second_derivative,
        )
    )
    if any(value.free_symbols or value.is_finite is not True or value.is_real is not True for value in values):
        raise ValueError("Observe all parameters and checking minutes per record before evaluating a plan")
    return ReviewDecision(*values)


review_plan = Method(
    "review_plan",
    _evaluate_plan,
    preconditions=(GLOBAL_MAXIMUM, positive("candidate.minutes_per_record", name="positive_workload")),
    uses=(
        "candidate.review.response.minutes_per_record",
        "candidate.minutes_saved_per_record",
        "candidate.minutes_per_record",
        "candidate.minutes_per_record_second_derivative",
    ),
)


def review_context(
    record: RecordWork | None = None, *, response: ExponentialReview | None = None
) -> VerificationContext[ReviewPlanning]:
    """Start a fresh context with these inputs and the stationary-point rule.

    Supply record estimates and a response component, or use the baseline
    with no introduced errors. Checking minutes and the staffing budget stay open.
    """
    if record is None:
        record = RecordWork(
            "record", entry_minutes_per_record=4, error_fraction=Rational(1, 10), repair_minutes_per_error=60
        )
    if response is None:
        response = ExponentialReview("response", detection_rate=Rational(1, 2), error_creation_rate=0)
    context = VerificationContext.for_model(ReviewPlanning, "records").with_rule(STATIONARY_MAXIMUM)
    if type(record) is not RecordWork:
        context = context.replace(context.symbols.candidate.review.record, type(record))
    if type(response) is not ExponentialReview:
        context = context.replace(context.symbols.candidate.review.response, type(response))
    return context.observe(
        {context.symbols.candidate.review.record: record, context.symbols.candidate.review.response: response}
    )


class RationalReview(ExponentialReview):
    """An alternative with the same initial slope and slower saturation."""

    caught_share = (
        ExponentialReview.detection_rate
        * ExponentialReview.minutes_per_record
        / (1 + ExponentialReview.detection_rate * ExponentialReview.minutes_per_record)
    )


class CappedReview(ExponentialReview):
    """A linear response with a plateau whose join violates the inherited C2 claim."""

    caught_share = Piecewise(
        (
            ExponentialReview.detection_rate * ExponentialReview.minutes_per_record,
            ExponentialReview.detection_rate * ExponentialReview.minutes_per_record < 1,
        ),
        (1, True),
    )


def transition_recurrence(P: MatrixBase, initial_error_fraction: Expr) -> Recurrence:
    """Evolve a two-state composition using a row-stochastic transition matrix.

    Rows and columns are ordered (correct, incorrect). Conservation of records
    leaves one independent share, so a scalar recurrence represents both states.
    Probability bounds and row sums are premises to check separately.
    """
    if P.shape != (2, 2):
        raise ValueError("The record-review transition matrix must have two rows and two columns")
    n = Index("n", nonnegative=True)
    errors = Series("p", n, real=True)
    previous = ImmutableMatrix([[1 - errors[n - 1], errors[n - 1]]])
    return Recurrence(errors[n], (previous * P)[0, 1], initial={0: initial_error_fraction})


def derive_continuous_transitions(
    minutes_per_record: Expr, detection_rate: Expr, error_creation_rate: Expr
) -> ImmutableMatrix:
    """Derive a continuous two-state response from increasingly short checks.

    First solve the recurrence for independent, nonnegative transition shares.
    Then parameterize those shares by short-step rates and take the limit at
    fixed total checking time. The zero-total-rate branch retains the affine
    limit too; with physical nonnegative rates this is the identity matrix.
    """
    a, c = (Dummy(name, nonnegative=True) for name in ("a", "c"))
    initial = Dummy("p", real=True)
    steps = Dummy("N", integer=True, positive=True)
    P = ImmutableMatrix([[1 - a, a], [c, 1 - c]])
    discrete = Lambda((a, c, initial), transition_recurrence(P, initial).at(steps).doit())

    minutes, correction, creation = (Dummy(name, positive=True) for name in ("t", "k", "r"))
    step = ImmutableMatrix(
        [
            [1 - creation * minutes / steps, creation * minutes / steps],
            [correction * minutes / steps, 1 - correction * minutes / steps],
        ]
    )
    after_steps = discrete(step[0, 1], step[1, 0], initial)
    continuous = Lambda((minutes, correction, creation, initial), simplify(limit(after_steps, steps, S.Infinity)))
    rows = []
    for starting_errors in (S.Zero, S.One):
        errors = Piecewise(
            (starting_errors + error_creation_rate * minutes_per_record, Eq(detection_rate + error_creation_rate, 0)),
            (continuous(minutes_per_record, detection_rate, error_creation_rate, starting_errors), True),
        )
        rows.append([1 - errors, errors])
    return ImmutableMatrix(rows).applyfunc(simplify)


# The parent inputs keep their identities when its response equations change.
P = derive_continuous_transitions(
    ExponentialReview.minutes_per_record, ExponentialReview.detection_rate, ExponentialReview.error_creation_rate
)


class ContinuousReview(ExponentialReview):
    """Allow correction and error creation throughout a single check, with k > 0."""

    # Refine this input's domain; the other inputs and claims are inherited.
    detection_rate = InstanceDummy("k", positive=True)
    introduced_error_share = P[0, 1]
    caught_share = P[1, 0]


class ReviewEquilibrium(Modello):
    """Infer an error fraction unchanged by one pass of fallible review."""

    review = component(ReviewModel)
    error_fraction = review.record.error_fraction
    error_fraction = review.remaining_error_fraction


class RepeatedReview(Modello):
    """Infer pass count and evolve record composition with the same matrix law."""

    review = component(ReviewModel)
    passes = InstanceDummy("N", integer=True, nonnegative=True)
    checking_minutes_per_record = passes * review.response.minutes_per_record
    progression = transition_recurrence(
        ImmutableMatrix(
            [
                [1 - review.response.introduced_error_share, review.response.introduced_error_share],
                [review.response.caught_share, 1 - review.response.caught_share],
            ]
        ),
        review.record.error_fraction,
    )
    final_error_fraction = progression.at(passes)


def write_reports(directory: Path, *, plot: bool = False) -> None:
    """Export the baseline, changed response, and blocked kinked response.

    JSON describes evidence and exact results. Reproduce it by running this
    module; it does not deserialize or execute expressions from an audit file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    measured = review_context()
    rational = measured.replace(measured.symbols.candidate.review.response, RationalReview)
    capped = measured.replace(measured.symbols.candidate.review.response, CappedReview)
    contexts = {
        "exponential": measured.observe({measured.symbols.candidate.review.response.minutes_per_record: 2 * log(3)}),
        "rational": rational.observe(
            {rational.symbols.candidate.review.response.minutes_per_record: 2 * (sqrt(3) - 1)}
        ),
        "capped": capped.observe({capped.symbols.candidate.review.response.minutes_per_record: 2}),
    }
    summary: dict = {
        "versions": {name: metadata.version(name) for name in ("modello", "sympy")},
        "inputs": {
            path: str(measured.expression(path))
            for path in (
                "candidate.review.record.entry_minutes_per_record",
                "candidate.review.record.error_fraction",
                "candidate.review.record.repair_minutes_per_error",
                "candidate.review.response.detection_rate",
                "candidate.review.response.error_creation_rate",
            )
        },
        "plans": {},
    }
    for name, current in contexts.items():
        try:
            result = current.run(review_plan)
        except VerificationError as error:
            report = error.report
            summary["plans"][name] = {"blocked": str(error)}
        else:
            report = result.report
            summary["plans"][name] = {
                field.name: {"exact": str(value), "decimal": float(value)}
                for field in fields(result.data)
                for value in (getattr(result.data, field.name),)
            }
        (directory / f"{name}.json").write_text(report.to_json(), encoding="utf-8")
        (directory / f"{name}.html").write_text(report.to_html(title=f"Record review: {name}"), encoding="utf-8")
    (directory / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plot:
        _write_plots(contexts, directory)


def _write_plots(contexts: dict[str, VerificationContext[ReviewPlanning]], directory: Path) -> None:
    """Draw the initial tradeoff and later replacements separately as numerical observations."""
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    def new_plot(title, ylabel):
        figure = Figure(figsize=(6, 3.8), layout="constrained")
        FigureCanvasSVG(figure)
        axes = figure.subplots()
        axes.set(title=title, xlabel=r"Checking minutes per record $t$", ylabel=ylabel)
        axes.axhline(0, color="0.5", linewidth=0.7)
        return figure, axes

    sample_minutes_per_record = [index / 50 for index in range(301)]
    baseline = contexts["exponential"]
    minutes_per_record = baseline.symbols.candidate.review.response.minutes_per_record
    minutes_saved_per_record = baseline.expression(
        baseline.symbols.candidate.minutes_saved_per_record, wrt=minutes_per_record
    ).doit()
    benefit, benefit_axes = new_plot("Why spend time checking?", "Minutes per record")
    for expression, label, style in (
        (minutes_saved_per_record, r"Minutes saved $G(t)$", "-"),
        (minutes_saved_per_record + minutes_per_record, r"Repair minutes avoided $a\,u(t)$", "-"),
        (minutes_per_record, r"Checking minutes $t$", "--"),
    ):
        evaluate = lambdify(minutes_per_record, expression, "math")
        benefit_axes.plot(
            sample_minutes_per_record,
            [evaluate(minutes) for minutes in sample_minutes_per_record],
            style,
            label=label,
        )
    benefit_axes.legend()
    benefit.savefig(directory / "benefit.svg", metadata={"Date": None})

    comparison, comparison_axes = new_plot("Changing the response curve", r"Minutes saved per record $G(t)$")
    for name, current in contexts.items():
        axis = current.symbols.candidate.review.response.minutes_per_record
        evaluate = lambdify(
            axis,
            current.expression(current.symbols.candidate.minutes_saved_per_record, wrt=axis).doit(),
            "math",
        )
        comparison_axes.plot(
            sample_minutes_per_record,
            [evaluate(minutes) for minutes in sample_minutes_per_record],
            label=name.capitalize(),
        )
    comparison_axes.legend()
    comparison.savefig(directory / "responses.svg", metadata={"Date": None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artefacts/record-review"))
    parser.add_argument(
        "--plot", action="store_true", help="also draw benefit.svg and responses.svg; needs modello[numerics]"
    )
    arguments = parser.parse_args()
    write_reports(arguments.output, plot=arguments.plot)
    print(f"Record-review evidence and results: {arguments.output}")


if __name__ == "__main__":
    main()

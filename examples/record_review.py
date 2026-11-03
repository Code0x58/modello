"""Choose checking time for digitised records using exact, verified premises.

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

from sympy import Derivative, Expr, Interval, Piecewise, Rational, S, Subs, diff, exp, lambdify, log, simplify, sqrt

from modello import Coordinate, InstanceDummy, Modello, component
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

EFFORT_DOMAIN = Interval(0, S.Infinity)


class ExponentialReview(VerificationModel):
    """Fraction of otherwise missed errors caught after ``effort`` minutes.

    ``rate`` is deliberately unrestricted: increasing response needs rate > 0.
    The curve is a modelling assumption, not a fitted or validated error law.
    """

    effort = Coordinate("t", nonnegative=True)
    rate = InstanceDummy("k", real=True)
    value = specifies(
        1 - exp(-rate * effort),
        strictly_increasing(
            "value",
            wrt="effort",
            domain=EFFORT_DOMAIN,
            reason="Extra checking must catch a larger fraction of the missed errors.",
        ),
        concave(
            "value",
            wrt="effort",
            domain=EFFORT_DOMAIN,
            reason="Each extra minute has diminishing benefit.",
        ),
        continuously_differentiable(
            "value",
            wrt="effort",
            domain=EFFORT_DOMAIN,
            order=2,
            reason="The plan includes a second-order approximation of workload near the chosen effort.",
        ),
    )


class RationalReview(ExponentialReview):
    """An alternative with the same initial slope and slower saturation."""

    value = ExponentialReview.rate * ExponentialReview.effort / (1 + ExponentialReview.rate * ExponentialReview.effort)


class CappedReview(ExponentialReview):
    """A tempting replacement whose kink violates the inherited C2 obligation."""

    value = Piecewise(
        (ExponentialReview.rate * ExponentialReview.effort, ExponentialReview.rate * ExponentialReview.effort < 1),
        (1, True),
    )


class ReviewPlanning(VerificationModel):
    """Expected staff minutes per record, including later repair of missed errors."""

    review = component(ExponentialReview)
    effort = ExponentialReview.effort
    entry_minutes = InstanceDummy("e", positive=True)
    avoidable_minutes = InstanceDummy("a", positive=True)
    candidate = InstanceDummy("s", nonnegative=True)
    net_saving = avoidable_minutes * review.value - effort
    expected_minutes = entry_minutes + avoidable_minutes - net_saving
    # A deferred derivative acts on the resolved curve, rather than the field's
    # placeholder symbol. It also generates a differentiability obligation.
    marginal_saving = Derivative(net_saving, effort, evaluate=False)
    candidate_marginal = Subs(marginal_saving, effort, candidate)
    candidate_saving = Subs(net_saving, effort, candidate)
    candidate_minutes = Subs(expected_minutes, effort, candidate)


class EntryBudget(Modello):
    """Continuous workload budget: supply any two quantities to infer the third."""

    records = InstanceDummy("n", positive=True)
    minutes_per_record = InstanceDummy("m", positive=True)
    staff_minutes = records * minutes_per_record


GLOBAL_MAXIMUM = Obligation(
    "global_maximum",
    ReviewPlanning.candidate_saving >= ReviewPlanning.net_saving,
    "rule",
    wrt="effort",
    domain=EFFORT_DOMAIN,
    reason="The proposed checking time maximizes net saving over all nonnegative efforts.",
)

STATIONARY_MAXIMUM = InferenceRule(
    "concave_stationary_maximum",
    GLOBAL_MAXIMUM,
    (
        concave("net_saving", wrt="effort", domain=EFFORT_DOMAIN, name="concavity"),
        continuously_differentiable(
            "net_saving",
            wrt="effort",
            domain=EFFORT_DOMAIN,
            order=1,
            name="differentiability",
        ),
        equal_to("candidate_marginal", 0, name="stationarity"),
        nonnegative("candidate", name="feasibility"),
    ),
    source="Concave supporting-line inequality; docs/examples/record-review/index.rst#why-this-is-a-global-optimum",
)


@dataclass(frozen=True)
class ReviewDecision:
    effort: Expr
    net_saving: Expr
    minutes_per_record: Expr
    workload_curvature: Expr


def _evaluate_plan(current: VerificationContext) -> ReviewDecision:
    effort = current.expression("candidate")
    curve = current.expression("expected_minutes", wrt="effort")
    axis = current.problem.model.effort
    values = (
        effort,
        current.expression("candidate_saving").doit(),
        current.expression("candidate_minutes").doit(),
        diff(curve, axis, 2).subs(axis, effort),
    )
    values = tuple(simplify(value) for value in values)
    if any(value.free_symbols or value.is_finite is not True or value.is_real is not True for value in values):
        raise ValueError("Observe all parameters and the candidate before evaluating a plan")
    return ReviewDecision(*values)


review_plan = Method(
    "review_plan",
    _evaluate_plan,
    preconditions=(GLOBAL_MAXIMUM, positive("candidate_minutes", name="positive_workload")),
    uses=("candidate", "candidate_saving", "candidate_minutes"),
)


def review_context() -> VerificationContext[ReviewPlanning]:
    """Illustrative inputs: four minutes to enter, six avoidable repair minutes."""
    return (
        VerificationContext.for_model(ReviewPlanning, "records")
        .with_rule(STATIONARY_MAXIMUM)
        .observe(
            entry_minutes=4,
            avoidable_minutes=6,
            review=ExponentialReview("response", rate=Rational(1, 2)),
        )
    )


def write_reports(directory: Path, *, plot: bool = False) -> None:
    """Export the baseline, changed response, and blocked kinked response.

    JSON describes evidence and exact results. Reproduce it by running this
    module; it does not deserialize or execute expressions from an audit file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    measured = review_context()
    contexts = {
        "exponential": measured.observe(candidate=2 * log(3)),
        "rational": measured.replace("review", RationalReview).observe(candidate=2 * (sqrt(3) - 1)),
        "capped": measured.replace("review", CappedReview).observe(candidate=2),
    }
    summary: dict = {
        "versions": {name: metadata.version(name) for name in ("modello", "sympy")},
        "inputs": {
            path: str(measured.expression(path)) for path in ("entry_minutes", "avoidable_minutes", "review.rate")
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
        _plot(contexts, directory / "curves.svg")


def _plot(contexts: dict[str, VerificationContext[ReviewPlanning]], path: Path) -> None:
    """Draw observations of the curves; the figure supplies no proof evidence."""
    from matplotlib.backends.backend_svg import FigureCanvasSVG
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 3.8), layout="constrained")
    FigureCanvasSVG(figure)
    benefit_axes, comparison_axes = figure.subplots(1, 2)
    times = [Rational(index, 50) for index in range(301)]
    baseline = contexts["exponential"]
    axis = baseline.problem.model.effort
    gain = baseline.expression("net_saving", wrt="effort")
    for expression, label, style in (
        (gain + axis, "Repair time avoided", "-"),
        (axis, "Checking time", "--"),
        (gain, "Net saving", "-"),
    ):
        evaluate = lambdify(axis, expression, "math")
        benefit_axes.plot(times, [evaluate(float(time)) for time in times], style, label=label)
    for name, current in contexts.items():
        evaluate = lambdify(axis, current.expression("net_saving", wrt="effort"), "math")
        comparison_axes.plot(times, [evaluate(float(time)) for time in times], label=name.capitalize())
    for axes, title in ((benefit_axes, "Why spend time checking?"), (comparison_axes, "Changing the response curve")):
        axes.set(title=title, xlabel="Checking time (minutes per record)", ylabel="Minutes per record")
        axes.axhline(0, color="0.5", linewidth=0.7)
        axes.legend(fontsize="small")
    comparison_axes.set_ylabel("Net minutes saved per record")
    figure.savefig(path, metadata={"Date": None})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artefacts/record-review"))
    parser.add_argument("--plot", action="store_true", help="also draw curves.svg; needs modello[numerics]")
    arguments = parser.parse_args()
    write_reports(arguments.output, plot=arguments.plot)
    print(f"Record-review evidence and results: {arguments.output}")


if __name__ == "__main__":
    main()

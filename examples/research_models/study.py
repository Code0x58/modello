"""Replay the constrained-model collection and write offline audit reports.

Use ``python -m examples.research_models --output artefacts/research-models``.
``--symbolic-only`` avoids every optional numerical and plotting dependency;
``--compact`` selects representative branches for documentation builds.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import cast

from sympy import Rational, S

from modello.contrib.research import ResearchCase, ResearchError, ResearchReport, ResearchResult

from ._study import Blocked, Computation, Study, Variant

MODELS = ("black-scholes", "participation-execution", "markowitz", "cir-bond", "perpetual-put")


def _attempt(call: Callable[[], ResearchResult], *, kind="exact scalar checks"):
    try:
        return Computation(call(), kind), None
    except ResearchError as error:
        return None, Blocked.from_error(error)


def _black_scholes(*, numerical: bool, compact: bool) -> Study:
    from . import black_scholes as model

    variants = {}
    if compact:
        case = model.study_case(schedule=True, now=Rational(1, 4))
        results = {"identities": Computation(case.run(model.symbolic_method))}
        if numerical:
            results["quadrature"] = Computation(case.run(model.valuation_method), "numerical observation")
        _, blocked = _attempt(lambda: case.run(model.dividend_free_method()))
        variants["affine_with_dividends"] = Variant(
            case, results, {"dividend_free_shortcut": blocked} if blocked else {}
        )
    else:
        demo = model.run_demo(numerical=numerical)
        for name, case in demo["cases"].items():
            results, rejected = {}, {}
            if name in demo["symbolic"]:
                results["identities"] = Computation(demo["symbolic"][name])
            if name in demo["numerical"]:
                results["quadrature"] = Computation(demo["numerical"][name], "numerical observation")
            if name == "no_dividends":
                results["dividend_free_shortcut"] = Computation(demo["dividend_free_shortcut"])
            elif name == "constant":
                rejected["dividend_free_shortcut"] = Blocked(
                    demo["rejected_dividend_free_shortcut"],
                    "Positive dividend yield refutes the dividend-free parity shortcut.",
                )
            variants[name] = Variant(
                case,
                results,
                rejected,
                {"payoff_boundary": {"kind": "exact one-sided limits", "values": demo["boundary"]}},
            )
        for name, row in demo["rejected_interior"].items():
            variants[name] = Variant(
                row["case"],
                blocked={
                    "interior_formula": Blocked(row["report"], "This input is outside the interior pricing domain.")
                },
            )
    return Study(
        "black-scholes",
        "European Black–Scholes–Merton",
        "Dividend yield and deterministic coefficient schedules, with exact identities and separate payoff boundaries.",
        (model.BSM_SOURCE, model.CDF_SOURCE),
        variants,
        (
            "Gaussian C2 composition is an explicit trusted rule; "
            "PDE residuals do not prove stochastic replication or uniqueness.",
        ),
    )


def _participation(*, numerical: bool, compact: bool) -> Study:
    from . import participation_execution as model

    if compact:
        case = model.participation_case(cap=S.Half, deadline=2)
        rows: dict[str, dict] = {"capped_deadline": {"case": case, "analytic": case.run(model.analytic_method)}}
        if numerical:
            rows["capped_deadline"]["numerical"] = case.run(model.numerical_method)
    else:
        rows = model.compare(numerical=numerical)
    variants = {}
    for name, row in rows.items():
        results, rejected = {}, {}
        for method in ("analytic", "numerical"):
            if isinstance(row.get(method), ResearchResult):
                results[method] = Computation(
                    row[method], "numerical observation" if method == "numerical" else "exact scalar checks"
                )
            if isinstance(row.get(f"{method}_report"), ResearchReport):
                rejected[method] = Blocked(row[f"{method}_report"], str(row[f"{method}_blocked"]))
        variants[name] = Variant(row["case"], results, rejected)
    return Study(
        "participation-execution",
        "Guéant constant-participation execution",
        "Caps, completion deadlines, risk-neutral nonattainment, and a replacement impact law.",
        (model.SOURCE,),
        variants,
        (
            "Optimality is within constant-volume, constant-participation scalar strategies; "
            "arbitrary execution paths are outside scope.",
        ),
    )


def _markowitz(*, numerical: bool, compact: bool) -> Study:
    from . import markowitz as model

    if compact:
        case = model.portfolio_case()
        restrictions = model.PortfolioRestrictions(target=Rational(7, 100), cap=Rational(4, 5))
        rows = {"target_and_cap": {"case": case, "result": model.allocate(case, restrictions)}}
    else:
        rows = model.run_demo()
    variants = {}
    for name, row in rows.items():
        result = row["result"]
        observations = {}
        if numerical and name not in {"shorts_allowed", "flat"}:
            observations["independent_minimum"] = model.numerical_reference(row["case"], result.data.restrictions)
        variants[name] = Variant(
            row["case"],
            {"allocation": Computation(result)},
            observations=observations,
            notes=result.data.restrictions.changes,
        )
    if not compact:
        for name, case, restrictions in (
            ("infeasible_target", model.portfolio_case(), model.PortfolioRestrictions(target=Rational(9, 100))),
            ("invalid_covariance", model.portfolio_case(covariance=-3), model.PortfolioRestrictions()),
        ):
            result, blocked = _attempt(lambda: model.allocate(case, restrictions))
            variants[name] = Variant(
                case, {"allocation": result} if result else {}, {"allocation": blocked} if blocked else {}
            )
    return Study(
        "markowitz",
        "Constrained two-asset allocation",
        "Exact scalar KKT certificates, active-set changes, removed restrictions, and nonunique flat minima.",
        (model.SOURCE,),
        variants,
        (
            "The budget reduces this example to one coordinate; "
            "no general matrix or multivariate proof engine is implied.",
        ),
    )


def _cir(*, numerical: bool, compact: bool) -> Study:
    from . import cir_bond as model

    if compact:
        case = model.cir_case(sigma=Rational(2, 5))
        row = {"case": case, "price": model.price(case), "verification": model.verify(case)}
        result, blocked = _attempt(
            lambda: case.run(model.strict_positive_boundary_method), kind="trusted stochastic rule"
        )
        if result:
            row["boundary"] = result.result
        if blocked:
            row["boundary_report"], row["boundary_blocked"] = blocked.report, blocked.message
        if numerical:
            row["numerical"] = case.run(model.numerical_method)
        rows = {"feller_not_satisfied": row}
    else:
        rows = model.run_demo(numerical=numerical)
    variants = {}
    for name, row in rows.items():
        results = {}
        for method in ("price", "verification", "boundary", "numerical"):
            result = row.get(method)
            if isinstance(result, ResearchResult):
                kind = (
                    "numerical observation"
                    if method == "numerical"
                    else "trusted stochastic rule"
                    if method in {"price", "boundary"}
                    else "exact scalar checks"
                )
                results[method] = Computation(result, kind)
        blocked = {}
        report = row.get("boundary_report")
        if isinstance(report, ResearchReport):
            blocked["boundary"] = Blocked(report, str(row["boundary_blocked"]))
        variants[name] = Variant(cast(ResearchCase, row["case"]), results, blocked)
    return Study(
        "cir-bond",
        "CIR zero-coupon bonds",
        "Affine valuation and Riccati checks remain valid when the sufficient Feller boundary condition is removed.",
        (model.SOURCE,),
        variants,
        (
            "The CIR path, boundary, and discounted-expectation conclusions "
            "are explicitly sourced trusted stochastic rules.",
        ),
    )


def _perpetual(*, numerical: bool, compact: bool) -> Study:
    from . import perpetual_put as model

    variants = {}
    if compact:
        case = model.perpetual_put_case()
        results = {"scalar_certificate": Computation(case.run(model.algebra_method))}
        if numerical:
            results["barrier_search"] = Computation(case.run(model.numerical_method), "numerical observation")
        rejected = {}
        for name, method in (
            ("global_c2", model.c2_consumer_method()),
            ("untrusted_stopping", model.verification_method),
        ):
            _, blocked = _attempt(lambda: case.run(method))
            if blocked:
                rejected[name] = blocked
        variants["smooth_fit"] = Variant(case, results, rejected)
    else:
        demo = model.run_demo(numerical=numerical)
        results = {"scalar_certificate": Computation(demo["algebra"])}
        if numerical:
            results["barrier_search"] = Computation(demo["numerical"], "numerical observation")
        rejected = {}
        for key in ("global_c2", "untrusted_verification"):
            row = demo[f"blocked_{key}"]
            rejected[key] = Blocked(row["report"], row["message"])
        variants["smooth_fit"] = Variant(demo["case"], results, rejected)
        variants["trusted_stopping"] = Variant(
            demo["trusted_case"],
            {"verification": Computation(demo["verification"], "explicitly trusted stochastic rule")},
        )
        row = demo["blocked_zero_rate"]
        variants["zero_rate"] = Variant(
            row["case"], blocked={"positive_rate_shortcut": Blocked(row["report"], row["message"])}
        )
        if numerical:
            variants["immediate_exercise"] = Variant(
                demo["exercise_case"],
                {"barrier_search": Computation(demo["numerical_exercise"], "numerical observation")},
            )
    return Study(
        "perpetual-put",
        "Perpetual American put",
        "Separate exercise and continuation regions, exact smooth fit, and a failed global C2 consumer.",
        (model.SOURCE,),
        variants,
        (
            "A scalar boundary certificate is separate from "
            "the explicitly adopted GBM optimal-stopping verification theorem.",
        ),
    )


def build_collection(*, models: Iterable[str] = MODELS, numerical=True, compact=False) -> tuple[Study, ...]:
    """Build named studies without importing unrelated model modules."""
    builders = dict(zip(MODELS, (_black_scholes, _participation, _markowitz, _cir, _perpetual), strict=True))
    selected = tuple(dict.fromkeys(models))
    if not selected or any(name not in builders for name in selected):
        raise ValueError(f"Select one or more model names from {MODELS}")
    return tuple(builders[name](numerical=numerical, compact=compact) for name in selected)


def write_collection(directory: Path, *, models: Iterable[str] = MODELS, numerical=True, compact=False):
    """Replay selected studies and export descriptive HTML/JSON/CSV evidence."""
    from ._reporting import write_reports

    studies = build_collection(models=models, numerical=numerical, compact=compact)
    return write_reports(studies, Path(directory), numerical=numerical, compact=compact)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artefacts/research-models"))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument(
        "--symbolic-only", action="store_true", help="Use only SymPy; skip numerical references and figures"
    )
    parser.add_argument("--compact", action="store_true", help="Select representative branches for a faster replay")
    args = parser.parse_args(argv)
    write_collection(args.output, models=args.models, numerical=not args.symbolic_only, compact=args.compact)
    print(f"Research collection: {args.output / 'report.html'}")


if __name__ == "__main__":
    main()

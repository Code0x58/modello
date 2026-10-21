"""Infer an impact interval, then reuse it to bound another trade's cost.

Run ``python -m examples.solution_families``. Add ``--plot cost.png`` to
export the bounded cost curve with the optional Matplotlib dependency.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sympy import Interval

from modello import (
    InstanceDummy,
    Modello,
    ModelProblem,
    SolutionFamily,
    UnresolvedSolutionError,
    component,
    inspect_model,
)


class Execution(Modello):
    """A deliberately small model: price impact is linear in trade quantity."""

    quantity = InstanceDummy("quantity", positive=True)
    reference = InstanceDummy("reference", positive=True)
    impact = InstanceDummy("impact", nonnegative=True)
    fee = InstanceDummy("fee", nonnegative=True)
    price = reference + impact * quantity
    cost = quantity * price + fee


class ImpactStudy(Modello):
    """Recorded and planned executions share their impact coefficient."""

    recorded = component(Execution)
    planned = component(Execution)
    impact = recorded.impact
    impact = planned.impact

    @classmethod
    def prepare(cls) -> ModelProblem[ImpactStudy]:
        """Fix the observed trade terms and the proposed trade size."""
        return inspect_model(
            cls,
            "impact study",
            recorded=Execution("recorded", quantity=2, reference=99, fee=1),
            planned=Execution("planned", quantity=10, reference=100, fee=1),
        )

    @classmethod
    def calibrated(cls) -> SolutionFamily[ImpactStudy]:
        """Retain every impact consistent with a bounded cost observation."""
        problem = cls.prepare()
        cost = problem.field("recorded", "cost")
        return problem.solutions(problem.field("impact")).where(cost.symbol >= 200, cost.symbol <= 202)

    @staticmethod
    def plot(family: SolutionFamily[ImpactStudy], destination: Path) -> None:
        """Plot the symbolic cost expression on its admissible impact interval."""
        from sympy.plotting import plot

        impact = family.problem.field("impact")
        cost = family.problem.field("planned", "cost")
        domain = family.values(impact)
        costs = family.values(cost)
        if (
            not isinstance(domain, Interval)
            or not isinstance(costs, Interval)
            or not all(value.is_finite is True for value in (domain.start, domain.end, costs.start, costs.end))
        ):
            raise UnresolvedSolutionError(family, "This plot needs bounded impact and cost intervals.")
        chart = plot(
            family.expression(cost),
            (impact.symbol, domain.start, domain.end),
            show=False,
            xlabel="Impact coefficient",
            ylabel="Planned execution cost",
            title="Planned cost under the calibrated impact interval",
            axis_center=(float(domain.start), float(costs.start)),
        )
        chart.save(str(destination))


def main() -> None:
    """Report the feasible interval and optionally export its cost curve."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path)
    arguments = parser.parse_args()
    family = ImpactStudy.calibrated()
    problem = family.problem
    print("Impact:", family.values(problem.field("impact")))
    print("Planned cost:", family.values(problem.field("planned", "cost")))
    if arguments.plot:
        ImpactStudy.plot(family, arguments.plot)


if __name__ == "__main__":
    main()

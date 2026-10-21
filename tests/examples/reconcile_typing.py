# mypy: warn-unused-ignores
"""Reconciliation results retain their model type, including direct trials."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from examples.applications import InvoiceLine, Resistor
    from examples.reconcile import MutationReport, MutationTrial, Reconciliation, Suspect
    from modello import ModelProblem, SolutionFamily, inspect_model

    def check_results() -> None:
        problem = inspect_model(InvoiceLine, "invoice", quantity=10, unit_price=1200, amount=120)
        price = problem.field("unit_price")
        study = Reconciliation(problem, [Suspect(price)])
        report: MutationReport[InvoiceLine] = study.scan()
        trial = report.matches[0]
        _problem: ModelProblem[InvoiceLine] = trial.problem
        _case: InvoiceLine = trial.problem.solutions().one()
        _family: SolutionFamily[InvoiceLine] = study.infer(price)
        direct = MutationTrial(price, trial.original, trial.proposed, trial.mutations, trial.problem, trial.analysis)
        _wrong_trial: MutationTrial[Resistor] = direct  # type: ignore[assignment]
        _wrong_report: MutationReport[Resistor] = study.scan()  # type: ignore[assignment]
        _wrong_case: Resistor = trial.problem.solutions().one()  # type: ignore[assignment]

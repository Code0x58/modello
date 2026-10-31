"""Shared scope checks for the research examples' computation adapters."""

from sympy import Symbol
from sympy.assumptions import global_assumptions

from modello.contrib.research import ResearchCase


def require_parameter_assumptions(case: ResearchCase, *coordinates: Symbol, message: str) -> None:
    """Reject local or ambient predicates that restrict a varying state."""
    states = frozenset(coordinates)
    if any(case._law(condition).free_symbols & states for condition in (*case.assumptions, *global_assumptions)):
        raise ValueError(message)

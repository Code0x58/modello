"""Shared scope checks for the verification examples' computation adapters."""

from sympy import Symbol
from sympy.assumptions import global_assumptions

from modello.contrib.verification import VerificationContext


def require_parameter_assumptions(case: VerificationContext, *coordinates: Symbol, message: str) -> None:
    """Reject local or ambient predicates that restrict a varying state."""
    states = frozenset(coordinates)
    if any(case._law(condition).free_symbols & states for condition in (*case.assumptions, *global_assumptions)):
        raise ValueError(message)

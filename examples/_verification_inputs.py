"""Shared input validation for the verification examples' computation adapters."""

from sympy import Symbol
from sympy.assumptions import global_assumptions

from modello.contrib.verification import VerificationContext


def validate_parameter_assumptions(context: VerificationContext, *coordinates: Symbol, message: str) -> None:
    """Reject local or ambient predicates that restrict a varying state."""
    states = frozenset(coordinates)
    if any(context._law(condition).free_symbols & states for condition in (*context.assumptions, *global_assumptions)):
        raise ValueError(message)

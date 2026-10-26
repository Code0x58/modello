"""Opt-in mathematical research cases, requirements, and evidence.

This module needs only SymPy. Numerical backends and plotting are imported by
the methods that use them, never by ordinary Modello or symbolic research.
"""

from ._case import ResearchCase as ResearchCase, ResearchModel as ResearchModel
from ._records import (
    Claim as Claim,
    Evidence as Evidence,
    Experiment as Experiment,
    Method as Method,
    ProofHint as ProofHint,
    Requirement as Requirement,
    Requires as Requires,
    ResearchError as ResearchError,
    ResearchReport as ResearchReport,
    ResearchResult as ResearchResult,
    Theorem as Theorem,
    concave as concave,
    convex as convex,
    curvature_bound as curvature_bound,
    decreasing as decreasing,
    differentiable as differentiable,
    exact_form as exact_form,
    increasing as increasing,
    nonnegative as nonnegative,
    nonzero as nonzero,
    positive as positive,
    predicate as predicate,
    smooth as smooth,
    uniform_bound as uniform_bound,
    upper_bound as upper_bound,
    vanishes as vanishes,
)

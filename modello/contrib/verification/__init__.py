"""Deductive verification of mathematical model properties.

This module uses the core dependencies. Numerical backends and plotting are imported by
the methods that use them, never by ordinary Modello or symbolic verification.
"""

from ._context import (
    VerificationContext as VerificationContext,
    VerificationModel as VerificationModel,
    specifies as specifies,
)
from ._records import (
    Evidence as Evidence,
    InferenceRule as InferenceRule,
    Method as Method,
    Obligation as Obligation,
    Sample as Sample,
    VerificationError as VerificationError,
    VerificationReport as VerificationReport,
    VerificationResult as VerificationResult,
    concave as concave,
    continuously_differentiable as continuously_differentiable,
    convex as convex,
    derivative_ratio_bound as derivative_ratio_bound,
    differentiable as differentiable,
    equal_to as equal_to,
    nonnegative as nonnegative,
    nonzero as nonzero,
    positive as positive,
    predicate as predicate,
    strictly_decreasing as strictly_decreasing,
    strictly_increasing as strictly_increasing,
    supremum_below as supremum_below,
    tends_to_zero as tends_to_zero,
)

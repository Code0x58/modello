"""Conservative scalar proof attempts with explicit domains and premises."""

from __future__ import annotations

from dataclasses import dataclass

from sympy import (
    Abs,
    AccumBounds,
    And,
    Basic,
    Derivative,
    Dummy,
    Eq,
    Expr,
    Interval,
    Limit,
    Mul,
    Ne,
    Not,
    Or,
    Piecewise,
    Poly,
    Pow,
    Q,
    S,
    Set,
    Subs,
    Symbol,
    ask,
    diff,
    factor,
    limit,
    preorder_traversal,
    reduce_inequalities,
    refine,
    simplify,
    sympify,
)
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef
from sympy.logic.inference import satisfiable
from sympy.polys.polyerrors import PolynomialError

from modello._operators import _defined_part

from ._records import Claim, ProofHint


@dataclass(frozen=True)
class Attempt:
    condition: Basic | None
    steps: tuple[str, ...]
    witness: tuple[tuple[Symbol, Expr], ...] = ()


def conjunction(*conditions: Basic | None) -> Basic | None:
    if any(value is S.false for value in conditions):
        return S.false
    if any(value is None for value in conditions):
        return None
    return And(*conditions)


def truth(condition: Basic, assumptions: tuple[Basic, ...]) -> Basic:
    """Check explicit premises without mutating global SymPy assumptions."""
    condition = sympify(condition)
    if condition in (S.true, S.false):
        return condition
    if condition in assumptions:
        return S.true
    if Not(condition) in assumptions:
        return S.false
    if isinstance(condition, And):
        return And(*(truth(arg, assumptions) for arg in condition.args))
    try:
        refined = refine(condition, And(*assumptions))
        if refined in (S.true, S.false):
            return refined
        if satisfiable(And(*assumptions, Not(condition)), use_lra_theory=True) is False:
            return S.true
        if satisfiable(And(*assumptions, condition), use_lra_theory=True) is False:
            return S.false
    except (TypeError, ValueError, NotImplementedError, AssertionError):
        pass
    return condition


def consistency(assumptions: tuple[Basic, ...]) -> bool | None:
    if not assumptions:
        return True
    combined = simplify(And(*assumptions))
    if combined is S.false:
        return False
    if combined is S.true:
        return True
    try:
        result = satisfiable(combined, use_lra_theory=True)
        return False if result is False else True
    except (TypeError, ValueError, NotImplementedError, AssertionError):
        return None


def _specialize(expression: Basic, variable: Symbol | None, assumptions: tuple[Basic, ...]):
    substitutions = {}
    for parameter in expression.free_symbols - ({variable} if variable is not None else set()):
        facts = {}
        for name in ("positive", "negative", "nonzero", "nonnegative", "nonpositive", "real", "integer"):
            try:
                answer = ask(getattr(Q, name)(parameter), And(*assumptions))
            except (ValueError, TypeError, NotImplementedError):
                answer = None
            if answer is True:
                facts[name] = True
        if facts:
            substitutions[parameter] = Dummy(parameter.name, **facts)
    return expression.xreplace(substitutions), {new: old for old, new in substitutions.items()}


def _parameter_guards(expression: Expr, variable: Symbol | None) -> Basic | None:
    """Keep parameter poles that continuous_domain does not quantify."""
    guards = []
    for part in preorder_traversal(expression):
        if not isinstance(part, Expr):
            continue
        if variable is not None and variable in part.free_symbols:
            if isinstance(part, Pow) and part.exp.free_symbols and part.base.is_positive is not True:
                return None
            continue
        if getattr(part, "is_finite", None) is True:
            continue
        if isinstance(part, Expr):
            numerator, denominator = part.as_numer_denom()
            if denominator != 1 and numerator.is_finite is True and denominator.is_finite is True:
                guards.append(Ne(denominator, 0))
                continue
        return None
    return And(*guards)


def _regular(expression: Expr, variable: Symbol | None, domain: Set, order: int) -> Attempt:
    validity = S.true
    if isinstance(expression, Piecewise) and expression.args[-1] == (S.NaN, S.true):
        validity = Or(*(condition for _, condition in expression.args[:-1]))
        expression = _defined_part(expression)
        if variable is not None and variable in validity.free_symbols:
            if validity.free_symbols != {variable}:
                return Attempt(None, ("A parameter-dependent coordinate guard remains unresolved.",))
            try:
                missing = domain - validity.as_set()
                if missing.is_empty is False:
                    return Attempt(S.false, (f"The original expression is undefined on {missing}.",))
                if missing.is_empty is not True:
                    return Attempt(None, ("The original expression's domain guard is unresolved.",))
                validity = S.true
            except (NotImplementedError, TypeError, ValueError):
                return Attempt(None, ("The original expression's domain guard is unresolved.",))
    guard = _parameter_guards(expression, variable)
    if guard is None:
        return Attempt(None, ("Parameter-dependent definedness is unresolved.",))
    if guard is S.false:
        return Attempt(S.false, ("A parameter denominator is zero.",))
    guard = And(guard, validity)
    if variable is None:
        return Attempt(guard, ("The parameter expression is defined under the retained guards.",))
    if expression.has(AppliedUndef, Derivative):
        return Attempt(None, ("A formal function or derivative does not establish regularity.",))
    if expression == Abs(variable) and order >= 1 and domain.contains(0) is S.true:
        return Attempt(S.false, ("The left and right derivatives of Abs differ at zero.",), ((variable, S.Zero),))
    if order > 2:
        return Attempt(None, ("Automatic regularity checks currently cover orders zero through two.",))
    for degree in range(order + 1):
        derivative = diff(expression, variable, degree)
        try:
            allowed = continuous_domain(derivative, variable, domain)
            missing = domain - allowed
            if missing.is_empty is False:
                return Attempt(S.false, (f"Derivative order {degree} is not continuous on {missing}.",))
            if missing.is_empty is not True:
                return Attempt(None, (f"The continuity domain of derivative order {degree} is unresolved.",))
        except (NotImplementedError, ValueError, TypeError):
            return Attempt(None, (f"The CAS cannot establish continuity of derivative order {degree}.",))
    return Attempt(guard, (f"Continuity domains certify C{order} on {domain}.",))


def _defined(expression: Expr, variable: Symbol | None, domain: Set) -> Attempt:
    """Continuity is sufficient for definedness; its failure is not a refutation."""
    regular = _regular(expression, variable, domain, 0)
    if regular.condition is not None and regular.condition is not S.false:
        return regular
    if regular.condition is S.false and any("original expression is undefined" in step for step in regular.steps):
        return regular
    body = _defined_part(expression)
    if body != expression:
        return Attempt(None, (*regular.steps, "The original expression's domain guard has not been established."))
    guard = _parameter_guards(body, variable)
    if body.is_finite is True and guard is not None:
        return Attempt(guard, ("The expression is finite; continuity is not required for this claim.",))
    if variable is not None:
        for point in _points(domain):
            value = body.subs(variable, point)
            if value.is_finite is False or value in (S.NaN, S.ComplexInfinity):
                return Attempt(S.false, (f"The expression is not a finite value at {point}.",), ((variable, point),))
    elif body.is_finite is False or body in (S.NaN, S.ComplexInfinity):
        return Attempt(S.false, ("The expression is not a finite value.",))
    return Attempt(None, (*regular.steps, "Definedness remains unresolved; discontinuity alone is not a refutation."))


def _differentiable(expression, variable, domain, order, point=None) -> Attempt:
    """Use Cn as a sufficient certificate and keep local questions local."""
    if point is not None:
        if expression == Abs(variable):
            return Attempt(
                Ne(point, 0),
                ("Absolute value is locally affine away from zero and has a cusp at zero.",),
                ((variable, point),) if point == 0 else (),
            )
        body = _defined_part(expression)
        value = expression.subs(variable, point)
        if value.is_finite is False or value in (S.NaN, S.ComplexInfinity):
            return Attempt(S.false, ("The function is undefined at the requested point.",), ((variable, point),))
        guard = _parameter_guards(body, variable)
        if guard is not None:
            try:
                neighborhood = domain
                if body != expression:
                    validity = Or(*(condition for _, condition in expression.args[:-1]))
                    if variable in validity.free_symbols:
                        if validity.free_symbols != {variable}:
                            return Attempt(None, ("The local coordinate domain depends on unresolved parameters.",))
                        neighborhood &= validity.as_set()
                    else:
                        guard = conjunction(guard, validity)
                for degree in range(order + 1):
                    neighborhood &= continuous_domain(diff(body, variable, degree), variable, domain)
                if neighborhood.interior.contains(point) is S.true:
                    return Attempt(guard, (f"A C{order} neighborhood contains the requested point {point}.",))
            except (NotImplementedError, TypeError, ValueError):
                pass
        return Attempt(None, ("Local differentiability at the requested point remains unresolved.",))
    regular = _regular(expression, variable, domain, order)
    if regular.condition is not None and regular.condition is not S.false:
        return regular
    defined = _defined(expression, variable, domain)
    if defined.condition is S.false:
        return defined
    if expression == Abs(variable) and domain.contains(0) is S.true:
        return Attempt(S.false, ("The two one-sided derivatives disagree at zero.",), ((variable, S.Zero),))
    return Attempt(None, (*regular.steps, "Failure of a sufficient Cn certificate does not refute differentiability."))


def _sign(expression: Expr, variable: Symbol | None, domain: Set, strict: bool) -> Attempt:
    guard = _defined(expression, variable, domain)
    if guard.condition is None or guard.condition is S.false:
        return guard
    if expression.is_real is False:
        return Attempt(S.false, ("An ordered sign claim requires a real value.",))
    if variable is not None and expression.is_real is not True:
        return Attempt(None, ("Real-valuedness on the coordinate domain is unresolved.",))
    original = expression
    expression = factor(expression)
    if expression.is_Mul:
        expression = Mul(*(term for term in expression.args if term.is_positive is not True))
    if expression.has(AppliedUndef, Derivative):
        return Attempt(None, ("The sign of a formal function is unresolved.",))
    condition = expression > 0 if strict else expression >= 0
    if condition in (S.true, S.false) or variable is None or variable not in expression.free_symbols:
        return Attempt(conjunction(guard.condition, condition), (f"Sign reduction gives {condition}.",))
    if expression.free_symbols != {variable}:
        return Attempt(None, (f"The parameter-dependent coordinate inequality {condition} is unresolved.",))
    try:
        allowed = reduce_inequalities([condition], variable).as_set()
        missing = domain - allowed
        if missing.is_empty is True:
            return Attempt(guard.condition, (f"The sign inequality holds throughout {domain}.",))
        if missing.is_empty is False:
            witness = _sign_witness(original, variable, domain, strict)
            return Attempt(S.false, (f"The sign inequality fails on {missing}.",), witness)
    except (NotImplementedError, ValueError, TypeError):
        pass
    return Attempt(None, ("The global sign inequality remains unresolved.",))


def _nonzero(expression, variable, domain):
    defined = _defined(expression, variable, domain)
    if defined.condition is None or defined.condition is S.false:
        return defined
    if variable is None or variable not in expression.free_symbols:
        return Attempt(conjunction(defined.condition, Ne(expression, 0)), ("Retain the nonzero-value predicate.",))
    if expression.is_zero is False:
        return Attempt(defined.condition, ("The defined expression cannot be zero.",))
    if expression.is_real is True:
        return _sign(expression**2, variable, domain, True)
    return Attempt(None, ("Zeros of this potentially complex-valued expression are unresolved.",))


def _points(domain: Set):
    points = [S.Zero, S.One, -S.One, S(2), S(-2)]
    if isinstance(domain, Interval):
        if domain.start.is_finite and domain.end.is_finite:
            points.extend([(domain.start + domain.end) / 2, (2 * domain.start + domain.end) / 3])
        elif domain.start.is_finite:
            points.extend([domain.start + 1, domain.start + 2])
        elif domain.end.is_finite:
            points.extend([domain.end - 1, domain.end - 2])
    return tuple(dict.fromkeys(point for point in points if domain.contains(point) is S.true))


def _sign_witness(expression, variable, domain, strict):
    for point in _points(domain):
        value = simplify(expression.subs(variable, point))
        valid = value > 0 if strict else value >= 0
        if valid is S.false:
            return ((variable, point),)
    return ()


def _monotone(expression, variable, domain) -> Attempt:
    if expression.is_real is not True:
        return Attempt(None, ("Ordered monotonicity needs established real-valuedness.",))
    if variable is None or not isinstance(domain, Interval) and domain != S.Reals:
        return Attempt(None, ("Monotonicity currently requires a real interval and one coordinate.",))
    regular = _regular(expression, variable, domain, 1)
    slope = diff(expression, variable)
    positive = _sign(slope, variable, domain, True)
    if regular.condition is not None and regular.condition is not S.false:
        if positive.condition is not None and positive.condition is not S.false:
            return Attempt(
                conjunction(regular.condition, positive.condition),
                (*regular.steps, *positive.steps, "An everywhere positive derivative proves strict increase."),
            )
        nonnegative = _sign(slope, variable, domain, False)
        try:
            polynomial = Poly(expression, variable)
            if polynomial.degree() > 0 and nonnegative.condition is not None and nonnegative.condition is not S.false:
                return Attempt(
                    conjunction(regular.condition, nonnegative.condition),
                    (
                        *regular.steps,
                        *nonnegative.steps,
                        "A nonconstant polynomial with nonnegative derivative is strictly increasing.",
                    ),
                )
        except (ValueError, TypeError, NotImplementedError, PolynomialError):
            pass
    for left in _points(domain):
        for right in _points(domain):
            if (left < right) is S.true:
                difference = simplify(expression.subs(variable, right) - expression.subs(variable, left))
                if (difference > 0) is S.false and difference.is_finite is True:
                    return Attempt(
                        S.false,
                        (f"The ordered points {left}, {right} violate strict increase.",),
                        ((variable, left), (Dummy("second_point", real=True), right)),
                    )
    return Attempt(None, (*regular.steps, "The derivative certificate did not settle strict monotonicity."))


def _convex(expression, variable, domain) -> Attempt:
    if expression.is_real is not True:
        defined = _defined(expression, variable, domain)
        return (
            defined
            if defined.condition is S.false
            else Attempt(None, ("Convexity needs established real-valuedness.",))
        )
    if expression == Abs(variable) and (domain == S.Reals or isinstance(domain, Interval)):
        return Attempt(S.true, ("The triangle inequality proves convexity of absolute value.",))
    regular = _regular(expression, variable, domain, 2)
    if regular.condition is not None and regular.condition is not S.false:
        curvature = _sign(diff(expression, variable, 2), variable, domain, False)
        return Attempt(
            conjunction(regular.condition, curvature.condition), (*regular.steps, *curvature.steps), curvature.witness
        )
    defined = _defined(expression, variable, domain)
    if defined.condition is S.false:
        return defined
    return Attempt(None, (*regular.steps, "Failure of a C2 certificate alone does not refute convexity."))


def _uniform(expression, variable, domain, bound, hint) -> Attempt:
    defined = _defined(expression, variable, domain)
    if defined.condition is S.false or defined.condition is None:
        return defined
    if variable is None or variable not in expression.free_symbols:
        return Attempt(
            conjunction(defined.condition, expression < bound),
            (f"The coordinate-independent supremum is {expression}.",),
        )
    if hint is None:
        return Attempt(None, ("Supply a checked uniform upper-bound hint; a pointwise inequality is insufficient.",))
    if variable in hint.upper.free_symbols:
        return Attempt(None, ("A uniform bound must be independent of the quantified coordinate.",))
    upper = _sign(simplify(hint.upper - expression), variable, domain, False)
    gap = hint.upper < bound
    if upper.condition is S.false or gap is S.false:
        return Attempt(None, ("The proposed upper bound does not certify this claim.",))
    return Attempt(
        conjunction(defined.condition, upper.condition, gap), (*upper.steps, f"sup <= {hint.upper} < {bound}.")
    )


def attempt(
    claim: Claim,
    expression: Basic,
    variable: Symbol | None,
    expected: Expr | None,
    assumptions: tuple[Basic, ...],
    hint: ProofHint | None,
) -> Attempt:
    if variable is not None:
        declared = S.Reals
        if variable.is_positive is True:
            declared = Interval.open(0, S.Infinity)
        elif variable.is_negative is True:
            declared = Interval.open(S.NegativeInfinity, 0)
        elif variable.is_nonnegative is True:
            declared = Interval(0, S.Infinity)
        elif variable.is_nonpositive is True:
            declared = Interval(S.NegativeInfinity, 0)
        if claim.domain.is_subset(declared) is not True:
            return Attempt(None, ("The requested domain exceeds the coordinate's declared domain.",))
        if claim.at is not None and claim.domain.contains(claim.at) is not S.true:
            return Attempt(None, ("The requested evaluation point is not established inside the claim's domain.",))
    if expression.has(AppliedUndef) and claim.property not in {"predicate", "exact_form"}:
        return Attempt(None, ("The subject contains an uninterpreted function.",))
    derivative_guards = []

    def derivatives(value, points):
        if isinstance(value, Subs):
            local = {
                symbol: point.subs(points, simultaneous=True) for symbol, point in zip(value.variables, value.point)
            }
            yield from derivatives(value.expr, {**points, **local})
            for point in value.point:
                yield from derivatives(point, points)
            return
        if isinstance(value, Derivative):
            variables = tuple(dict.fromkeys(value.variables))
            if len(variables) != 1:
                yield Attempt(None, ("Mixed derivative regularity remains unresolved.",))
                return
            variable = variables[0]
            curve = value.expr.subs({symbol: point for symbol, point in points.items() if symbol != variable})
            yield _differentiable(curve, variable, claim.domain, len(value.variables), points.get(variable))
        for arg in value.args:
            yield from derivatives(arg, points)

    for regularity in derivatives(expression, {}):
        if regularity.condition is None or regularity.condition is S.false:
            return regularity
        derivative_guards.append(regularity.condition)
    if expression.has(Derivative):
        expression = expression.doit()
    specialized, reverse = _specialize(expression, variable, assumptions)
    specialized_expected = expected
    forward = {old: new for new, old in reverse.items()}
    if expected is not None:
        specialized_expected = expected.xreplace(forward)
    specialized_hint = None if hint is None else ProofHint(hint.upper.xreplace(forward))
    kind, expression = claim.property, specialized
    if kind == "predicate":
        result = Attempt(
            truth(expression, tuple(item.xreplace(forward) for item in assumptions)),
            ("The parameter predicate is retained or established by the explicit assumptions.",),
        )
    elif not isinstance(expression, Expr):
        result = Attempt(None, ("This property needs a scalar expression.",))
    elif kind == "positive":
        result = _sign(expression, variable, claim.domain, True)
    elif kind == "nonnegative":
        result = _sign(expression, variable, claim.domain, False)
    elif kind == "nonzero":
        result = _nonzero(expression, variable, claim.domain)
    elif kind == "smooth":
        result = _regular(expression, variable, claim.domain, claim.order)
    elif kind == "differentiable":
        result = _differentiable(
            expression, variable, claim.domain, claim.order, None if claim.at is None else claim.at.xreplace(forward)
        )
    elif kind in {"increasing", "decreasing"}:
        result = _monotone(expression if kind == "increasing" else -expression, variable, claim.domain)
    elif kind in {"convex", "concave"}:
        result = _convex(expression if kind == "convex" else -expression, variable, claim.domain)
    elif kind == "limit":
        body = _defined_part(expression)
        guard = _parameter_guards(body, variable)
        if body != expression:
            validity = Or(*(condition for _, condition in expression.args[:-1]))
            if variable not in validity.free_symbols:
                guard = conjunction(guard, validity)
            else:
                try:
                    if validity.as_set().sup != S.Infinity:
                        return Attempt(None, ("An unbounded right-hand domain has not been established.",))
                except (NotImplementedError, ValueError, TypeError):
                    return Attempt(None, ("The eventual parameter-dependent domain is unresolved.",))
        try:
            value = limit(body, variable, S.Infinity)
            condition = None if value.has(Limit, AppliedUndef, AccumBounds) else Eq(value, specialized_expected)
            result = Attempt(
                conjunction(guard, condition), (f"Right-tail limit on the function's eventual domain: {value}.",)
            )
        except (NotImplementedError, ValueError, TypeError):
            result = Attempt(None, ("The parameter-dependent tail limit is unresolved.",))
    elif kind == "exact_form":
        if specialized_expected is None:
            result = Attempt(None, ("No expected form was supplied.",))
        else:
            difference = simplify(expression - specialized_expected)
            guards = conjunction(
                _defined(expression, variable, claim.domain).condition,
                _defined(specialized_expected, variable, claim.domain).condition,
            )
            if difference == 0:
                result = Attempt(guards, ("The expressions agree identically under the retained domain guards.",))
            else:
                witness: tuple[tuple[Symbol, Expr], ...] = ()
                for point in _points(claim.domain) if variable is not None else ():
                    value = simplify(difference.subs(variable, point))
                    if value.is_zero is False and value.is_finite is True:
                        witness = ((variable, point),)
                        break
                result = Attempt(
                    S.false if witness else None,
                    ("A checked point refutes the identity." if witness else "The identity is unresolved.",),
                    witness,
                )
    elif kind == "uniform_bound":
        result = _uniform(expression, variable, claim.domain, claim.bound, specialized_hint)
    elif kind == "curvature_bound":
        regular = _differentiable(expression, variable, claim.domain, 2)
        slope = diff(expression, variable)
        nonzero = _nonzero(slope, variable, claim.domain)
        if nonzero.condition is S.false:
            result = Attempt(
                S.false, (*nonzero.steps, "The original curvature ratio has a zero denominator."), nonzero.witness
            )
        else:
            ratio = simplify(expression * diff(expression, variable, 2) / slope**2)
            upper = _uniform(ratio, variable, claim.domain, claim.bound, specialized_hint)
            result = Attempt(
                conjunction(regular.condition, nonzero.condition, upper.condition),
                (*regular.steps, *nonzero.steps, f"Curvature ratio: {ratio}.", *upper.steps),
            )
    else:
        result = Attempt(None, ("No registered theorem establishes this conclusion.",))
    condition = result.condition.xreplace(reverse) if result.condition is not None else None
    condition = conjunction(condition, *derivative_guards)
    if condition is not None:
        condition = truth(condition, assumptions)
        if variable is not None and variable in condition.free_symbols:
            condition = None
    return Attempt(
        condition,
        result.steps,
        tuple((symbol.xreplace(reverse), value.xreplace(reverse)) for symbol, value in result.witness),
    )

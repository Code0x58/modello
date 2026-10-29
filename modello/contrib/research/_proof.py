"""Conservative scalar proof attempts with explicit domains and premises."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

from sympy import (
    Abs,
    AccumBounds,
    Add,
    And,
    Basic,
    Derivative,
    Dummy,
    Eq,
    Expr,
    FiniteSet,
    Function,
    Integral,
    Interval,
    Limit,
    Max,
    Min,
    Mul,
    Ne,
    Not,
    Or,
    Piecewise,
    Poly,
    Pow,
    Product,
    Q,
    Range,
    S,
    Set,
    Subs,
    Sum,
    Symbol,
    ask,
    ceiling,
    conjugate,
    cos,
    cosh,
    diff,
    erf,
    erfc,
    exp,
    factor,
    floor,
    frac,
    im,
    limit,
    log,
    piecewise_fold,
    preorder_traversal,
    re,
    reduce_inequalities,
    sign,
    simplify,
    sin,
    sinh,
    sympify,
)
from sympy.assumptions import AssumptionsContext
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef
from sympy.logic.algorithms.lra_theory import UnhandledInput
from sympy.logic.boolalg import Boolean
from sympy.logic.inference import satisfiable
from sympy.polys.polyerrors import PolynomialError

from modello._evaluation import _BINDERS, _evaluate, _rebuild
from modello._operators import (
    _bound_coordinates,
    _conditioned,
    _defined_part,
    _denominator_condition,
    _empty_reduction,
    _held_reduction,
)
from modello.solutions import _field_domain, _membership

from ._piecewise import _join, piecewise_regular
from ._records import Requirement


@dataclass(frozen=True)
class Attempt:
    condition: Basic | None
    steps: tuple[str, ...]
    witness: tuple[tuple[Symbol, Expr], ...] = ()
    sufficient_only: bool = False


_TOTAL_FUNCTIONS = (Abs, exp, sin, cos, sinh, cosh, erf, erfc, frac, floor, ceiling, sign, conjugate, re, im)


class _SourceFinite(Boolean):
    """Retain operand definedness without consulting ambient assumptions."""

    is_Atom = True

    def __new__(cls, operand):
        body = _defined_part(operand)
        status = _finite_status(body)
        if status is True:
            return _denominator_condition(operand)
        if status is False:
            return S.false
        if body != operand:
            return And(_denominator_condition(operand), cls(body))
        return super().__new__(cls, operand)

    def _numpycode(self, printer):
        return self._code(printer, "numpy")

    def _pythoncode(self, printer):
        return self._code(printer, "cmath")

    def _code(self, printer, module):
        # Backend values alone do not establish arbitrary source domains:
        # numpy.angle(0), for example, extends SymPy's undefined arg(0).
        if any(part.func not in (*_TOTAL_FUNCTIONS, log) for part in self.args[0].atoms(Function)):
            raise NotImplementedError("Numerical compilation cannot establish this operand's source domain")
        return f"{printer._module_format(module + '.isfinite')}({printer._print(self.args[0])})"


def _finite_status(operand):
    # The proof walker must establish the original derivative scope before
    # native evaluation can supply a finite-looking value at a nonsmooth point.
    if operand.has(Derivative):
        return None
    if not operand.free_symbols and operand.has(Function):
        operand = operand.doit()
    if operand.is_finite is False or operand in (S.NaN, S.ComplexInfinity):
        return False
    # Generic finite flags miss holes such as log(0) and arg(0). Certify only
    # fixed values or functions known to be defined on their finite operands.
    if operand.is_finite is True and all(
        part.func in _TOTAL_FUNCTIONS
        or not part.free_symbols
        or part.func is log
        and part.args[0].is_finite is True
        and part.args[0].is_zero is False
        for part in operand.atoms(Function)
    ):
        return True
    return None


def guard_source(expression: Basic) -> Basic:
    """Keep operand definedness through arithmetic collapse and native binders.

    The private guard stays opaque to ambient assumptions during binding.
    Proofs interpret it on their own explicitly quantified domains.
    """

    @cache
    def visit(value):
        if not value.args or isinstance(value, _SourceFinite):
            return value
        args = tuple(visit(arg) for arg in value.args)
        if isinstance(value, (Sum, Product, Integral)) and args != value.args:
            # Their constructors fold Piecewise through nested local binders.
            result = _held_reduction(value, args)
        elif not isinstance(value, Expr) or isinstance(value, (*_BINDERS, Piecewise)):
            result = value.func(*args) if args != value.args else value
        else:
            result = _rebuild(value, args, _denominator_condition(value)) if args != value.args else value
        if isinstance(result, Piecewise):
            result = piecewise_fold(result)
        if (
            isinstance(value, Expr)
            and not isinstance(value, (Add, Mul, Piecewise, *_BINDERS))
            and (isinstance(value, Function) or value.is_finite is not True)
            and not (isinstance(value, Function) and value.func in _TOTAL_FUNCTIONS)
        ):
            return _conditioned(result, _SourceFinite(value))
        return result

    return visit(expression)


def _guard_scope(condition, variable, domain):
    """A sufficient coordinate scope and retained parameter conditions for source guards."""
    parameter_guards = []
    unknown = False

    def finite(predicate):
        nonlocal unknown
        operand = predicate.args[0]
        if variable is None or variable not in operand.free_symbols:
            return predicate
        guard = _parameter_guards(operand, variable)
        try:
            allowed = continuous_domain(operand, variable, domain)
            if guard is None:
                unknown = True
                return S.true
            parameter_guards.append(guard)
            return allowed.contains(variable)
        except (NotImplementedError, TypeError, ValueError):
            unknown = True
            return S.true

    sufficient = condition.replace(lambda part: isinstance(part, _SourceFinite), finite)
    # Disjoint real-continuity certificates do not establish undefinedness:
    # finite complex operands can still be used on the requested domain.
    if sufficient is S.false and condition.has(_SourceFinite):
        unknown = True
    if variable is None or variable not in sufficient.free_symbols:
        return And(sufficient, *parameter_guards), domain, unknown
    coordinate_guards = []
    for item in And.make_args(sufficient):
        if variable in item.free_symbols:
            coordinate_guards.append(item)
        else:
            parameter_guards.append(item)
    coordinate_guard = And(*coordinate_guards)
    if coordinate_guard.free_symbols - {variable}:
        return And(*parameter_guards), domain, True
    try:
        return And(*parameter_guards), domain.intersect(coordinate_guard.as_set()), unknown
    except (NotImplementedError, TypeError, ValueError):
        return And(*parameter_guards), domain, True


def _source_domain(expression, variable, domain):
    """Separate a guarded value from its original operand-definedness proof."""
    expression = piecewise_fold(expression)
    if expression.is_finite is False or expression in (S.NaN, S.ComplexInfinity):
        return expression, Attempt(S.false, ("The original expression is not finite.",))
    if not isinstance(expression, Piecewise) or expression.args[-1] != (S.NaN, S.true):
        return expression, Attempt(S.true, ())
    validity = Or(*(condition for _, condition in expression.args[:-1]))
    body = _defined_part(expression)
    if variable is not None:
        for point in _points(domain):
            if validity.subs(variable, point) is S.false:
                return body, Attempt(
                    S.false, (f"The original expression is undefined at {point}.",), ((variable, point),)
                )
    parameters, allowed, unknown = _guard_scope(validity, variable, domain)
    if parameters is S.false and not unknown:
        return body, Attempt(S.false, ("The original expression has an undefined source operand.",))
    missing = domain - allowed
    if not unknown and missing.is_empty is True:
        return body, Attempt(parameters, ())
    if not validity.has(_SourceFinite) and not unknown and missing.is_empty is False:
        return body, Attempt(S.false, (f"The original expression is undefined on {missing}.",))
    return body, Attempt(None, ("Original operand definedness on the requested domain remains unresolved.",))


def coordinate_domain(variable: Symbol | None) -> Set:
    """The real domain already guaranteed by a coordinate declaration."""
    return S.Reals if variable is None else _field_domain(variable).intersect(S.Reals)


def _calculus_scope(variable: Symbol | None) -> Attempt | None:
    """Keep discrete coordinates distinct from discrete samples of a real curve."""
    try:
        if coordinate_domain(variable).interior.is_empty is False:
            return None
    except (NotImplementedError, TypeError, ValueError):
        pass
    return Attempt(None, ("The coordinate's intrinsic domain has no established real interior for differentiation.",))


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
    if isinstance(condition, _SourceFinite):
        operand = condition.args[0]
        parameters = tuple(operand.free_symbols)
        if all(parameter.is_finite is True for parameter in parameters) and operand.is_rational_function(*parameters):
            condition = _membership(operand, S.Complexes, parameters)
    try:
        answer = ask(condition, And(*assumptions), context=AssumptionsContext())
        if answer is not None:
            return S.true if answer else S.false
        if satisfiable(And(*assumptions, Not(condition)), use_lra_theory=True) is False:
            return S.true
        if satisfiable(And(*assumptions, condition), use_lra_theory=True) is False:
            return S.false
    except (TypeError, ValueError, NotImplementedError, AssertionError, UnhandledInput):
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
        # Do not let a global predicate certify itself before its conjunction
        # with observed or explicit premises has been checked.
        if ask(combined, context=AssumptionsContext()) is False:
            return False
        result = satisfiable(combined, use_lra_theory=True)
        return False if result is False else True
    except (TypeError, ValueError, NotImplementedError, AssertionError, UnhandledInput):
        return None


def _specialize(expression: Basic, variable: Symbol | None, assumptions: tuple[Basic, ...]):
    substitutions = {}
    for parameter in expression.free_symbols - ({variable} if variable is not None else set()):
        facts = {}
        for name in ("positive", "negative", "nonzero", "nonnegative", "nonpositive", "real", "integer"):
            try:
                answer = ask(getattr(Q, name)(parameter), And(*assumptions))
            except (ValueError, TypeError, NotImplementedError, UnhandledInput):
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


def _singular_witness(expression: Expr, variable: Symbol, domain: Set, missing: Set):
    """A divergent one-sided limit refutes continuity without assuming parameter values."""
    points = tuple(missing) if isinstance(missing, FiniteSet) else _points(missing)
    for point in points[:5]:
        if point.is_real is not True or point.is_finite is not True:
            continue
        for direction, side in (
            ("-", Interval.open(S.NegativeInfinity, point)),
            ("+", Interval.open(point, S.Infinity)),
        ):
            try:
                if domain.intersect(side).closure.contains(point) is not S.true:
                    continue
                value = limit(expression, variable, point, dir=direction)
                if value.is_finite is False:
                    return ((variable, point),)
            except (NotImplementedError, ValueError, TypeError):
                continue
    return ()


def _regular(expression: Expr, variable: Symbol | None, domain: Set, order: int) -> Attempt:
    if order > 0 and (scope := _calculus_scope(variable)) is not None:
        return scope
    expression, source = _source_domain(expression, variable, domain)
    if source.condition is None or source.condition is S.false:
        return source
    if isinstance(expression, Piecewise):
        joined = piecewise_regular(expression, variable, domain, order, _regular)
        if joined is not None:
            return Attempt(conjunction(source.condition, joined.condition), joined.steps, joined.witness)
    guard = _parameter_guards(expression, variable)
    if guard is None:
        return Attempt(None, ("Parameter-dependent definedness is unresolved.",))
    if guard is S.false:
        return Attempt(S.false, ("A parameter denominator is zero.",))
    guard = And(guard, source.condition)
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
                witness = _singular_witness(derivative, variable, domain, missing)
                if witness:
                    return Attempt(
                        S.false,
                        (f"Derivative order {degree} has a nonfinite one-sided limit at {witness[0][1]}.",),
                        witness,
                    )
                return Attempt(
                    None,
                    (
                        f"The continuity domain of derivative order {degree} excludes {missing}, "
                        "but no discontinuity at this derivative order has been established.",
                    ),
                )
            if missing.is_empty is not True:
                return Attempt(None, (f"The continuity domain of derivative order {degree} is unresolved.",))
        except (NotImplementedError, ValueError, TypeError):
            return Attempt(None, (f"The CAS cannot establish continuity of derivative order {degree}.",))
    return Attempt(guard, (f"Continuity domains certify C{order} on {domain}.",))


def _defined(expression: Expr, variable: Symbol | None, domain: Set) -> Attempt:
    """Continuity is sufficient for definedness; its failure is not a refutation."""
    body, source = _source_domain(expression, variable, domain)
    if source.condition is None or source.condition is S.false:
        return source
    regular = _regular(body, variable, domain, 0)
    if regular.condition is not None and regular.condition is not S.false:
        return Attempt(conjunction(source.condition, regular.condition), regular.steps, regular.witness)
    guard = _parameter_guards(body, variable)
    if body.is_finite is True and guard is not None:
        return Attempt(
            conjunction(source.condition, guard),
            ("The expression is finite; continuity is not required for this claim.",),
        )
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
    if (scope := _calculus_scope(variable)) is not None:
        return scope
    expression = piecewise_fold(expression)
    if (
        point is None
        and variable is not None
        and isinstance(expression, Piecewise)
        and expression.args[-1] == (S.NaN, S.true)
    ):
        _, defined = _source_domain(expression, variable, domain)
        if defined.condition is None or defined.condition is S.false:
            return defined
        validity = Or(*(condition for _, condition in expression.args[:-1]))
        _, allowed, unknown = _guard_scope(validity, variable, S.Reals)
        if unknown or domain.is_subset(allowed.interior) is not True:
            return Attempt(None, ("Ordinary derivatives require a defined neighborhood around every requested point.",))
    if point is not None:
        if domain.interior.contains(point) is not S.true:
            return Attempt(None, ("The requested evaluation point has no established neighborhood inside the domain.",))
        if isinstance(expression, Piecewise):
            branches = _join(expression, variable)
            if branches is not None:
                boundary, left, right, _ = branches
                for branch, side in (
                    (left, Interval.open(S.NegativeInfinity, boundary)),
                    (right, Interval.open(boundary, S.Infinity)),
                ):
                    neighborhood = domain.intersect(side)
                    if neighborhood.interior.contains(point) is S.true:
                        return _differentiable(branch, variable, neighborhood, order, point)
            joined = piecewise_regular(expression, variable, domain, order, _regular)
            if joined is not None:
                if joined.condition is not None and joined.condition is not S.false:
                    if joined.condition is S.true or branches is not None and Eq(point, branches[0]) is S.true:
                        return Attempt(joined.condition, joined.steps)
                elif joined.failure_order is not None and joined.failure_order <= order:
                    if (variable, point) in joined.witness:
                        return Attempt(S.false, joined.steps, joined.witness)
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
                    parameters, allowed, unknown = _guard_scope(validity, variable, domain)
                    if unknown:
                        return Attempt(None, ("The local source domain remains unresolved.",))
                    neighborhood &= allowed
                    guard = conjunction(guard, parameters)
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
    if isinstance(expression, Piecewise):
        joined = piecewise_regular(expression, variable, domain, order, _regular)
        if joined is not None and joined.condition is S.false and joined.failure_order is not None:
            if joined.failure_order <= order:
                return Attempt(S.false, joined.steps, joined.witness)
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
                nonconstant = Or(*(Ne(coefficient, 0) for coefficient in polynomial.all_coeffs()[:-1]))
                return Attempt(
                    conjunction(regular.condition, nonnegative.condition, nonconstant),
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
    if variable is None:
        return Attempt(None, ("Convexity currently requires one coordinate and a real interval or singleton.",))
    if isinstance(domain, FiniteSet) and len(domain) == 1:
        value = expression.subs(variable, next(iter(domain)))
        defined = _defined(value, None, S.Reals)
        if defined.condition is S.false:
            return defined
        if value.is_real is True:
            return Attempt(
                defined.condition,
                (*defined.steps, "Every finite real-valued function on a singleton is convex."),
            )
        return Attempt(None, (*defined.steps, "Singleton convexity needs a finite real value at its point."))
    if not isinstance(domain, Interval) and domain != S.Reals:
        return Attempt(None, ("The curvature characterization requires a real interval with nonempty interior.",))
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
    if variable in hint.free_symbols:
        return Attempt(None, ("A uniform bound must be independent of the quantified coordinate.",))
    upper = _sign(simplify(hint - expression), variable, domain, False)
    gap = hint < bound
    if upper.condition is S.false or gap is S.false:
        return Attempt(None, ("The proposed upper bound does not certify this claim.",))
    return Attempt(
        conjunction(defined.condition, upper.condition, gap),
        (*upper.steps, f"sup <= {hint} < {bound}."),
        sufficient_only=True,
    )


def native_domains(value, resolve=lambda value: value, protected=frozenset()):
    """Supported local ranges; complex space marks an unsupported real scope."""
    variables = tuple(_bound_coordinates(value))
    domains = dict.fromkeys(variables, S.Complexes)
    if isinstance(value, Limit):
        point, direction = resolve(value.args[2]), str(value.args[3])
        if point.is_real is True and direction in {"+", "-"}:
            domains[variables[0]] = (
                Interval.open(point, S.Infinity) if direction == "+" else Interval.open(S.NegativeInfinity, point)
            )
    elif len(value.limits) == 1 and len(value.limits[0]) == 3:
        variable, start, end = value.limits[0]
        start, end = resolve(start), resolve(end)
        if isinstance(value, Integral) and start == end:
            domains[variable] = S.EmptySet
        elif not (start.free_symbols | end.free_symbols) & protected:
            if isinstance(value, Integral) and start.is_real is True and end.is_real is True:
                domains[variable] = Interval(Min(start, end), Max(start, end))
            elif start.is_integer is True and end.is_integer is True:
                domains[variable] = Range(Min(start, end + 1), Max(start, end + 1))
    return domains


def derivative_premises(value, variable, domain, points=None):
    """Check retained native derivatives before evaluation or arithmetic collapse."""
    points = {} if points is None else points
    if isinstance(value, (Integral, Sum, Product, Limit)):
        endpoints = (
            (value.args[2],)
            if isinstance(value, Limit)
            else tuple(endpoint for limit in value.limits for endpoint in limit[1:])
        )
        for endpoint in endpoints:
            yield from derivative_premises(endpoint, variable, domain, points)
        scopes = native_domains(value, lambda endpoint: evaluate_native(endpoint.subs(points, simultaneous=True)))
        if any(scope.is_empty is True for scope in scopes.values()):
            return
        if any(scope.is_subset(coordinate_domain(symbol)) is not True for symbol, scope in scopes.items()):
            yield Attempt(None, ("The native binder's coordinate scope remains unresolved.",))
            return
        inner = next(iter(scopes))
        local = {symbol: point for symbol, point in points.items() if symbol not in scopes}
        for premise in derivative_premises(value.args[0], inner, scopes[inner], local):
            condition = premise.condition
            if condition is not None and inner in condition.free_symbols:
                parameters, allowed, unknown = _guard_scope(condition, inner, scopes[inner])
                missing = scopes[inner] - allowed
                condition = (
                    parameters
                    if not unknown and missing.is_empty is True
                    else S.false
                    if not unknown and not condition.has(_SourceFinite) and missing.is_empty is False
                    else None
                )
            if condition is S.false and isinstance(value, (Integral, Limit)):
                condition = None
            yield Attempt(condition, premise.steps, premise.witness if condition is S.false else ())
        return
    if isinstance(value, Subs):
        local = {symbol: point.subs(points, simultaneous=True) for symbol, point in zip(value.variables, value.point)}
        yield from derivative_premises(value.expr, variable, domain, {**points, **local})
        for point in value.point:
            yield from derivative_premises(point, variable, domain, points)
        return
    if isinstance(value, Piecewise) and value.has(Derivative):
        remaining = domain
        guard = S.true
        for branch, condition in value.args:
            reachable = guard.subs(points, simultaneous=True)
            if reachable is S.false:
                return
            for premise in derivative_premises(condition, variable, remaining, points):
                if reachable is S.true or premise.condition is None:
                    yield premise
                else:
                    required = Or(~reachable, premise.condition)
                    yield Attempt(required, premise.steps)
            active = And(guard, condition).subs(points, simultaneous=True)
            guard = And(guard, ~condition)
            if active is S.false:
                continue
            parameters, allowed, unknown = _guard_scope(active, variable, remaining)
            if not unknown and parameters is S.true and (remaining - allowed).is_empty is True:
                yield from derivative_premises(branch, variable, remaining, points)
                return
            if (
                variable is not None
                and active.free_symbols <= {variable}
                and not active.has(_SourceFinite)
                and not unknown
            ):
                selected = remaining.intersect(allowed)
                remaining -= allowed
                if selected.is_empty is None:
                    yield Attempt(None, ("The active native-derivative branch domain is unresolved.",))
                    return
                if selected.is_empty is False:
                    yield from derivative_premises(branch, variable, selected, points)
                if remaining.is_empty is True:
                    return
            else:
                for premise in derivative_premises(branch, variable, remaining, points):
                    condition = None if premise.condition is None else Or(~active, premise.condition)
                    yield Attempt(condition, premise.steps)
        return
    if isinstance(value, Derivative):
        variables = tuple(dict.fromkeys(value.variables))
        if len(variables) != 1:
            yield Attempt(None, ("Mixed derivative regularity remains unresolved.",))
            return
        inner = variables[0]
        curve = value.expr.subs({symbol: point for symbol, point in points.items() if symbol != inner})
        inner_domain = domain if inner == variable and inner not in points else coordinate_domain(inner)
        regularity = _differentiable(curve, inner, inner_domain, len(value.variables), points.get(inner))
        if inner != variable and inner not in points and regularity.condition in (None, S.false):
            yield Attempt(
                None,
                (
                    "An inner derivative needs its own coordinate domain or evaluation scope; "
                    "the outer domain does not apply.",
                ),
            )
        else:
            yield regularity
    for arg in value.args:
        yield from derivative_premises(arg, variable, domain, points)


def evaluate_native(value):
    """Normalize native values, pruning inactive discrete bodies first."""
    if value is None:
        return None
    value = value.replace(
        lambda part: isinstance(part, (Sum, Product)),
        lambda part: part if (empty := _empty_reduction(part)) is None else empty,
    )
    return value.replace(
        lambda part: isinstance(part, (Derivative, Subs)),
        _evaluate,
    )


def attempt(
    claim: Requirement,
    expression: Basic,
    variable: Symbol | None,
    expected: Expr | None,
    assumptions: tuple[Basic, ...],
    hint: Expr | None,
    sources: tuple[Basic, ...] = (),
    hint_sources: tuple[Basic, ...] = (),
) -> Attempt:
    derivative_guards: list[Basic] = []
    hint_guards: list[Basic] = []
    source_values: list[Expr] = []
    hint_values: list[Expr] = []
    for operands, optional in ((sources, False), (hint_sources, True)):
        for operand in operands:
            for regularity in derivative_premises(operand, variable, claim.domain):
                if regularity.condition is None or regularity.condition is S.false:
                    return (
                        Attempt(None, (*regularity.steps, "The proposed hint has unestablished observation sources."))
                        if optional
                        else regularity
                    )
                (hint_guards if optional else derivative_guards).append(regularity.condition)
            value = piecewise_fold(evaluate_native(operand))
            if not isinstance(value, Expr):
                return Attempt(None, ("Observation source checks require scalar expressions.",))
            finite = _defined(_defined_part(value), None, S.Reals)
            if finite.condition is None or finite.condition is S.false:
                return Attempt(None, finite.steps) if optional else finite
            (hint_guards if optional else derivative_guards).append(finite.condition)
            (hint_values if optional else source_values).append(value)
    # Bounds and evaluation scopes have their own native coordinates. Their
    # derivatives must not borrow the quantified subject's interval.
    for value in (claim.domain, claim.bound, claim.at):
        if value is None:
            continue
        for regularity in derivative_premises(value, None, S.Reals):
            if regularity.condition is None or regularity.condition is S.false:
                return regularity
            derivative_guards.append(regularity.condition)
    if hint is not None:
        for regularity in derivative_premises(hint, None, S.Reals):
            if regularity.condition is None or regularity.condition is S.false:
                return Attempt(None, (*regularity.steps, "The proposed bound hint has unestablished native premises."))
            hint_guards.append(regularity.condition)
    claim = replace(
        claim,
        domain=evaluate_native(claim.domain),
        bound=evaluate_native(claim.bound),
        at=evaluate_native(claim.at),
    )
    hint = evaluate_native(hint)
    if variable is not None:
        declared = coordinate_domain(variable)
        if claim.domain.is_subset(declared) is not True:
            return Attempt(None, ("The requested domain exceeds the coordinate's declared domain.",))
        if claim.at is not None and claim.domain.contains(claim.at) is not S.true:
            return Attempt(None, ("The requested evaluation point is not established inside the claim's domain.",))
    for value in (expression, expected):
        if value is None:
            continue
        for regularity in derivative_premises(value, variable, claim.domain):
            if regularity.condition is None or regularity.condition is S.false:
                return regularity
            derivative_guards.append(regularity.condition)

    # Check original local scopes before evaluating substitutions or collapsing
    # empty discrete reductions.
    expression = evaluate_native(expression)
    expected = evaluate_native(expected)
    if expression.has(AppliedUndef) and claim.property not in {"predicate", "exact_form"}:
        return Attempt(None, ("The subject contains an uninterpreted function.",))
    # Source operands are checked before native evaluation, then their domain
    # guards apply to the subject's scope, including local differentiability.
    if source_values:
        condition = And(*(_denominator_condition(value) for value in source_values))
        if isinstance(expression, Expr):
            expression = Piecewise((expression, condition), (S.NaN, True))
        elif isinstance(expression, Boolean):
            expression = And(expression, condition)
    if hint_values and hint is not None:
        condition = And(*(_denominator_condition(value) for value in hint_values))
        hint = Piecewise((hint, condition), (S.NaN, True))
    expression = piecewise_fold(expression)
    expected = piecewise_fold(expected)
    specialized, reverse = _specialize(expression, variable, assumptions)
    specialized_expected = expected
    forward = {old: new for new, old in reverse.items()}
    if expected is not None:
        specialized_expected = expected.xreplace(forward)
    specialized_hint = None if hint is None else hint.xreplace(forward)
    kind, expression = claim.property, specialized
    source = Attempt(S.true, ())
    if isinstance(expression, Expr) and kind not in {"predicate", "limit", "differentiable", "curvature_bound"}:
        expression, source = _source_domain(expression, variable, claim.domain)
    if kind == "exact_form" and specialized_expected is not None:
        specialized_expected, expected_source = _source_domain(specialized_expected, variable, claim.domain)
        source = Attempt(
            conjunction(source.condition, expected_source.condition),
            (*source.steps, *expected_source.steps),
            source.witness or expected_source.witness,
        )
    if source.condition is None or source.condition is S.false:
        result = source
    elif kind == "predicate":
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
            parameters, allowed, unknown = _guard_scope(validity, variable, claim.domain)
            if unknown or allowed.sup != S.Infinity:
                return Attempt(None, ("An unbounded right-hand source domain has not been established.",))
            guard = conjunction(guard, parameters)
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
            difference = simplify(_defined_part(expression) - _defined_part(specialized_expected))
            guards = conjunction(
                _defined(expression, variable, claim.domain).condition,
                _defined(specialized_expected, variable, claim.domain).condition,
            )
            if guards is S.false:
                result = Attempt(S.false, ("At least one side is undefined on the requested domain.",))
            elif difference == 0:
                result = Attempt(guards, ("The expressions agree identically under the retained domain guards.",))
            elif variable is None or variable not in difference.free_symbols:
                result = Attempt(
                    conjunction(guards, Eq(difference, 0)),
                    ("The coordinate-independent equality is checked under the retained domain guards.",),
                )
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
        if (scope := _calculus_scope(variable)) is not None:
            return scope
        regular = _differentiable(expression, variable, claim.domain, 2)
        curve, curve_source = _source_domain(expression, variable, claim.domain)
        slope = diff(curve, variable)
        nonzero = _nonzero(slope, variable, claim.domain)
        if nonzero.condition is S.false:
            result = Attempt(
                S.false, (*nonzero.steps, "The original curvature ratio has a zero denominator."), nonzero.witness
            )
        else:
            ratio = simplify(curve * diff(curve, variable, 2) / slope**2)
            upper = _uniform(ratio, variable, claim.domain, claim.bound, specialized_hint)
            result = Attempt(
                conjunction(regular.condition, curve_source.condition, nonzero.condition, upper.condition),
                (*regular.steps, *nonzero.steps, f"Curvature ratio: {ratio}.", *upper.steps),
                sufficient_only=upper.sufficient_only,
            )
    else:
        result = Attempt(None, ("No registered theorem establishes this conclusion.",))
    hint_condition = conjunction(*hint_guards)
    if hint_condition is not None and (variable is None or variable not in hint_condition.free_symbols):
        hint_condition = truth(hint_condition, assumptions)
    if hint_condition is S.false:
        return Attempt(None, (*result.steps, "The proposed hint's native premises are not established."))
    condition = conjunction(source.condition, result.condition)
    condition = condition.xreplace(reverse) if condition is not None else None
    condition = conjunction(condition, *derivative_guards, hint_condition)
    if condition is not None:
        if variable is not None and variable in condition.free_symbols:
            condition = None
        else:
            condition = truth(condition, assumptions)
    if condition is S.false and result.sufficient_only:
        condition = None
    return Attempt(
        condition,
        result.steps,
        tuple((symbol.xreplace(reverse), value.xreplace(reverse)) for symbol, value in result.witness),
    )

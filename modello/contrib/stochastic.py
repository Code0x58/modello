"""One-dimensional Itô laws and explicit Brownian solution operators."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any

from sympy import (
    And,
    Basic,
    Dummy,
    Eq,
    Expr,
    Integer,
    Or,
    Piecewise,
    S,
    Symbol,
    Tuple,
    default_sort_key,
    diff,
    exp,
    simplify,
)

from modello._evaluation import _ConditionalValue, _evaluate_scalar, _pending, _substitute
from modello._operators import (
    _affine_coefficients,
    _bound_coordinates,
    _conditioned,
    _DeclarationSymbol,
    _scalar_atoms,
    _scalar_expression,
    _ScopedOperator,
    _SymbolicTemplate,
    _value_condition,
)
from modello._symbols import _minimal_assumptions
from modello._values import ScalarValue
from modello.errors import ModelInputError
from modello.series import Series, _SeriesTerm

__all__ = ["ItoProcess"]

if TYPE_CHECKING:
    from modello._expressions import _DefinitionResolver


def _check_point_expression(expression: Expr, state: Expr, time: Symbol) -> None:
    """A process trajectory under a time binder is not a point-state function."""
    if isinstance(state, _SeriesTerm) and any(
        time in _bound_coordinates(part) and state in _scalar_atoms(part, _SeriesTerm)
        for part in _scalar_atoms(expression, Basic)
    ):
        raise ModelInputError("Time-bound process expressions require a path or joint-process model")


class _StateDomain(_DeclarationSymbol):
    """A law's identity and state assumptions, independent of local coordinates."""


class _ProcessContext(_SymbolicTemplate):
    """Keep dynamic dependencies separate from observed point coefficients."""

    @property
    def protected_symbols(self) -> tuple[Basic, ...]:
        return (*self.args[1], *(state for _, state in self.args[0]), *self.args[3])

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset(range(4))

    @property
    def free_symbols(self) -> set[Symbol]:
        return set()

    @property
    def needs_parametrization(self) -> bool:
        unresolved = set(self.args[3])
        return any(value.free_symbols & unresolved for coefficient in self._coefficients for value in coefficient)

    @cached_property
    def _coefficients(self) -> tuple[_ConditionalValue, ...]:
        return tuple(_evaluate_scalar(self._attach(expression)) for expression in self.args[2])

    def others(self, law: ItoProcess) -> set[Basic]:
        own = {law.state, *(state for origin, state in self.args[0] if origin == law._state_domain)}
        return {state for _, state in self.args[0]} - own

    def depends_on_other_states(self, law: ItoProcess) -> bool:
        others = self.others(law)
        for coefficient in self._coefficients:
            if any((value.free_symbols | _scalar_atoms(value, _SeriesTerm)) & others for value in coefficient):
                return True
        return False

    def _attach(self, expression: Basic) -> Basic:
        if isinstance(expression, _ProcessContext):
            return expression
        if isinstance(expression, Piecewise):
            branches = []
            for value, condition in expression.args:
                if condition is S.false:
                    continue
                branches.append((self._attach(value), condition))
                if condition is S.true:
                    break
            return Piecewise(*branches)
        args = tuple(self._attach(arg) for arg in expression.args)
        value = expression.func(*args) if args != expression.args else expression
        if isinstance(value, ItoProcess):
            previous = value._context
            declarations = (
                self.args[0] if previous is None else Tuple(*dict.fromkeys((*previous.args[0], *self.args[0])))
            )
            context = _ProcessContext(
                declarations,
                Tuple(*value.bound_symbols) if previous is None else previous.args[1],
                Tuple(value.drift_expression, value.diffusion_expression) if previous is None else previous.args[2],
                self.args[3] if previous is None else Tuple(*dict.fromkeys((*previous.args[3], *self.args[3]))),
            )
            return value._with_context(context)
        return value


class ItoProcess(_SymbolicTemplate):
    """Declare ``dX = drift*dt + diffusion*dW`` for one standard Wiener driver.

    State and time are independent local calculus coordinates. ``drift(f)``
    includes the Itô correction and explicit time derivative; ``generator(f)``
    gives the spatial generator. Observations and inverse parameter solving use
    ordinary model fields. No distribution is assigned to a supplied sample.
    """

    def __new__(
        cls,
        state: Expr,
        drift: ScalarValue,
        diffusion: ScalarValue,
        time: Symbol,
        driver: Series | None = None,
        _state_domain: _StateDomain | None = None,
        _context: _ProcessContext | Basic | None = None,
    ) -> ItoProcess:
        message = "An Itô law needs a real scalar state, continuous time, drift, and diffusion"
        coefficients = tuple(_scalar_expression(value, message) for value in (drift, diffusion))
        if isinstance(time, Symbol):
            for value in coefficients:
                _check_point_expression(value, state, time)
        if (
            not isinstance(state, (Symbol, _SeriesTerm))
            or state.is_real is False
            or state.is_commutative is not True
            or not isinstance(time, Symbol)
            or time.is_real is False
            or time.is_integer is True
            or state == time
            or any(value.is_commutative is not True or value.is_real is False for value in coefficients)
        ):
            raise ModelInputError(message)
        if isinstance(state, _SeriesTerm) and (state.series.discrete or state.index != time):
            raise ModelInputError("A process state must use this law's continuous time coordinate")
        driver = Series("W", time, real=True) if driver is None else driver
        if not isinstance(driver, Series) or driver.discrete or driver.index != time:
            raise ModelInputError("The Wiener driver must use the same continuous time coordinate")
        if _state_domain is None:
            _state_domain = _StateDomain("state", **{"real": True, **state.assumptions0})
        if not isinstance(_state_domain, _StateDomain):
            raise ModelInputError("An Itô law needs its declared real-state domain")
        context = S.EmptySet if _context is None else _context
        if context is not S.EmptySet and not isinstance(context, _ProcessContext):
            raise ModelInputError("An Itô law needs its process dependency context")
        return Basic.__new__(cls, state, *coefficients, time, driver, _state_domain, context)

    @property
    def state(self) -> Expr:
        return self.args[0]

    @property
    def drift_expression(self) -> Expr:
        return self.args[1]

    @property
    def diffusion_expression(self) -> Expr:
        return self.args[2]

    @property
    def time(self) -> Symbol:
        return self.args[3]

    @property
    def driver(self) -> Series:
        return self.args[4]

    @property
    def _state_domain(self) -> _StateDomain:
        return self.args[5]

    @property
    def _context(self) -> _ProcessContext | None:
        return None if self.args[6] is S.EmptySet else self.args[6]

    def _with_context(self, context: _ProcessContext) -> ItoProcess:
        return self.func(*self.args[:6], context)

    @property
    def _has_context(self) -> bool:
        return self._context is not None

    def _in_context(self, resolver: _DefinitionResolver, protected: frozenset[Basic]) -> ItoProcess:
        if not resolver.attach_context or self._has_context:
            return self
        processes = tuple(template for template in resolver.templates if isinstance(template, ItoProcess))
        dynamic = frozenset(process.state for process in processes)
        unresolved = resolver.cyclic | resolver.ambiguous
        source = resolver._raw_symbolic(self, protected | dynamic | unresolved)
        declarations = Tuple(
            *sorted({Tuple(process._state_domain, process.state) for process in processes}, key=default_sort_key)
        )
        context = _ProcessContext(
            declarations,
            Tuple(*source.bound_symbols),
            Tuple(source.drift_expression, source.diffusion_expression),
            Tuple(*sorted(unresolved, key=default_sort_key)),
        )
        return self._with_context(_substitute(context, resolver.observations))

    @property
    def _state_requirements(self) -> dict[str, bool]:
        return _minimal_assumptions(self._state_domain)

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return (self.state, self.time)

    @property
    def _binding_signature(self) -> tuple[type, tuple[bool, ...]]:
        return type(self), (self._state_domain.is_integer is True, False)

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset(range(5))

    @property
    def free_symbols(self) -> set[Symbol]:
        return (self.drift_expression.free_symbols | self.diffusion_expression.free_symbols) - {self.state, self.time}

    @property
    def equations(self) -> tuple[Basic, ...]:
        return (
            Eq(
                _Differential(self.state),
                self.drift_expression * _Differential(self.time)
                + self.diffusion_expression * _Differential(self.driver[self.time]),
                evaluate=False,
            ),
        )

    def drift(self, expression: ScalarValue) -> Expr:
        """Return ``f_t + a*f_x + b**2*f_xx/2``, expanding model formulas first."""
        return _ItoQuantity(self, expression, 0)

    def diffusion(self, expression: ScalarValue) -> Expr:
        """Return the coefficient ``b*f_x`` of this law's Wiener differential."""
        return _ItoQuantity(self, expression, 1)

    def generator(self, expression: ScalarValue) -> Expr:
        """Return the spatial generator ``a*f_x + b**2*f_xx/2``."""
        return _ItoQuantity(self, expression, 2)

    def quadratic_variation_rate(self, expression: ScalarValue | None = None) -> Expr:
        """Return the ``dt`` coefficient of the transformed quadratic variation."""
        return self.diffusion(self.state if expression is None else expression) ** 2

    def exact(self, initial: ScalarValue, elapsed: ScalarValue, increment: ScalarValue) -> Expr:
        """Exact arithmetic/geometric Brownian value conditional on ``ΔW``.

        Coefficients must be constant additive or proportional to the state.
        ``increment`` denotes a realised Wiener increment over nonnegative
        ``elapsed``; this operation neither samples nor estimates that increment.
        """
        return _ItoSolution(self, initial, elapsed, increment, 0)

    def mean(self, initial: ScalarValue, elapsed: ScalarValue) -> Expr:
        """Conditional mean given the initial state for the supported exact laws."""
        return _ItoSolution(self, initial, elapsed, S.Zero, 1)

    def variance(self, initial: ScalarValue, elapsed: ScalarValue) -> Expr:
        """Conditional variance given the initial state for the supported exact laws."""
        return _ItoSolution(self, initial, elapsed, S.Zero, 2)

    @cached_property
    def _local_state(self) -> Dummy:
        """A local calculus coordinate carrying this law's state premises."""
        return Dummy("state", **self._state_domain.assumptions0)

    @cached_property
    def _local_coefficients(self) -> tuple[_ConditionalValue, _ConditionalValue]:
        drift, diffusion = (
            _evaluate_scalar(expression.xreplace({self.state: self._local_state}))
            for expression in (self.drift_expression, self.diffusion_expression)
        )
        return drift, diffusion

    def _coefficient_condition(self, *, exact: bool = False) -> Basic:
        """Keep coefficient poles, rejecting path-dependent holes in exact laws."""
        condition = And(*(value.condition for value in self._local_coefficients))
        if exact and condition.has(self._local_state, self.time):
            raise NotImplementedError("Exact Itô coefficients must be defined throughout the declared state domain")
        return condition.xreplace({self._local_state: self.state})

    @cached_property
    def _exact_coefficients(self) -> tuple[str, Expr, Expr, Expr]:
        if self._context is not None and self._context.needs_parametrization:
            raise NotImplementedError(
                "Exact Itô coefficients need an explicit parametrization of cyclic or multiply defined quantities"
            )
        if self._context is not None and self._context.depends_on_other_states(self):
            raise NotImplementedError(
                "Exact Itô queries cannot treat other registered process states as constant coefficients"
            )
        x = self._local_state
        a, b = (value.value for value in self._local_coefficients)
        if _pending(a) or _pending(b):
            raise NotImplementedError("Exact Itô coefficients must resolve before evaluating nonzero-time queries")
        if _scalar_atoms(a, _SeriesTerm) or _scalar_atoms(b, _SeriesTerm):
            raise NotImplementedError("Exact Itô coefficients cannot depend on another unmodelled process")
        drift = _affine_coefficients(a, x)
        diffusion = _affine_coefficients(b, x)
        if (
            drift is not None
            and diffusion is not None
            and not any(self.time in value.free_symbols for value in (*drift, *diffusion))
        ):
            mu, a = drift
            sigma, b = diffusion
            if mu == 0 and sigma == 0:
                return "arithmetic", a, b, S.Zero
            if a == 0 and b == 0:
                return "geometric", mu, sigma, S.Zero
            if sigma == 0:
                return "affine", a, b, mu
        raise NotImplementedError(
            "Exact Itô values and moments support constant arithmetic, geometric, or affine additive coefficients"
        )

    def _sympystr(self, printer: Any) -> str:
        return printer.doprint(self.equations[0])

    def _latex(self, printer: Any) -> str:
        terms: list[str] = []
        for coefficient, variable in (
            (self.drift_expression, self.time),
            (self.diffusion_expression, self.driver[self.time]),
        ):
            if coefficient == 0:
                continue
            negative = coefficient.could_extract_minus_sign()
            if negative:
                coefficient = -coefficient
            sign = "-" if negative else "+" if terms else ""
            body = "" if coefficient == 1 else printer._print(coefficient)
            if coefficient.is_Add:
                body = rf"\left({body}\right)"
            terms.append(rf"{sign}{body}\,\mathrm{{d}}{printer._print(variable)}")
        return rf"\mathrm{{d}}{printer._print(self.state)}=" + ("".join(terms) or "0")


class _Differential(Expr):
    is_commutative = True

    def _sympystr(self, printer: Any) -> str:
        return f"d({printer.doprint(self.args[0])})"

    def _latex(self, printer: Any) -> str:
        return rf"\mathrm{{d}}{printer._print(self.args[0])}"


class _ItoQuantity(_ScopedOperator):
    is_commutative = True

    def __new__(
        cls,
        law: ItoProcess,
        expression: ScalarValue,
        operation: int | Integer,
        _coordinates: Tuple | None = None,
    ) -> _ItoQuantity:
        value = _scalar_expression(expression, "An Itô transformation needs a scalar expression")
        _coordinates = Tuple(*law.bound_symbols) if _coordinates is None else _coordinates
        if not isinstance(_coordinates, Tuple) or len(_coordinates) != 2:
            raise ModelInputError("An Itô transformation needs its state and time coordinates")
        _check_point_expression(value, _coordinates[0], _coordinates[1])
        if value.is_commutative is not True:
            raise ModelInputError("An Itô transformation needs a scalar expression")
        terms = _scalar_atoms(value, _SeriesTerm)
        if any(
            term != _coordinates[0] and (not term.series.discrete or _coordinates[1] in term.index.free_symbols)
            for term in terms
        ):
            raise ModelInputError("A one-state Itô transformation cannot differentiate another unmodelled process")
        if law._context is not None and (value.free_symbols | terms) & law._context.others(law):
            raise ModelInputError(
                "A one-state Itô transformation cannot differentiate another registered process state"
            )
        return Expr.__new__(cls, law, value, Integer(operation), _coordinates)

    @property
    def protected_symbols(self) -> tuple[Basic, ...]:
        # The result is a function of these external coordinates. Unlike the
        # law's local coordinates, they must remain free under alpha-renaming.
        return tuple(self.args[3])

    @property
    def scoped_arguments(self) -> frozenset[int]:
        return frozenset({1, 3})

    @property
    def free_symbols(self) -> set[Symbol]:
        law, expression, _, coordinates = self.args
        frame = dict(zip(law.bound_symbols, coordinates, strict=True))
        return (
            expression.free_symbols
            | law.drift_expression.xreplace(frame).free_symbols
            | law.diffusion_expression.xreplace(frame).free_symbols
        )

    @property
    def calculus_expression(self) -> Expr:
        return self.args[1]

    def _in_context(self, resolver: _DefinitionResolver, protected: frozenset[Basic]) -> _ItoQuantity:
        """One-state queries must not freeze another declared process."""
        if not resolver.evaluate_operators:
            return self
        processes = tuple(template for template in resolver.templates if isinstance(template, ItoProcess))
        if not processes:
            return self
        dynamic = frozenset(process.state for process in processes)
        resolved = resolver._raw_symbolic(self, protected | dynamic)
        law = resolved.args[0]
        state, time = resolved.args[3]
        expression = resolved.args[1].xreplace({state: law.state, time: law.time})
        expression = resolver(expression, protected | dynamic | frozenset(law.bound_symbols))
        states = expression.free_symbols | _scalar_atoms(expression, _SeriesTerm)
        if states & (dynamic - {law.state}):
            raise ModelInputError(
                "A one-state Itô transformation cannot differentiate another registered process state"
            )
        return self

    def _eval_subs(self, old: Basic, new: Basic) -> Basic | None:
        if old in self.protected_symbols:
            if not isinstance(new, Expr):
                raise ModelInputError("An Itô observation must be scalar")
            requirements = self.args[0]._state_requirements if old == self.args[3][0] else {"real": True}
            condition = _value_condition(requirements, new)
            if condition is S.false:
                return S.NaN
            value = super()._eval_subs(old, new)
            if value is not None:
                return Piecewise((value, condition), (S.NaN, True))
        return super()._eval_subs(old, new)

    def doit(self, **hints: Any) -> Expr:
        law, expression, operation = self.args[:3]
        state, time = self.args[3]
        x = law._local_state
        transformed = _evaluate_scalar(expression.xreplace({state: x, time: law.time}))
        condition = And(transformed.condition, law._coefficient_condition())
        if condition is S.false:
            return S.NaN
        body = transformed.value
        a, b = (value.value for value in law._local_coefficients)
        result = b * diff(body, x) if operation == 1 else a * diff(body, x) + b**2 * diff(body, x, 2) / 2
        if operation == 0:
            result += diff(body, law.time)
        result = simplify(result.xreplace({x: state, law.time: time}))
        condition = condition.xreplace({x: state, law.state: state, law.time: time})
        return _conditioned(result, condition)

    def _sympystr(self, printer: Any) -> str:
        name = ("ItoDrift", "ItoDiffusion", "Generator")[int(self.args[2])]
        return f"{name}({printer.doprint(self.args[1])})"

    def _latex(self, printer: Any) -> str:
        name = (r"\operatorname{drift}_{\mathrm{I}}", r"\operatorname{diffusion}_{\mathrm{I}}", r"\mathcal{L}")[
            int(self.args[2])
        ]
        return rf"{name}\!\left[{printer._print(self.args[1])}\right]"


class _ItoSolution(_ScopedOperator):
    is_commutative = True

    def __new__(
        cls,
        law: ItoProcess,
        initial: ScalarValue,
        elapsed: ScalarValue,
        increment: ScalarValue,
        operation: int | Integer,
    ) -> _ItoSolution:
        message = "Initial state, elapsed time, and Wiener increment must be scalar"
        values = tuple(_scalar_expression(value, message) for value in (initial, elapsed, increment))
        if any(value.is_commutative is not True or value.is_real is False for value in values):
            raise ModelInputError(message)
        return Expr.__new__(cls, law, *values, Integer(operation))

    @property
    def bound_symbols(self) -> tuple[Basic, ...]:
        return self.args[0].bound_symbols

    def doit(self, **hints: Any) -> Expr:
        law, initial, elapsed, increment, operation = self.args
        if elapsed.is_nonnegative is False:
            return S.NaN
        domain = And(_value_condition(law._state_requirements, initial), elapsed >= 0)
        if operation == 0:
            domain = And(domain, Or(elapsed > 0, Eq(increment, 0)))
        if domain is S.false:
            return S.NaN
        if elapsed.is_zero is True:
            result = S.Zero if operation == 2 else initial
        else:
            domain = And(domain, law._coefficient_condition(exact=True))
            if domain is S.false:
                return S.NaN
            result = self._result()
        if operation == 0:
            domain = And(domain, _value_condition(law._state_requirements, result))
        return _conditioned(simplify(result), domain)

    def _result(self) -> Expr:
        """Calculate a supported nonzero-time query after validating its domain."""
        law, initial, elapsed, increment, operation = self.args
        kind, a, b, gradient = law._exact_coefficients
        if kind == "arithmetic":
            mean = initial + a * elapsed
            result = b**2 * elapsed if operation == 2 else mean
            if operation == 0:
                result += b * increment
        elif kind == "affine":
            if operation == 0 and b.is_zero is not True:
                raise NotImplementedError(
                    "An exact affine additive path requires a weighted Wiener integral; use its conditional moments"
                )
            rate = gradient if operation != 2 else 2 * gradient
            integral = Piecewise((elapsed, Eq(rate, 0)), ((exp(rate * elapsed) - 1) / rate, True))
            result = initial * exp(gradient * elapsed) + a * integral if operation != 2 else b**2 * integral
        else:
            mean = initial * exp(a * elapsed)
            if operation == 2:
                result = mean**2 * (exp(b**2 * elapsed) - 1)
            elif operation == 0:
                result = initial * exp((a - b**2 / 2) * elapsed + b * increment)
            else:
                result = mean
        return result

    def _sympystr(self, printer: Any) -> str:
        name = ("ItoValue", "ItoMean", "ItoVariance")[int(self.args[4])]
        return f"{name}(" + ", ".join(printer.doprint(value) for value in (self.args[0].state, *self.args[1:4])) + ")"

    def _latex(self, printer: Any) -> str:
        law, initial, elapsed, increment, operation = self.args
        name = (r"\operatorname{value}_{\mathrm{I}}", r"\mathbb{E}", r"\mathbb{V}")[int(self.args[4])]
        base = printer._print(law.state.series if isinstance(law.state, _SeriesTerm) else law.state)
        condition = rf"{base}_{{{printer._print(law.time)}}}={printer._print(initial)}"
        if operation == 0:
            condition += rf",\;\Delta {printer._print(law.driver)}={printer._print(increment)}"
        return rf"{name}\!\left[{base}_{{{printer._print(law.time + elapsed)}}}\;\middle|\;{condition}\right]"

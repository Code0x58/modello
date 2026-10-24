"""One-dimensional Itô laws and explicit Brownian solution operators."""

from __future__ import annotations

from typing import Any

from sympy import And, Basic, Dummy, Eq, Expr, Integer, Or, Piecewise, S, Symbol, Tuple, diff, exp, simplify, sympify

from modello._evaluation import _evaluate
from modello._operators import (
    _affine_coefficients,
    _DeclarationSymbol,
    _ScopedOperator,
    _SymbolicTemplate,
    _value_condition,
)
from modello._symbols import _minimal_assumptions
from modello._values import ScalarValue
from modello.errors import ModelInputError
from modello.series import Series, _SeriesTerm

__all__ = ["ItoProcess"]


class _StateDomain(_DeclarationSymbol):
    """A law's identity and state assumptions, independent of local coordinates."""


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
    ) -> ItoProcess:
        drift, diffusion = sympify(drift), sympify(diffusion)
        if (
            not isinstance(state, (Symbol, _SeriesTerm))
            or state.is_real is False
            or state.is_commutative is not True
            or not isinstance(time, Symbol)
            or time.is_real is False
            or time.is_integer is True
            or state == time
            or any(
                not isinstance(value, Expr)
                or value.is_Matrix
                or value.is_commutative is not True
                or value.is_real is False
                for value in (drift, diffusion)
            )
        ):
            raise ModelInputError("An Itô law needs a real scalar state, continuous time, drift, and diffusion")
        if isinstance(state, _SeriesTerm) and (state.series.discrete or state.index != time):
            raise ModelInputError("A process state must use this law's continuous time coordinate")
        driver = Series("W", time, real=True) if driver is None else driver
        if not isinstance(driver, Series) or driver.discrete or driver.index != time:
            raise ModelInputError("The Wiener driver must use the same continuous time coordinate")
        if _state_domain is None:
            _state_domain = _StateDomain("state", **{"real": True, **state.assumptions0})
        if not isinstance(_state_domain, _StateDomain):
            raise ModelInputError("An Itô law needs its declared real-state domain")
        return Basic.__new__(cls, state, drift, diffusion, time, driver, _state_domain)

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

    def _state_variable(self) -> Dummy:
        """Use the real-state premise together with the declared state domain."""
        return Dummy("state", **self._state_domain.assumptions0)

    def _coefficients_at(self, state: Symbol) -> tuple[Expr, Expr]:
        """Evaluate both coefficients in the same local calculus coordinate."""
        drift, diffusion = (
            _evaluate(expression.xreplace({self.state: state}))
            for expression in (self.drift_expression, self.diffusion_expression)
        )
        return drift, diffusion

    def _coefficients(self) -> tuple[str, Expr, Expr, Expr]:
        x = self._state_variable()
        a, b = self._coefficients_at(x)
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
        expression = sympify(expression)
        _coordinates = Tuple(*law.bound_symbols) if _coordinates is None else _coordinates
        if not isinstance(_coordinates, Tuple) or len(_coordinates) != 2:
            raise ModelInputError("An Itô transformation needs its state and time coordinates")
        if not isinstance(expression, Expr) or expression.is_Matrix or expression.is_commutative is not True:
            raise ModelInputError("An Itô transformation needs a scalar expression")
        if any(term != _coordinates[0] for term in expression.atoms(_SeriesTerm)):
            raise ModelInputError("A one-state Itô transformation cannot differentiate another unmodelled process")
        return Expr.__new__(cls, law, expression, Integer(operation), _coordinates)

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

    def doit(self, **hints: Any) -> Expr:
        law, expression, operation = self.args[:3]
        state, time = self.args[3]
        x = law._state_variable()
        body = _evaluate(expression.xreplace({state: x, time: law.time}))
        a, b = law._coefficients_at(x)
        result = b * diff(body, x) if operation == 1 else a * diff(body, x) + b**2 * diff(body, x, 2) / 2
        if operation == 0:
            result += diff(body, law.time)
        return simplify(result.xreplace({x: state, law.time: time}))

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
        values = tuple(sympify(value) for value in (initial, elapsed, increment))
        if any(
            not isinstance(value, Expr) or value.is_Matrix or value.is_commutative is not True or value.is_real is False
            for value in values
        ):
            raise ModelInputError("Initial state, elapsed time, and Wiener increment must be scalar")
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
            result = self._result()
        return Piecewise((simplify(result), domain), (S.NaN, True))

    def _result(self) -> Expr:
        """Calculate a supported nonzero-time query after validating its domain."""
        law, initial, elapsed, increment, operation = self.args
        kind, a, b, gradient = law._coefficients()
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

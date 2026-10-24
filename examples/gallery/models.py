"""Six independent examples; each adds one kind of relationship."""

from sympy import Derivative, Integral, log

from examples.portfolio.models import Execution, HoldingState
from modello import (
    Coordinate,
    Index,
    InstanceDummy,
    ItoProcess,
    Modello,
    Recurrence,
    Series,
    Vector,
    model_recurrence,
    model_vector,
    vector,
)


class VectorSummary(Modello):
    """Ordinary scalar reductions over an observation-sized vector."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    samples = Vector("x", size=N, real=True)
    total = samples.sum()
    product = samples.product()
    squared_norm = samples.dot(samples)


class ObservedAutoregression(Modello):
    """A relative law with an independently chosen observation window."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    opening = InstanceDummy("opening", real=True)
    retention = InstanceDummy("retention", real=True)
    innovations = Vector("epsilon", size=N, real=True)
    n = Index("n", nonnegative=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], retention * T[n - 1] + innovations[n - 1], initial={0: opening})
    levels = law.window(0, N)
    closing = T[N]


class AccountSequence(Modello):
    """Executions connect typed account states through relative field equations."""

    N = InstanceDummy("N", integer=True, nonnegative=True)
    fills = model_vector(Execution, size=N)
    states = model_vector(HoldingState, size=N + 1)
    n = Index("n")
    law = model_recurrence(
        states[n],
        quantity=states[n - 1].quantity + fills[n - 1].quantity,
        cash=states[n - 1].cash - fills[n - 1].notional - fills[n - 1].fee,
    )
    terminal = states[-1]
    total_fees = fills.project(lambda fill: fill.fee).sum()


class LinearExposure(Modello):
    """An integral under an explicitly stated interpolation rule."""

    opening = InstanceDummy("opening", real=True)
    closing = InstanceDummy("closing", real=True)
    duration = InstanceDummy("duration", positive=True)
    t = Coordinate("t", real=True)
    exposure = Integral(opening + (closing - opening) * t / duration, (t, 0, duration))


class LinearTrend(Modello):
    """Vector residuals and derivatives of an unweighted squared-error objective."""

    N = InstanceDummy("N", integer=True, positive=True)
    times = Vector("t", size=N, real=True)
    observations = Vector("y", size=N, real=True)
    slope = InstanceDummy("slope", real=True)
    intercept = InstanceDummy("intercept", real=True)
    i = Index("i")
    fitted = vector(intercept + slope * times[i], (i, 0, N - 1))
    residuals = observations - fitted
    squared_error = residuals.dot(residuals)
    slope_gradient = Derivative(squared_error, slope)
    intercept_gradient = Derivative(squared_error, intercept)


class GeometricDiffusion(Modello):
    """Itô transforms and conditional moments of geometric Brownian motion."""

    t = Coordinate("t", real=True)
    S = Series("S", t, positive=True)
    growth = InstanceDummy("growth", real=True)
    volatility = InstanceDummy("volatility", nonnegative=True)
    opening = InstanceDummy("opening", positive=True)
    duration = InstanceDummy("duration", nonnegative=True)
    wiener_increment = InstanceDummy("wiener_increment", real=True)
    process = ItoProcess(S[t], growth * S[t], volatility * S[t], t)
    log_drift = process.drift(log(S[t]))
    log_diffusion = process.diffusion(log(S[t]))
    closing = process.exact(opening, duration, wiener_increment)
    mean = process.mean(opening, duration)
    variance = process.variance(opening, duration)

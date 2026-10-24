"""The vector and recurrent models used in the four-step feature walkthrough."""

from sympy import Derivative

from examples.portfolio.models import Execution, HoldingState
from modello import Index, InstanceDummy, Modello, Vector, model_recurrence, model_vector, vector


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

"""The calibration model and shared account declaration for the walkthrough."""

from sympy import Derivative

from examples.portfolio.models import AccountSequence as AccountSequence
from modello import Index, InstanceDummy, Modello, Vector, vector


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

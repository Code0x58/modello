"""Domain-extension examples, separate from the fundamental model declarations."""

from sympy import log

from modello import Coordinate, InstanceDummy, Modello, Series
from modello.contrib.stochastic import ItoProcess


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

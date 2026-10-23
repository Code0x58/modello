"""Relative recurrences and continuous laws in ordinary flat models."""

from __future__ import annotations

from functools import lru_cache

from sympy import Expr, log

from examples.symbolic import SymbolicModel
from modello import Coordinate, Index, InstanceDummy, ItoProcess, Recurrence, Series, Vector


class RelativeGrowth(SymbolicModel):
    """One recurrence, queried at a variable position and a separate sample window."""

    n = Index("n", nonnegative=True)
    T = Series("T", n, positive=True)
    opening = InstanceDummy("opening", positive=True)
    ratio = InstanceDummy("ratio", positive=True)
    periods = InstanceDummy("periods", integer=True, nonnegative=True)
    law = Recurrence(T[n], ratio * T[n - 1], initial={0: opening})
    closing = T[periods]
    samples = law.window(0, 4)


class AutoregressivePath(SymbolicModel):
    """A deterministic conditional path; supplied innovations carry no distribution."""

    opening = InstanceDummy("opening", real=True)
    retention = InstanceDummy("retention", real=True)
    innovations: Vector
    levels: Vector
    closing: Expr


@lru_cache(maxsize=None, typed=True)
def autoregressive_path(periods: int) -> type[AutoregressivePath]:
    """Choose the available innovation window without rewriting the relative law."""
    if isinstance(periods, bool) or not isinstance(periods, int) or periods < 0:
        raise ValueError("The innovation count must be a nonnegative integer")

    class Path(AutoregressivePath):
        n = Index("n", nonnegative=True)
        T = Series("T", n, real=True)
        innovations = Vector("epsilon", size=periods, real=True)
        law = Recurrence(
            T[n], AutoregressivePath.retention * T[n - 1] + innovations[n - 1], initial={0: AutoregressivePath.opening}
        )
        levels = law.window(0, periods)
        closing = law.at(periods)

    return Path


class GeometricDiffusion(SymbolicModel):
    """A geometric Brownian law, transformed coefficients, and conditional moments."""

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


class MeanReversion(SymbolicModel):
    """Exact conditional moments of an affine additive diffusion, including zero rate."""

    t = Coordinate("t", real=True)
    X = Series("X", t, real=True)
    rate = InstanceDummy("rate", nonnegative=True)
    equilibrium = InstanceDummy("equilibrium", real=True)
    volatility = InstanceDummy("volatility", nonnegative=True)
    opening = InstanceDummy("opening", real=True)
    duration = InstanceDummy("duration", nonnegative=True)
    process = ItoProcess(X[t], rate * (equilibrium - X[t]), volatility, t)
    mean = process.mean(opening, duration)
    variance = process.variance(opening, duration)


class BrownianCompensator(SymbolicModel):
    """Infer a quadratic compensator by imposing zero Itô drift."""

    t = Coordinate("t", real=True)
    W = Coordinate("W", real=True)
    rate = InstanceDummy("rate", real=True)
    process = ItoProcess(W, 0, 1, t)
    drift = process.drift(W**2 - rate * t)


if __name__ == "__main__":
    from sympy import Rational

    growth = RelativeGrowth("growth", opening=1, ratio=2, periods=10)
    print("T_10:", growth.closing)
    print("Separate T_0 through T_4 window:", growth.samples)
    path = autoregressive_path(4)(
        "inverse", opening=2, retention=Rational(1, 2), innovations=[1, 0, None, 0], closing=Rational(5, 4)
    )
    print("Inferred innovations:", path.innovations)
    diffusion = GeometricDiffusion(
        "diffusion", growth=Rational(1, 10), volatility=Rational(1, 5), opening=100, duration=1, wiener_increment=0
    )
    print("Log drift and diffusion:", diffusion.log_drift, diffusion.log_diffusion)
    print("Conditional mean and variance:", diffusion.mean, diffusion.variance)
    print("Brownian compensator:", BrownianCompensator("compensated", drift=0).rate)

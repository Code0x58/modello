"""Ideal projectile motion and the envelope of its fixed-speed trajectories."""

from __future__ import annotations

from sympy import Eq, diff

from modello import InstanceDummy, Modello, ModelProblem, ScalarValue, SolutionFamily, component, inspect_model


class FreeFlight(Modello):
    """Where the throw would be at the same time without gravity, in metres."""

    speed = InstanceDummy("speed", positive=True)
    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)
    z = InstanceDummy("z", real=True)
    time_squared = (x**2 + y**2 + z**2) / speed**2


class Projectile(Modello):
    """A throw from the origin under uniform gravity, without air resistance."""

    free = component(FreeFlight)
    gravity = InstanceDummy("gravity", positive=True)
    x = free.x
    y = free.y
    target_height = InstanceDummy("target_height", real=True)
    height = free.z - gravity * free.time_squared / 2
    clearance = height - target_height

    @classmethod
    def prepare(cls, *, speed: ScalarValue = 20, gravity: ScalarValue = 10) -> ModelProblem[Projectile]:
        """Fix speed (m/s) and gravity (m/s²), leaving destinations unknown."""
        return inspect_model(cls, "projectile", free=FreeFlight("without gravity", speed=speed), gravity=gravity)

    @classmethod
    def envelope(cls, *, speed: ScalarValue = 20, gravity: ScalarValue = 10) -> SolutionFamily[Projectile]:
        """Maximise height over launch directions at each horizontal location."""
        family = cls.prepare(speed=speed, gravity=gravity).solutions()
        flight = family.symbols
        height = family.expression(flight.height)
        return family.where(Eq(diff(height, flight.free.z), 0))

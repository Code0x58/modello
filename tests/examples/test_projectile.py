"""Projectile envelopes preserve physical reachability and coupled quantities."""

from sympy import Rational, S, Tuple

from examples.projectile import FreeFlight, Projectile
from examples.solution_views import ModelView


def test_projectile_envelope_preserves_physical_reachability_and_coupling():
    forward = Projectile("forward", free=FreeFlight("free", speed=20, x=16, y=0, z=32), gravity=10, target_height=0)
    assert forward.free.time_squared == Rational(16, 5)
    assert forward.height == 16
    problem = Projectile.prepare()
    shot = problem.symbols
    x, y, z = shot.x, shot.y, shot.free.z
    height = problem.solutions(x, y, z, shot.target_height).expression(shot.height)
    envelope = Projectile.envelope()
    safety = envelope.symbols
    ceiling = envelope.expression(safety.height)
    assert envelope.expression(safety.free.z) == 40
    assert (ceiling - (20 - safety.x**2 / 80 - safety.y**2 / 80)).expand() == 0
    rebased = ceiling.subs({safety.x: x, safety.y: y})
    assert (rebased - height - (z - 40) ** 2 / 80).expand() == 0
    region = envelope.where(safety.clearance >= 0, safety.target_height >= 0)
    assert tuple(field.symbol for field in region.parameters) == (safety.x, safety.y, safety.target_height)
    points = ((0, 0, 20), (40, 0, 0), (0, 0, 10), (0, 0, 21), (41, 0, 0), (0, 0, -1))
    for point, expected in zip(points, (S.true, S.true, S.true, S.false, S.false, S.false), strict=True):
        assert region.parameter_domain.contains(Tuple(*point)) is expected
    anchor = Projectile("reference", free=FreeFlight("free", speed=20, x=0, y=0, z=40), gravity=10, target_height=0)
    planar = region.section(safety.x, safety.target_height, through=anchor)
    view = ModelView(planar.parameters, planar)
    assert view.control_dimension is None
    assert view.image_dimension(safety.x, safety.target_height) is None
    assert planar.parameter_domain.contains(Tuple(0, 20)) is S.true
    assert planar.parameter_domain.contains(Tuple(40, 1)) is S.false
    slower = Projectile.envelope(speed=10, gravity=10)
    other = slower.symbols
    assert (slower.expression(other.height) - (5 - other.x**2 / 20 - other.y**2 / 20)).expand() == 0

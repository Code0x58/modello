"""Calibration must constrain the same impact used by the planned execution."""

from sympy import Interval, Rational

from examples.solution_families import Execution, ImpactStudy


def test_bounded_calibration_predicts_cost_and_retains_typed_components():
    family = ImpactStudy.calibrated()
    p = family.problem
    study = family.symbols
    assert family.parameters == (p.field("impact"),)
    assert family.values(study.impact) == Interval(Rational(1, 4), Rational(3, 4))
    assert family.values(study.planned.cost) == Interval(1026, 1076)
    chosen = family.observe(impact=Rational(1, 2)).one()
    assert isinstance(chosen, ImpactStudy)
    assert isinstance(chosen.recorded, Execution) and isinstance(chosen.planned, Execution)
    assert chosen.recorded.impact == chosen.planned.impact == Rational(1, 2)
    assert chosen.recorded.cost == 201
    assert chosen.planned.cost == 1051

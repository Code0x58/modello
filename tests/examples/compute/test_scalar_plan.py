"""Named symbolic execution at points and on aligned series."""

from datetime import date

import pytest
from sympy import Symbol

from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
from examples.compute.series import Series


def test_named_plan_agrees_between_points_and_vectors():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({"price": p, "quantity": q}), NamedMap({"value": p * q, "ratio": p / q}))
    assert plan.evaluate(NamedMap({"quantity": 2, "price": 10})) == {"value": 20, "ratio": 5}
    times = (date(2026, 1, 1), date(2026, 1, 2))
    result = plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times, (10, 12))}))
    assert result["value"].values == (20, 48)
    assert result["ratio"].values == (5, 3)
    with pytest.raises(ValueError, match="same observation times"):
        plan.over(times, NamedMap({"quantity": Series(times, (2, 4)), "price": Series(times[:1], (10,))}))
    with pytest.raises(ValueError, match="Member names differ"):
        plan.evaluate(NamedMap({"price": 10}))
    with pytest.raises(ValueError, match="finite"):
        plan.evaluate(NamedMap({"price": float("nan"), "quantity": 2}))


def test_plan_handles_constants_and_rejects_unbound_or_aliased_inputs():
    p, q = Symbol("p"), Symbol("q")
    plan = ScalarPlan(NamedMap({}), NamedMap({"cash": 100}))
    assert plan.evaluate(NamedMap({})) == {"cash": 100}
    assert plan.over((date(2026, 1, 1),), NamedMap({}))["cash"].values == (100,)
    with pytest.raises(ValueError, match="independent symbols"):
        ScalarPlan(NamedMap({"p": p, "q": p}), NamedMap({"value": p}))
    with pytest.raises(ValueError, match="unspecified inputs"):
        ScalarPlan(NamedMap({"p": p}), NamedMap({"value": p + q}))

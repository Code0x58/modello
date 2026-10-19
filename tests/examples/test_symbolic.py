# mypy: warn-unused-ignores
"""Selected knowns and unknowns compile different views of one equation model."""

from datetime import date
from typing import ClassVar

import pytest
from sympy import Symbol

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.symbolic import SymbolicModel
from modello import InstanceDummy


class Affine(SymbolicModel):
    slope = InstanceDummy("slope", real=True)
    intercept = InstanceDummy("intercept", real=True)
    sample = InstanceDummy("sample", real=True)
    prediction = slope * sample + intercept

    inputs: ClassVar[tuple[str, ...]] = ("slope", "intercept", "sample")
    outputs: ClassVar[tuple[str, ...]] = ("prediction",)


def test_one_definition_supports_forward_inverse_and_coefficient_fitting():
    forward = Affine("prediction", slope=2, intercept=3, sample=4)
    inverse = Affine("input", slope=2, intercept=3, prediction=11)
    fitted = Affine("coefficient", intercept=3, sample=4, prediction=11)
    assert forward.prediction == 11
    assert inverse.sample == 4
    assert fitted.slope == 2
    assert Affine.plan() is Affine.plan()


def test_compilation_preserves_the_case_while_changing_runtime_unknowns():
    forward_case = Affine("forward", slope=2, intercept=3)
    observed = Symbol("observed", real=True)
    inverse_case = Affine("inverse", slope=2, intercept=3, prediction=observed)
    forward = forward_case.compile(inputs=("sample",))
    inverse = inverse_case.compile(inputs=("prediction",), outputs=("sample",))
    assert forward.evaluate(NamedMap({"sample": 4})) == {"prediction": 11}
    assert inverse.evaluate(NamedMap({"prediction": 11})) == {"sample": 4}
    assert forward_case.sample.free_symbols
    assert inverse_case.prediction == observed
    times = (date(2026, 1, 1), date(2026, 1, 2))
    assert inverse.over(times, NamedMap({"prediction": Series(times, (11, 15))}))["sample"].values == (4, 6)


def test_a_subclasss_equations_receive_their_own_compiled_view():
    class Shifted(Affine):
        prediction = Affine.slope * Affine.sample + Affine.intercept + 1

    values = NamedMap({"slope": 2, "intercept": 3, "sample": 4})
    assert Affine.plan().evaluate(values)["prediction"] == 11
    assert Shifted.plan().evaluate(values)["prediction"] == 12
    assert Shifted.plan() is not Affine.plan()


def test_a_view_cannot_silently_drop_unbound_parameters():
    with pytest.raises(ValueError, match="unspecified inputs"):
        Affine("missing").compile(inputs=("sample",))


def _typing_contract() -> None:
    values: NamedMap[float] = Affine.plan().evaluate(NamedMap({"slope": 2, "intercept": 3, "sample": 4}))
    assert values["prediction"] == 11
    _wrong_value: str = values["prediction"]  # type: ignore[assignment]
    Affine("case").compile(inputs=(1,))  # type: ignore[arg-type]

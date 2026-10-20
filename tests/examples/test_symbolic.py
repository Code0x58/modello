# mypy: warn-unused-ignores
"""Selected knowns and unknowns compile different views of one equation model."""

from datetime import date
from typing import TYPE_CHECKING

import pytest
from sympy import Symbol

from examples.compute.series import Series
from examples.model_plan import Known, ModelPlan
from examples.symbolic import SymbolicModel
from modello import InstanceDummy


class Affine(SymbolicModel):
    slope = InstanceDummy("slope", real=True)
    intercept = InstanceDummy("intercept", real=True)
    sample = InstanceDummy("sample", real=True)
    prediction = slope * sample + intercept


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
    forward = forward_case.compile()
    inverse = inverse_case.compile()
    assert float(forward.evaluate(Known(forward_case.sample, 4)).prediction) == 11
    assert float(inverse.evaluate(Known(inverse_case.prediction, 11)).sample) == 4
    assert forward_case.sample.free_symbols
    assert inverse_case.prediction == observed
    times = (date(2026, 1, 1), date(2026, 1, 2))
    path = inverse.over(times, Known(inverse_case.prediction, Series(times, (11, 15))))
    assert path.column(inverse_case.sample).values == (4, 6)


def test_a_subclasss_equations_receive_their_own_compiled_view():
    class Shifted(Affine):
        prediction = Affine.slope * Affine.sample + Affine.intercept + 1

    for model_type, expected in ((Affine, 11), (Shifted, 12)):
        view = model_type.plan()
        model = view.source
        result = view.evaluate(Known(model.slope, 2), Known(model.intercept, 3), Known(model.sample, 4))
        assert type(result) is model_type and float(result.prediction) == expected
    assert Shifted.plan() is not Affine.plan()


def test_a_view_cannot_silently_drop_unbound_parameters():
    with pytest.raises(ValueError, match="unspecified inputs"):
        source = Affine("missing")
        source.compile(inputs=(source.sample,))


if TYPE_CHECKING:

    def _typing_contract(source: Affine) -> None:
        view: ModelPlan[Affine] = source.compile()
        cached: ModelPlan[Affine] = Affine.plan()
        result: Affine = view.evaluate(Known(source.sample, 4))
        assert cached and result
        _wrong_value: str = view.evaluate(Known(source.sample, 4))  # type: ignore[assignment]
        source.compile(inputs=("sample",))  # type: ignore[arg-type]

"""Installed typing contracts for the opt-in research API."""

from typing import TYPE_CHECKING

from modello import Coordinate, InstanceDummy
from modello.contrib.research import Method, Requirement, ResearchCase, ResearchModel, increasing, requires


class TypedCurve(ResearchModel):
    x = Coordinate("x", real=True)
    a = InstanceDummy("a", real=True)
    value = a * x**3


if TYPE_CHECKING:
    from typing_extensions import assert_type

    case = ResearchCase.for_model(TypedCurve)
    assert_type(case, ResearchCase[TypedCurve])
    assert_type(case.symbols, TypedCurve)
    assert_type(case.observe(a=1), ResearchCase[TypedCurve])
    assert_type(case.assume(case.symbols.a > 0), ResearchCase[TypedCurve])
    assert_type(case.with_hint("bound", upper=1), ResearchCase[TypedCurve])
    assert_type(case.without_assumptions(reason="Keep the unchanged branch typed."), ResearchCase[TypedCurve])
    assert_type(case.without_requirements(reason="Keep the unchanged branch typed."), ResearchCase[TypedCurve])
    assert_type(increasing("value", wrt="x"), Requirement)
    assert_type(requires(TypedCurve.x, increasing("value", wrt="x")), Coordinate)
    assert_type(requires(TypedCurve, increasing("value", wrt="x")), type[TypedCurve])
    assert_type(case.run(Method("constant", lambda current: 1)).data, int)
    assert_type(Method("constant", lambda current: 1).bind(case), Method[int])

    def method_factory(current: ResearchCase[TypedCurve]) -> Method[int]:
        return Method("constant", lambda inner: 1).bind(current)

    assert_type(case.run(method_factory).data, int)

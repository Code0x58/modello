"""Installed typing contracts for the opt-in research API."""

from typing import TYPE_CHECKING

from modello import Coordinate, InstanceDummy
from modello.contrib.verification import (
    Method,
    Obligation,
    VerificationContext,
    VerificationModel,
    specifies,
    strictly_increasing,
)


class TypedCurve(VerificationModel):
    x = Coordinate("x", real=True)
    a = InstanceDummy("a", real=True)
    value = a * x**3


if TYPE_CHECKING:
    from typing_extensions import assert_type

    case = VerificationContext.for_model(TypedCurve)
    assert_type(case, VerificationContext[TypedCurve])
    assert_type(case.symbols, TypedCurve)
    assert_type(case.observe(a=1), VerificationContext[TypedCurve])
    assert_type(case.assume(case.symbols.a > 0), VerificationContext[TypedCurve])
    assert_type(case.with_bound_hint("bound", upper=1), VerificationContext[TypedCurve])
    assert_type(case.without_assumptions(reason="Keep the unchanged branch typed."), VerificationContext[TypedCurve])
    assert_type(case.without_obligations(reason="Keep the unchanged branch typed."), VerificationContext[TypedCurve])
    assert_type(strictly_increasing("value", wrt="x"), Obligation)
    assert_type(specifies(TypedCurve.x, strictly_increasing("value", wrt="x")), Coordinate)
    assert_type(specifies(TypedCurve, strictly_increasing("value", wrt="x")), type[TypedCurve])
    assert_type(case.run(Method("constant", lambda current: 1)).data, int)
    assert_type(Method("constant", lambda current: 1).bind(case), Method[int])

    def method_factory(current: VerificationContext[TypedCurve]) -> Method[int]:
        return Method("constant", lambda inner: 1).bind(current)

    assert_type(case.run(method_factory).data, int)

# mypy: warn-unused-ignores
"""Installed typing contracts for the opt-in verification API."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from modello import Coordinate, InstanceDummy, SolutionFamily, component
from modello.contrib.verification import (
    Evidence,
    InferenceRule,
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


class TypedStudy(VerificationModel):
    curve = component(TypedCurve)


@dataclass(frozen=True)
class TypedSpecification:
    sign: Obligation
    rule: InferenceRule | None = None


if TYPE_CHECKING:
    from typing import assert_type

    context = VerificationContext.for_model(TypedCurve)
    assert_type(context, VerificationContext[TypedCurve])
    assert_type(context.verify().evidence, tuple[Evidence, ...])
    assert_type(context.symbols, TypedCurve)
    assert_type(context.model, TypedCurve)
    assert_type(context.model.a, InstanceDummy)
    assert_type(context.specification, None)
    named = context.with_specification(
        lambda symbols: TypedSpecification(strictly_increasing(symbols.value, wrt=symbols.x))
    )
    assert_type(named, VerificationContext[TypedCurve, TypedSpecification])
    assert_type(named.specification, TypedSpecification)
    assert_type(named.specification.sign, Obligation)
    assert_type(named.observe(a=1), VerificationContext[TypedCurve, TypedSpecification])
    assert_type(named.assume(named.symbols.a > 0), VerificationContext[TypedCurve, TypedSpecification])
    assert_type(named.explain(named.specification.sign), Evidence)
    assert_type(named.verify().explain(named.specification.sign), Evidence)
    assert_type(named.obligations, tuple[Obligation, ...])
    assert_type(named.rules, tuple[InferenceRule, ...])
    named.specification.missing  # type: ignore[attr-defined]
    assert_type(named.with_specification(lambda symbols: None), VerificationContext[TypedCurve])
    assert_type(context.family, SolutionFamily[TypedCurve])
    assert_type(context.observe(a=1), VerificationContext[TypedCurve])
    assert_type(context.observe({context.symbols.a: 1}), VerificationContext[TypedCurve])
    scalar_inputs = {context.symbols.a: 1}
    assert_type(context.observe(scalar_inputs), VerificationContext[TypedCurve])
    study = VerificationContext.for_model(TypedStudy)
    assert_type(study.model.curve, TypedCurve)
    assert_type(study.symbols.curve, TypedCurve)
    assert_type(study.observe({study.symbols.curve: TypedCurve("curve", a=1)}), VerificationContext[TypedStudy])
    component_inputs = {study.symbols.curve: TypedCurve("curve", a=1)}
    assert_type(study.observe(component_inputs), VerificationContext[TypedStudy])
    assert_type(study.replace(study.symbols.curve, TypedCurve), VerificationContext[TypedStudy])
    study.model.missing  # type: ignore[attr-defined]
    study.symbols.curve.missing  # type: ignore[attr-defined]
    assert_type(context.assume(context.symbols.a > 0), VerificationContext[TypedCurve])
    assert_type(context.with_bound_hint("bound", upper=1), VerificationContext[TypedCurve])
    assert_type(context.without_assumptions(reason="Keep the unchanged branch typed."), VerificationContext[TypedCurve])
    assert_type(context.without_obligations(reason="Keep the unchanged branch typed."), VerificationContext[TypedCurve])
    assert_type(strictly_increasing("value", wrt="x"), Obligation)
    assert_type(specifies(TypedCurve.x, strictly_increasing("value", wrt="x")), Coordinate)
    assert_type(specifies(TypedCurve, strictly_increasing("value", wrt="x")), type[TypedCurve])
    assert_type(context.run(Method("constant", lambda current: 1)).data, int)
    assert_type(Method("constant", lambda current: 1).bind(context), Method[int])

    def method_factory(current: VerificationContext[TypedCurve]) -> Method[int]:
        return Method("constant", lambda inner: 1).bind(current)

    assert_type(context.run(method_factory).data, int)

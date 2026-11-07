"""Bind names from an ordinary frozen dataclass to proof declarations."""

from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import TypeVar, cast

from modello.errors import ModelDefinitionError

from ._records import InferenceRule, Obligation

SpecificationT = TypeVar("SpecificationT")
ProofDeclaration = Obligation | InferenceRule


def _specification_nodes(value: object, path: str = ""):
    if isinstance(value, (Obligation, InferenceRule)):
        yield path, value
    elif value is None:
        return
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _specification_nodes(item, f"{path}[{index}]")
    elif is_dataclass(value) and not isinstance(value, type) and getattr(type(value), "__dataclass_params__").frozen:
        for definition in fields(value):
            if not definition.init:
                raise ModelDefinitionError("Specification fields must be initialized dataclass fields")
            name = f"{path}.{definition.name}" if path else definition.name
            yield from _specification_nodes(getattr(value, definition.name), name)
    else:
        raise ModelDefinitionError(
            "A specification contains obligations, inference rules, tuples, and frozen dataclasses"
        )


def _named_specification(value: SpecificationT) -> tuple[SpecificationT, tuple[ProofDeclaration, ...]]:
    """Name each declaration once and keep shared inference-rule references intact."""
    if value is not None and (not is_dataclass(value) or isinstance(value, type)):
        raise ModelDefinitionError("Return a frozen dataclass as the specification, or None to remove it")
    nodes = tuple(_specification_nodes(value))
    named: dict[int, ProofDeclaration] = {}
    # Conclusions and premises can refer to obligations declared later in the
    # dataclass, so name all obligations before reconnecting the inference rules.
    for path, declaration in nodes:
        if isinstance(declaration, Obligation) and id(declaration) not in named:
            named[id(declaration)] = replace(declaration, name=path)
    for path, declaration in nodes:
        if isinstance(declaration, InferenceRule) and id(declaration) not in named:
            named[id(declaration)] = replace(
                declaration,
                name=path,
                conclusion=cast(Obligation, named.get(id(declaration.conclusion), declaration.conclusion)),
                premises=tuple(cast(Obligation, named.get(id(item), item)) for item in declaration.premises),
            )

    def rebuild(item):
        if isinstance(item, (Obligation, InferenceRule)):
            return named[id(item)]
        if item is None:
            return None
        if isinstance(item, tuple):
            values = tuple(rebuild(member) for member in item)
            return type(item)(*values) if hasattr(item, "_fields") else values
        return replace(
            item, **{definition.name: rebuild(getattr(item, definition.name)) for definition in fields(item)}
        )

    # A shared record has one canonical name, just as aliased model quantities
    # share one identity. The original dataclass and records remain unchanged.
    return cast(SpecificationT, rebuild(value)), tuple(named.values())

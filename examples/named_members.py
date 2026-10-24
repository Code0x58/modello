"""The research layer's adapter from Modello components to named numeric domains."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any, Callable, ClassVar, Generic, TypeVar, cast

from examples.data import NamedMap
from modello import Modello, ModelloMeta, ModelloMetaNamespace, ScalarValue, VectorValue, _ModelReference

_Model = TypeVar("_Model", bound=Modello)
_Binding = TypeVar("_Binding")
_Definition = TypeVar("_Definition", bound=Callable[..., type[Modello]])
_Value = TypeVar("_Value", covariant=True)
_Result = TypeVar("_Result")


def member_name(members: NamedMap[_Model], member: _Model) -> str:
    """Resolve a declared member by its actual model instance, preserving scope."""
    for name, candidate in members.items():
        if candidate is member:
            return name
    raise ValueError("Select a member of this model case")


@dataclass(frozen=True)
class MemberValues(Generic[_Model, _Value]):
    """A projection of declared model members, selected through typed references.

    A projection can omit members, for example a holding needing no order.
    Names are retained for traversal and export; selection uses the source case.
    """

    members: NamedMap[_Model]
    by_name: NamedMap[_Value]

    def __post_init__(self) -> None:
        if len({id(member) for member in self.members.values()}) != len(self.members):
            raise ValueError("Each declared member must have its own model instance")
        if self.by_name.keys() - self.members.keys():
            raise ValueError("Projected values must belong to declared members")

    def for_member(self, member: _Model) -> _Value:
        """Read a value for this case's member; an omitted member raises KeyError."""
        return self.by_name[member_name(self.members, member)]

    def map(self, project: Callable[[_Value], _Result]) -> MemberValues[_Model, _Result]:
        """Transform the values while retaining their declared member bindings."""
        return MemberValues(self.members, self.by_name.map(project))

    def __iter__(self) -> Iterator[_Value]:
        return iter(self.by_name.values())

    def __len__(self) -> int:
        return len(self.by_name)


def cached_definition(factory: _Definition) -> _Definition:
    """Reuse a model class while retaining its constructor's generic signature."""
    return cast(_Definition, cache(factory))


def _model_type(group: Modello | type[Modello]) -> type[Modello]:
    if isinstance(group, type):
        return group
    if isinstance(group, _ModelReference):
        return group._modello_model
    return type(group)


def component_members(group: Modello | type[Modello], member_type: type[_Model]) -> NamedMap[_Model]:
    """View a model's named children as members of one common model type.

    This module isolates the experiment's dependency on Modello metadata.
    It handles a model class, a class-body reference, or a populated instance.
    A group contains only components; scalar results belong on the parent.
    """
    model = _model_type(group)
    namespace = model._modello_namespace
    if namespace.dummies or namespace.vectors:
        raise TypeError("A member group must contain only model components")
    for name, child in namespace.nested_models.items():
        if not issubclass(child._modello_model, member_type):
            raise TypeError(f"{name} must be a {member_type.__name__} component")
    return NamedMap((name, cast(_Model, getattr(group, name))) for name in namespace.nested_models)


@dataclass(frozen=True)
class _BoundComponent(Generic[_Model, _Binding]):
    model: type[_Model]
    binding: _Binding

    def __set_name__(self, owner: type, name: str) -> None:
        raise TypeError(f"Declare {owner.__name__}.{name} in a BoundMembers model")


def bound_component(model: type[_Model], binding: _Binding) -> _Model:
    """Declare a typed model component and its external binding together."""
    return cast(_Model, _BoundComponent(model, binding))


@dataclass(frozen=True)
class ModelFactory(Generic[_Model, _Binding]):
    """Construct fresh models or declare their typed external bindings.

    Calling the factory uses the model's ordinary knowns and joint solver.
    ``component(binding)`` instead declares a child in a ``BoundMembers`` class.
    Neither operation loads the binding's external data.
    """

    model: type[_Model]
    binding_type: type[_Binding]

    def __call__(self, name: str, **knowns: ScalarValue | VectorValue | Modello) -> _Model:
        return self.model(name, **knowns)

    def component(self, binding: _Binding) -> _Model:
        """Couple a component declaration with a binding of the configured type."""
        if not isinstance(binding, self.binding_type):
            raise TypeError(f"{self.model.__name__} needs a {self.binding_type.__name__} binding")
        return bound_component(self.model, binding)


class _BindingNamespace(ModelloMetaNamespace):
    def __init__(self, name: str, bases: tuple[type, ...]) -> None:
        super().__init__(name, bases)
        self.bindings: dict[str, object] = {}
        for base in bases:
            # Follow the core's component inheritance precedence, including
            # clearing bindings when a later base supplies an unbound component.
            namespace = getattr(base, "_modello_namespace", None)
            if namespace is not None:
                for key in namespace.nested_models:
                    self.bindings.pop(key, None)
            self.bindings.update(getattr(base, "_member_bindings", {}))

    def __setitem__(self, key: str, value: object) -> None:
        if isinstance(value, _BoundComponent):
            self.bindings[key] = value.binding
            value = value.model
        else:
            self.bindings.pop(key, None)
        super().__setitem__(key, value)


class _BoundMembersMeta(ModelloMeta):
    @classmethod
    def __prepare__(metacls, __name: str, __bases: tuple[type, ...], **kwds: Any) -> _BindingNamespace:
        return _BindingNamespace(__name, __bases)

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: _BindingNamespace) -> type:
        namespace["_member_bindings"] = NamedMap(namespace.bindings)
        return super().__new__(mcs, name, bases, namespace)


class BoundMembers(Modello, Generic[_Binding], metaclass=_BoundMembersMeta):
    """A declared component domain with one binding per component, inherited together."""

    _member_bindings: ClassVar[NamedMap[object]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        component_members(cls, Modello).combine(cls._member_bindings, lambda model, binding: binding)


def member_bindings(group: BoundMembers[_Binding] | type[BoundMembers[_Binding]]) -> NamedMap[_Binding]:
    """Project bindings from a class, class-body reference, or solved member group."""
    model = _model_type(group)
    if not issubclass(model, BoundMembers):
        raise TypeError("Declare supplied components in a BoundMembers model")
    return cast(NamedMap[_Binding], model._member_bindings)

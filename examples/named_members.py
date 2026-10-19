"""The research layer's adapter from Modello components to named numeric domains."""

from typing import TypeVar, cast

from examples.compute.named_map import NamedMap
from modello import Modello, _ModelReference

_Model = TypeVar("_Model", bound=Modello)


def component_members(group: Modello | type[Modello], member_type: type[_Model]) -> NamedMap[_Model]:
    """View a model's named children as members of one common model type.

    This small adapter is the experiment's only dependency on Modello metadata.
    It handles a model class, a class-body reference, or a populated instance.
    A group contains only components; scalar results belong on the parent.
    """
    if isinstance(group, type):
        model = group
    elif isinstance(group, _ModelReference):
        model = group._modello_model
    else:
        model = type(group)
    namespace = model._modello_namespace
    if namespace.dummies:
        raise TypeError("A member group must contain only model components")
    for name, child in namespace.nested_models.items():
        if not issubclass(child._modello_model, member_type):
            raise TypeError(f"{name} must be a {member_type.__name__} component")
    return NamedMap((name, cast(_Model, getattr(group, name))) for name in namespace.nested_models)

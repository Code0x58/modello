"""Extension metadata collected alongside transparent model declarations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, TypeVar, cast

from modello.errors import ModelDefinitionError

if TYPE_CHECKING:
    from modello import Modello

__all__ = ["Annotation", "annotate", "annotate_model"]

_ValueT = TypeVar("_ValueT")
_ModelT = TypeVar("_ModelT", bound="Modello")
_AnnotationT = TypeVar("_AnnotationT", bound="Annotation")


class Annotation:
    """Base for extension metadata; it never participates in model equations.

    Subclasses usually declare immutable metadata with a frozen dataclass.
    ``for_model`` provides a typed view of the registry for this exact class.
    """

    @classmethod
    def for_model(cls: type[_AnnotationT], model: type[Modello]) -> Mapping[str, _AnnotationT]:
        """Read metadata by relative dotted path; ``''`` describes the model."""
        return cast(Mapping[str, _AnnotationT], model.__modello_annotations__.get(cls, MappingProxyType({})))


@dataclass(frozen=True)
class _Annotated:
    value: object
    annotations: tuple[Annotation, ...]


def annotate(value: _ValueT, *annotations: Annotation) -> _ValueT:
    """Attach metadata to a value assigned in a Modello class body.

    The class namespace unwraps the value before subsequent expressions use
    it. The value's static type is preserved, like ``component(Model)``.
    Stacked declarations retain different annotation types; the last instance
    of the same exact type wins. This is a declaration helper, not a wrapper
    for arithmetic outside a model class body.
    """
    _validate(annotations)
    if not annotations:
        return value
    if isinstance(value, _Annotated):
        return cast(_ValueT, _Annotated(value.value, (*value.annotations, *annotations)))
    return cast(_ValueT, _Annotated(value, annotations))


def annotate_model(model: type[_ModelT], *annotations: Annotation) -> type[_ModelT]:
    """Attach model-level metadata at the registry's empty path.

    Extension decorators can call this after a class has been created. The
    class itself is returned, preserving its identity and type.
    """
    from modello import Modello

    if not isinstance(model, type) or not issubclass(model, Modello):
        raise ModelDefinitionError("Model annotations require a Modello class")
    _validate(annotations)
    groups = {kind: dict(fields) for kind, fields in model.__modello_annotations__.items()}
    for annotation in annotations:
        groups.setdefault(type(annotation), {})[""] = annotation
    model.__modello_annotations__ = _freeze(groups)
    return model


def _validate(annotations: tuple[Annotation, ...]) -> None:
    if any(not isinstance(annotation, Annotation) for annotation in annotations):
        raise ModelDefinitionError("Extension metadata must derive from Annotation")


def _freeze(
    groups: Mapping[type[Annotation], Mapping[str, Annotation]],
) -> Mapping[type[Annotation], Mapping[str, Annotation]]:
    """Snapshot both mapping levels so subclasses cannot mutate their parents."""
    return MappingProxyType({kind: MappingProxyType(dict(fields)) for kind, fields in groups.items()})

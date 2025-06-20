#!/usr/bin/env python
"""Module for symbolic modeling of systems."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, ClassVar, cast

from sympy import Basic, Dummy, Eq, solve
# more verbose path as mypy sees sympy.simplify as a module
from sympy.simplify.simplify import simplify


class ModelloSentinelClass:
    """This class is used for quick type.mro() checks."""


class InstanceDummy(Dummy):
    """Dummy which will create a bound bummy on Modello instantiation."""

    def bound(self, model_name: str) -> "BoundInstanceDummy":
        """Return a dummy bound to a modello instance."""
        return BoundInstanceDummy(model_name + "_" + self.name, **self.assumptions0)


#    # for debugging
#    def _sympystr(self, printer):
#        return "%s[%s]" % (self.name, self.dummy_index)


class BoundInstanceDummy(InstanceDummy):
    """Dummy associated with a Modello instance."""


class ModelloMetaNamespace(dict):
    """Namespace used when building :class:`Modello` subclasses."""

    def __init__(self, name: str, bases: tuple[type, ...]) -> None:
        """Create a namespace for a Modello class to use."""
        super().__init__()
        self.name = name
        # map of attributes to sympy Basic (e.g expression, value) objects
        self.attrs: dict[str, Basic] = {}
        # map of attributes to InstanceDummy instances - metadata used by derived classes
        self.dummies: dict[str, InstanceDummy] = {}
        # map of attributes to non-modello managed objects
        self.other_attrs: dict[str, object] = {}
        # map of nested modello attributes to (model class, dummy mapping)
        self.nested_models: dict[str, tuple[type["Modello"], dict[InstanceDummy, InstanceDummy]]] = {}
        # map of dummies to dummies that override them - metadata used by derived classes
        self.dummy_overrides: dict[Dummy, Dummy] = {}

        # build up the attributes from the base classes
        for base in bases:
            if not issubclass(base, ModelloSentinelClass):
                continue
            parent_namespace = getattr(base, "_modello_namespace", None)
            # TODO: read the following (regarding python's method resolution order) and make sure all is ok:
            #  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.3910&rep=rep1&type=pdf

            if parent_namespace:
                # find which dummies are overridden by this base modello
                for attr in self.dummies.keys() & parent_namespace.dummies.keys():
                    override_dummy = parent_namespace.dummies[attr]
                    base_dummy = self.dummies[attr]
                    self.dummies[attr] = override_dummy
                    self.dummy_overrides[base_dummy] = override_dummy

                self.attrs.update(parent_namespace.attrs)
                self.dummies.update(parent_namespace.dummies)
                self.other_attrs.update(parent_namespace.other_attrs)
                self.nested_models.update(parent_namespace.nested_models)
                self.update(parent_namespace)
            # substitute overridden dummies in the attributes
            if self.dummy_overrides:
                for attr, value in self.attrs.items():
                    self.attrs[attr] = value.subs(self.dummy_overrides)

    def __setitem__(self, key: str, value: object) -> None:
        """Manage modello attributes as values are assigned."""
        if isinstance(value, Basic):
            if key in self:
                dummy = self.dummies[key]
            elif isinstance(value, InstanceDummy):
                dummy = value
            else:
                dummy = InstanceDummy(key, **value.assumptions0)
            self.attrs[key] = simplify(value).subs(self.dummy_overrides)
            self.dummies[key] = dummy
            value = dummy
        elif key in self.attrs:
            # cannot overried a part of inherited expressions with a non-expression
            raise ValueError(
                "Cannot assign %s.%s to a non-expression" % (self.name, key)
            )
        else:
            if isinstance(value, type) and issubclass(value, ModelloSentinelClass):
                model_cls = cast(type["Modello"], value)
                dummy_map: dict[InstanceDummy, InstanceDummy] = {}
                proxy = type(f"{model_cls.__name__}Proxy", (), {})()
                for attr_name, class_dummy in model_cls._modello_namespace.dummies.items():
                    proxy_dummy = InstanceDummy(f"{key}_{class_dummy.name}", **class_dummy.assumptions0)
                    setattr(proxy, attr_name, proxy_dummy)
                    dummy_map[class_dummy] = proxy_dummy
                # include proxies for child models so expressions can be substituted
                for _nested_attr, (_, child_map) in model_cls._modello_nested_models.items():
                    for child_dummy, child_proxy in child_map.items():
                        proxy_dummy = InstanceDummy(
                            f"{key}_{child_proxy.name}", **child_proxy.assumptions0
                        )
                        dummy_map[child_proxy] = proxy_dummy
                for attr_name, expr in model_cls._modello_namespace.attrs.items():
                    if attr_name not in model_cls._modello_namespace.dummies:
                        setattr(proxy, attr_name, expr.subs(dummy_map))
                self.nested_models[key] = (model_cls, dummy_map)
                value = proxy
            self.other_attrs[key] = value
        super().__setitem__(key, value)


class ModelloMeta(type):
    """Used to make Modello class definitions use dummies."""

    @classmethod
    def __prepare__(
        metacls, __name: str, __bases: tuple[type, ...], **kwds: Any
    ) -> MutableMapping[str, Any]:
        """Return a ModelloMetaNamespace instead of a plain dict to accumlate attributes on."""
        return ModelloMetaNamespace(__name, __bases)

    def __new__(
        mcs,
        name: str,
        bases: tuple[type, ...],
        meta_namespace: ModelloMetaNamespace,
    ) -> Any:
        """Return a new class with modello attributes."""
        namespace = dict(meta_namespace)
        # could follow django's model of _meta? conflicts?
        namespace["_modello_namespace"] = meta_namespace
        namespace["_modello_class_constraints"] = {
            dummy: meta_namespace.attrs[attr]
            for attr, dummy in meta_namespace.dummies.items()
            if meta_namespace.attrs[attr] is not dummy
        }
        namespace["_modello_nested_models"] = meta_namespace.nested_models
        return super().__new__(mcs, name, bases, namespace)


class Modello(ModelloSentinelClass, metaclass=ModelloMeta):
    """Base class for building symbolic models."""

    _modello_namespace: ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace(
        "", ()
    )
    _modello_class_constraints: dict[InstanceDummy, Basic] = {}
    _modello_nested_models: dict[str, tuple[type["Modello"], dict[InstanceDummy, InstanceDummy]]] = {}

    # ------------------------------------------------------------------
    # Private helpers used by ``__init__``
    # ------------------------------------------------------------------
    @classmethod
    def _parse_nested_values(
        cls, value_map: Mapping[str, Basic]
    ) -> tuple[dict[str, Modello | dict[str, Basic]], dict[str, Basic]]:
        """Split ``value_map`` into nested and local values.

        Attributes matching ``cls._modello_nested_models`` are returned in the
        ``nested`` dictionary. Unknown or ``None`` values produce an empty mapping
        entry. The returned ``values`` mapping contains only attributes local to
        this model.
        """

        nested: dict[str, Modello | dict[str, Basic]] = {}
        remaining: dict[str, Basic] = {}
        for key, val in value_map.items():
            if key not in cls._modello_nested_models:
                remaining[key] = val
                continue

            match val:
                case Modello() | dict():
                    nested[key] = val
                case _:
                    nested[key] = {}
        return nested, remaining

    @classmethod
    def _create_instance_dummies(
        cls, name: str
    ) -> tuple[
        dict[InstanceDummy, BoundInstanceDummy],
        dict[str, dict[InstanceDummy, BoundInstanceDummy]],
    ]:
        """Bind all ``InstanceDummy`` objects to this instance.

        Returns a tuple containing a mapping of this model's dummies to their
        bound counterparts and a mapping for each nested model that relates the
        child's class dummies to their bound dummies.
        """

        instance_dummies: dict[InstanceDummy, BoundInstanceDummy] = {
            d: d.bound(name) for d in cls._modello_namespace.dummies.values()
        }

        nested_dummy_map: dict[str, dict[InstanceDummy, BoundInstanceDummy]] = {}
        for attr, (_, mapping) in cls._modello_nested_models.items():
            dummy_map: dict[InstanceDummy, BoundInstanceDummy] = {}
            for class_dummy, proxy_dummy in mapping.items():
                bound = proxy_dummy.bound(name)
                instance_dummies[proxy_dummy] = bound
                dummy_map[class_dummy] = bound
            nested_dummy_map[attr] = dummy_map

        return instance_dummies, nested_dummy_map

    @classmethod
    def _collect_instance_constraints(
        cls,
        value_map: Mapping[str, Basic],
        nested_values: Mapping[str, Modello | dict[str, Basic]],
        instance_dummies: dict[InstanceDummy, BoundInstanceDummy],
    ) -> dict[BoundInstanceDummy, Basic]:
        """Convert provided values into instance constraints."""

        constraints: dict[BoundInstanceDummy, Basic] = {}

        for attr, value in value_map.items():
            simplified = simplify(value).subs(instance_dummies)
            class_dummy = getattr(cls, attr)
            constraints[instance_dummies[class_dummy]] = simplified

        for attr, data in nested_values.items():
            model_cls, mapping = cls._modello_nested_models[attr]
            values = (
                {k: getattr(data, k) for k in model_cls._modello_namespace.dummies.keys()}
                if isinstance(data, Modello)
                else data
            )
            for key, val in values.items():
                class_dummy = model_cls._modello_namespace.dummies[key]
                proxy_dummy = mapping[class_dummy]
                constraints[instance_dummies[proxy_dummy]] = simplify(val).subs(instance_dummies)

        return constraints

    @classmethod
    def _build_constraints(
        cls,
        instance_dummies: dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: dict[BoundInstanceDummy, Basic],
    ) -> list[Eq]:
        """Compile a list of equations representing this instance."""

        constraints = [
            Eq(instance_dummies[class_dummy], value.subs(instance_dummies))
            for class_dummy, value in cls._modello_class_constraints.items()
        ]
        for attr, (model_cls, mapping) in cls._modello_nested_models.items():
            for class_dummy, expr in model_cls._modello_class_constraints.items():
                proxy_dummy = mapping[class_dummy]
                constraints.append(Eq(instance_dummies[proxy_dummy], expr.subs(mapping).subs(instance_dummies)))

        constraints.extend(Eq(d, v) for d, v in instance_constraints.items())
        return constraints

    @staticmethod
    def _solve(constraints: list[Eq]) -> dict[BoundInstanceDummy, Basic]:
        """Solve ``constraints`` and return a mapping of dummies to values."""

        if not constraints:
            return {}

        solutions = solve(constraints, particular=True, dict=True)
        if len(solutions) != 1:
            raise ValueError(f"{len(solutions)} solutions")
        return solutions[0]

    @staticmethod
    def _resolve_bound_value(
        class_dummy: InstanceDummy,
        bound_dummy: BoundInstanceDummy,
        *,
        solution: dict[BoundInstanceDummy, Basic],
        instance_constraints: dict[BoundInstanceDummy, Basic],
        class_constraints: dict[InstanceDummy, Basic],
        subs_map: Mapping[InstanceDummy, BoundInstanceDummy],
    ) -> Basic:
        """Return the resolved value for ``bound_dummy``."""

        if bound_dummy in solution:
            return solution[bound_dummy]
        if bound_dummy in instance_constraints:
            return instance_constraints[bound_dummy]
        if class_dummy in class_constraints:
            return class_constraints[class_dummy].subs(subs_map)
        return bound_dummy

    @classmethod
    def _assign_local_values(
        cls,
        solution: dict[BoundInstanceDummy, Basic],
        instance_dummies: dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: dict[BoundInstanceDummy, Basic],
    ) -> dict[str, Basic]:
        """Return resolved values for this model's own attributes."""

        values: dict[str, Basic] = {}
        for attr, class_dummy in cls._modello_namespace.dummies.items():
            bound = instance_dummies[class_dummy]
            values[attr] = cls._resolve_bound_value(
                class_dummy,
                bound,
                solution=solution,
                instance_constraints=instance_constraints,
                class_constraints=cls._modello_class_constraints,
                subs_map=instance_dummies,
            )
        return values

    @classmethod
    def _instantiate_nested_models(
        cls,
        name: str,
        solution: dict[BoundInstanceDummy, Basic],
        instance_dummies: dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: dict[BoundInstanceDummy, Basic],
    ) -> dict[str, "Modello"]:
        """Instantiate nested models and return them."""

        nested_instances: dict[str, Modello] = {}
        for attr, (model_cls, mapping) in cls._modello_nested_models.items():
            value_kwargs: dict[str, Basic] = {}
            for child_attr, class_dummy in model_cls._modello_namespace.dummies.items():
                proxy_dummy = mapping[class_dummy]
                bound = instance_dummies[proxy_dummy]
                value_kwargs[child_attr] = cls._resolve_bound_value(
                    class_dummy,
                    bound,
                    solution=solution,
                    instance_constraints=instance_constraints,
                    class_constraints=model_cls._modello_class_constraints,
                    subs_map={**mapping, **instance_dummies},
                )
            nested_instances[attr] = model_cls(f"{name}_{attr}", **value_kwargs)
        return nested_instances

    # ------------------------------------------------------------------
    # ``__init__`` orchestrates the above helpers
    # ------------------------------------------------------------------
    def __init__(self, name: str, **value_map: Basic) -> None:
        """Initialise a model instance and solve for all attributes."""

        nested_values, local_values = self._parse_nested_values(value_map)
        instance_dummies, nested_dummy_map = self._create_instance_dummies(name)
        self._modello_instance_dummies = instance_dummies

        instance_constraints = self._collect_instance_constraints(
            local_values, nested_values, instance_dummies
        )
        self._modello_instance_constraints = instance_constraints

        constraints = self._build_constraints(instance_dummies, instance_constraints)
        self._modello_constraints = constraints

        solution = self._solve(constraints)

        values = self._assign_local_values(solution, instance_dummies, instance_constraints)
        for attr, val in values.items():
            setattr(self, attr, val)

        nested_instances = self._instantiate_nested_models(
            name, solution, instance_dummies, instance_constraints
        )
        for attr, instance in nested_instances.items():
            setattr(self, attr, instance)

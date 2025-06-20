#!/usr/bin/env python
"""Module for symbolic modeling of systems."""
import typing

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
    """This is so that Modello class definitions implicitly define symbols."""

    def __init__(self, name: str, bases: typing.Tuple[type, ...]) -> None:
        """Create a namespace for a Modello class to use."""
        super().__init__()
        self.name = name
        # map of attributes to sympy Basic (e.g expression, value) objects
        self.attrs: typing.Dict[str, Basic] = {}
        # map of attributes to InstanceDummy instances - metadata used by derived classes
        self.dummies: typing.Dict[str, InstanceDummy] = {}
        # map of attributes to non-modello managed objects
        self.other_attrs: typing.Dict[str, object] = {}
        # map of nested modello attributes to (model class, dummy mapping)
        self.nested_models: typing.Dict[str, typing.Tuple[typing.Type["Modello"], typing.Dict[InstanceDummy, InstanceDummy]]] = {}
        # map of dummies to dummies that override them - metadata used by derived classes
        self.dummy_overrides: typing.Dict[Dummy, Dummy] = {}

        # build up the attributes from the base classes
        for base in bases:
            if ModelloSentinelClass not in base.mro():
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
            if isinstance(value, type) and ModelloSentinelClass in value.mro():
                model_cls: typing.Type["Modello"] = value
                dummy_map: typing.Dict[InstanceDummy, InstanceDummy] = {}
                proxy = type(f"{model_cls.__name__}Proxy", (), {})()
                for attr_name, class_dummy in model_cls._modello_namespace.dummies.items():
                    proxy_dummy = InstanceDummy(f"{key}_{class_dummy.name}", **class_dummy.assumptions0)
                    setattr(proxy, attr_name, proxy_dummy)
                    dummy_map[class_dummy] = proxy_dummy
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
        metacls, __name: str, __bases: typing.Tuple[type, ...], **kwds: typing.Any
    ) -> typing.MutableMapping[str, typing.Any]:
        """Return a ModelloMetaNamespace instead of a plain dict to accumlate attributes on."""
        return ModelloMetaNamespace(__name, __bases)

    def __new__(
        mcs,
        name: str,
        bases: typing.Tuple[type, ...],
        meta_namespace: ModelloMetaNamespace,
    ) -> typing.Any:
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

    _modello_namespace: typing.ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace(
        "", ()
    )
    _modello_class_constraints: typing.Dict[InstanceDummy, Basic] = {}
    _modello_nested_models: typing.Dict[str, typing.Tuple[typing.Type["Modello"], typing.Dict[InstanceDummy, InstanceDummy]]] = {}

    # ------------------------------------------------------------------
    # Private helpers used by ``__init__``
    # ------------------------------------------------------------------
    def _parse_nested_values(self, value_map: typing.Dict[str, Basic]) -> typing.Dict[str, typing.Union["Modello", typing.Dict[str, Basic]]]:
        """Extract nested model data from ``value_map``.

        Unknown or ``None`` values are replaced with an empty mapping.
        The extracted values are removed from ``value_map``.
        """

        nested_values: typing.Dict[str, typing.Union["Modello", typing.Dict[str, Basic]]] = {}
        for attr in self._modello_nested_models:
            raw = value_map.pop(attr, {})
            if isinstance(raw, Modello):
                nested_values[attr] = raw
            elif isinstance(raw, dict):
                nested_values[attr] = raw
            else:
                nested_values[attr] = {}
        return nested_values

    def _create_instance_dummies(
        self, name: str
    ) -> tuple[
        typing.Dict[InstanceDummy, BoundInstanceDummy],
        typing.Dict[str, typing.Dict[InstanceDummy, BoundInstanceDummy]],
    ]:
        """Bind all ``InstanceDummy`` objects to this instance.

        Returns a tuple of ``instance_dummies`` for this model and a mapping for
        each nested model that relates the child's class dummies to their bound
        counterparts.
        """

        instance_dummies: dict[InstanceDummy, BoundInstanceDummy] = {
            class_dummy: class_dummy.bound(name)
            for class_dummy in self._modello_namespace.dummies.values()
        }

        nested_dummy_map: dict[str, dict[InstanceDummy, BoundInstanceDummy]] = {}
        for attr, (_, mapping) in self._modello_nested_models.items():
            dummy_map: dict[InstanceDummy, BoundInstanceDummy] = {}
            for class_dummy, proxy_dummy in mapping.items():
                bound = proxy_dummy.bound(name)
                instance_dummies[proxy_dummy] = bound
                dummy_map[class_dummy] = bound
            nested_dummy_map[attr] = dummy_map

        return instance_dummies, nested_dummy_map

    def _collect_instance_constraints(
        self,
        value_map: typing.Dict[str, Basic],
        nested_values: typing.Dict[str, typing.Union["Modello", typing.Dict[str, Basic]]],
        instance_dummies: typing.Dict[InstanceDummy, BoundInstanceDummy],
        nested_dummy_map: typing.Dict[str, typing.Dict[InstanceDummy, BoundInstanceDummy]],
    ) -> typing.Dict[BoundInstanceDummy, Basic]:
        """Convert provided values into instance constraints."""

        constraints: dict[BoundInstanceDummy, Basic] = {}

        for attr, value in value_map.items():
            simplified = simplify(value).subs(instance_dummies)
            value_map[attr] = simplified
            class_dummy = getattr(self, attr)
            constraints[instance_dummies[class_dummy]] = simplified

        for attr, data in nested_values.items():
            model_cls, mapping = self._modello_nested_models[attr]
            if isinstance(data, Modello):
                for child_attr, class_dummy in model_cls._modello_namespace.dummies.items():
                    proxy_dummy = mapping[class_dummy]
                    val = getattr(data, child_attr)
                    constraints[instance_dummies[proxy_dummy]] = simplify(val).subs(instance_dummies)
            else:
                for key, val in data.items():
                    class_dummy = model_cls._modello_namespace.dummies[key]
                    proxy_dummy = mapping[class_dummy]
                    constraints[instance_dummies[proxy_dummy]] = simplify(val).subs(instance_dummies)

        return constraints

    def _build_constraints(
        self,
        instance_dummies: typing.Dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: typing.Dict[BoundInstanceDummy, Basic],
        nested_dummy_map: typing.Dict[str, typing.Dict[InstanceDummy, BoundInstanceDummy]],
    ) -> list[Eq]:
        """Compile a list of equations representing this instance."""

        constraints = [
            Eq(instance_dummies[class_dummy], value.subs(instance_dummies))
            for class_dummy, value in self._modello_class_constraints.items()
        ]

        for attr, (model_cls, mapping) in self._modello_nested_models.items():
            for class_dummy, expr in model_cls._modello_class_constraints.items():
                proxy_dummy = mapping[class_dummy]
                constraints.append(Eq(instance_dummies[proxy_dummy], expr.subs(mapping).subs(instance_dummies)))

        constraints.extend(Eq(d, v) for d, v in instance_constraints.items())
        return constraints

    def _solve(self, constraints: list[Eq]) -> dict[BoundInstanceDummy, Basic]:
        """Solve ``constraints`` and return a mapping of dummies to values."""

        if not constraints:
            return {}

        solutions = solve(constraints, particular=True, dict=True)
        if len(solutions) != 1:
            raise ValueError(f"{len(solutions)} solutions")
        return solutions[0]

    def _assign_local_values(
        self,
        solution: dict[BoundInstanceDummy, Basic],
        instance_dummies: typing.Dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: typing.Dict[BoundInstanceDummy, Basic],
    ) -> None:
        """Set resolved values on this instance's own attributes."""

        for attr, class_dummy in self._modello_namespace.dummies.items():
            instance_dummy = instance_dummies[class_dummy]
            if instance_dummy in solution:
                value = solution[instance_dummy]
            elif instance_dummy in instance_constraints:
                value = instance_constraints[instance_dummy]
            elif class_dummy in self._modello_class_constraints:
                value = self._modello_class_constraints[class_dummy].subs(instance_dummies)
            else:
                value = instance_dummy
            setattr(self, attr, value)

    def _instantiate_nested_models(
        self,
        name: str,
        solution: dict[BoundInstanceDummy, Basic],
        instance_dummies: typing.Dict[InstanceDummy, BoundInstanceDummy],
        instance_constraints: typing.Dict[BoundInstanceDummy, Basic],
        nested_dummy_map: typing.Dict[str, typing.Dict[InstanceDummy, BoundInstanceDummy]],
    ) -> None:
        """Create nested model instances using solved values."""

        for attr, (model_cls, mapping) in self._modello_nested_models.items():
            value_kwargs: dict[str, Basic] = {}
            for child_attr, class_dummy in model_cls._modello_namespace.dummies.items():
                proxy_dummy = mapping[class_dummy]
                inst_dummy = instance_dummies[proxy_dummy]
                if inst_dummy in solution:
                    val = solution[inst_dummy]
                elif inst_dummy in instance_constraints:
                    val = instance_constraints[inst_dummy]
                elif class_dummy in model_cls._modello_class_constraints:
                    val = model_cls._modello_class_constraints[class_dummy].subs(mapping).subs(instance_dummies)
                else:
                    val = inst_dummy
                value_kwargs[child_attr] = val
            nested_instance = model_cls(f"{name}_{attr}", **value_kwargs)
            setattr(self, attr, nested_instance)

    # ------------------------------------------------------------------
    # ``__init__`` orchestrates the above helpers
    # ------------------------------------------------------------------
    def __init__(self, name: str, **value_map: Basic) -> None:
        """Initialise a model instance and solve for all attributes."""

        nested_values = self._parse_nested_values(value_map)
        instance_dummies, nested_dummy_map = self._create_instance_dummies(name)
        self._modello_instance_dummies = instance_dummies

        instance_constraints = self._collect_instance_constraints(
            value_map, nested_values, instance_dummies, nested_dummy_map
        )
        self._modello_instance_constraints = instance_constraints

        constraints = self._build_constraints(instance_dummies, instance_constraints, nested_dummy_map)
        self._modello_constraints = constraints

        solution = self._solve(constraints)

        self._assign_local_values(solution, instance_dummies, instance_constraints)
        self._instantiate_nested_models(
            name, solution, instance_dummies, instance_constraints, nested_dummy_map
        )

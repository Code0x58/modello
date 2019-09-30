#!/usr/bin/env python
"""Module for symbolic modeling of systems."""
import typing

from sympy import Basic, Dummy, Eq, simplify, solve


class ModelloSentinelClass:
    """This class is used for quick type.mro() checks."""


class InstanceDummy(Dummy):
    """Dummy which will create a bound bummy on Modello instantiation."""

    def bound(self, model_name: str) -> "BoundInstanceDummy":
        """Return an dummy bound to a modello instance."""
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
        self.name = name
        # map of attributes to sympy Basic (e.g expression, value) objects
        self.attrs: typing.Dict[str, Basic] = {}
        # map of attributes to InstanceDummy instances - metadata used by derived classes
        self.dummies: typing.Dict[str, Dummy] = {}
        # map of attributes to non-modello managed objects
        self.other_attrs: typing.Dict[str, object] = {}
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
            self.other_attrs[key] = value
        super().__setitem__(key, value)


class ModelloMeta(type):
    """Used to make Modello class definitions use dummies."""

    @classmethod
    def __prepare__(
        metacls, __name: str, __bases: typing.Tuple[type, ...], **kwds: typing.Any
    ) -> typing.Mapping[str, typing.Any]:
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
        return super().__new__(mcs, name, bases, namespace)


class Modello(ModelloSentinelClass, metaclass=ModelloMeta):
    """Base class for building symbolic models."""

    _modello_namespace: typing.ClassVar[ModelloMetaNamespace] = ModelloMetaNamespace(
        "", ()
    )
    _modello_class_constraints: typing.Dict[InstanceDummy, Basic] = {}

    def __init__(self, name: str, **value_map: typing.Dict[str, Basic]) -> None:
        """Initialise a model instance and solve for each attribute."""
        instance_dummies = {
            class_dummy: class_dummy.bound(name)
            for class_dummy in self._modello_namespace.dummies.values()
        }
        self._modello_instance_dummies = instance_dummies

        instance_constraints = {}
        for attr, value in value_map.items():
            value = simplify(value).subs(instance_dummies)
            value_map[attr] = value
            class_dummy = getattr(self, attr)
            instance_dummy = instance_dummies[class_dummy]
            instance_constraints[instance_dummy] = value
        self._modello_instance_constraints: typing.Dict[
            BoundInstanceDummy, Basic
        ] = instance_constraints

        constraints = [
            Eq(instance_dummies[class_dummy], value.subs(instance_dummies))
            for class_dummy, value in self._modello_class_constraints.items()
        ]
        constraints.extend(
            Eq(instance_dummy, value)
            for instance_dummy, value in instance_constraints.items()
        )
        # handy for debugging
        self._modello_constraints: typing.List[Eq] = constraints

        if constraints:
            solutions = solve(constraints, particular=True, dict=True)
            if len(solutions) != 1:
                raise ValueError("%s solutions" % len(solutions))
            solution = solutions[0]
        else:
            solution = {}

        for attr, class_dummy in self._modello_namespace.dummies.items():
            instance_dummy = instance_dummies[class_dummy]
            if instance_dummy in solution:
                value = solution[instance_dummy]
            elif instance_dummy in instance_constraints:
                value = instance_constraints[instance_dummy]
            elif class_dummy in self._modello_class_constraints:
                value = self._modello_class_constraints[class_dummy].subs(
                    instance_dummies
                )
            else:
                value = instance_dummy
            setattr(self, attr, value)

Exploring named-member solving
===============================

The general Modello feature under investigation is solving relationships over
a declared domain of named, typed members. A model should be able to project
member expressions, combine values by name, and reduce them to an aggregate
while retaining the joint solver's forward and inverse behavior.

The :doc:`portfolio example </examples/portfolio/index>` is one application of
that idea. Account replay, signals, market feeds, performance statistics, and
reports are application layers. Numeric series and compiled expression plans
remain experimental helpers in ``examples/compute/`` and may later become a
separate package.

What the prototype demonstrates
--------------------------------

``examples/named_members.py`` adapts existing typed components into a
``NamedMap``. A reduction expands into ordinary scalar expressions during model
definition, so it can already participate in inverse solving outside finance:

>>> from modello import Modello, InstanceDummy, component
>>> from examples.named_members import component_members
>>> class Reading(Modello):
...     value = InstanceDummy("value", positive=True)
>>> class Readings(Modello):
...     inlet = component(Reading)
...     outlet = component(Reading)
>>> class Total(Modello):
...     readings = component(Readings)
...     total = sum(member.value for member in component_members(readings, Reading).values())
>>> known = Readings("measured", inlet=Reading("inlet", value=3))
>>> result = Total("combined", readings=known, total=10)
>>> result.readings.outlet.value
7
>>> dict(component_members(result.readings, Reading).map(lambda member: member.value))
{'inlet': 3, 'outlet': 7}

Membership comes from the component declaration. Projection retains names;
combination checks that both maps have exactly the same names and aligns by
name rather than iteration position. Adding a declaration creates a new model
definition. Numeric feeds bind to that domain afterward.

Application views can retain the same typed member references. ``MemberValues``
associates a projection with its source case, so a consumer can select an actual
member and map the values without reconstructing string keys:

>>> from examples.named_members import MemberValues
>>> members = component_members(result.readings, Reading)
>>> values = MemberValues(members, members.map(lambda reading: float(reading.value)))
>>> values.for_member(result.readings.outlet)
7.0
>>> values.map(lambda value: value * 2).for_member(result.readings.inlet)
6.0

Selection checks case ownership. A projection can omit members while keeping
the full source domain, as with holdings that need no order. This association
lives in the example adapter; the core continues to solve scalar equations.

Questions before a public API
------------------------------

The prototype uses existing component fields and a small adapter over private
model metadata. It has no public named-map field or map-valued solver input.
A general feature would need to define typed member declarations, constraints
on projected members, and how names participate in errors and introspection.
Static typing also needs to distinguish a common member type from knowledge of
particular keys and their concrete subtypes.

Those semantics belong with model construction and solving. Observation axes,
missing-data policies, execution backends, and financial conventions can remain
separate consumers of the resulting expressions.

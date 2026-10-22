"""Small declarations shared by snapshot, diagnosis, and solution tests."""

from modello import InstanceDummy, Modello, component


class Reading(Modello):
    x = InstanceDummy("x", real=True)
    squared = x**2


class Check(Modello):
    reading = component(Reading)
    total = reading.squared + 1


class Sum(Modello):
    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)
    total = x + y


class PositiveReading(Reading):
    x = InstanceDummy("x", positive=True)
    alias = x


class Envelope(Modello):
    check = component(Check)


class PositiveScalar(Modello):
    x = InstanceDummy("x", positive=True)
    alias = x

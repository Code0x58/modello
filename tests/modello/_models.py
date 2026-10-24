"""Small declarations shared by snapshot, diagnosis, and solution tests."""

from sympy import sin

from modello import Index, InstanceDummy, Modello, Recurrence, Series, Vector, component


class Scalar(Modello):
    x = InstanceDummy("x")


class ScalarComponent(Modello):
    child = component(Scalar)


class Increment(Modello):
    x = InstanceDummy("x")
    y = x + 1


class Pair(Modello):
    a = InstanceDummy("a")
    b = InstanceDummy("b")
    total = a + b


class PairParent(Modello):
    child = component(Pair)
    total = child.total + 1


class Operators(Modello):
    x = InstanceDummy("x", commutative=False)
    y = InstanceDummy("y", commutative=False)
    commutator = x * y - y * x


class Wave(Modello):
    angle = InstanceDummy("angle", real=True)
    sine = sin(angle)


class Circle(Modello):
    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)
    squared_radius = x**2 + y**2


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


class DrivenSequence(Modello):
    N = InstanceDummy("N", integer=True, nonnegative=True)
    opening = InstanceDummy("opening", real=True)
    retention = InstanceDummy("retention", real=True)
    innovations = Vector("epsilon", size=N, real=True)
    n = Index("n", nonnegative=True)
    T = Series("T", n, real=True)
    law = Recurrence(T[n], retention * T[n - 1] + innovations[n - 1], initial={0: opening})
    levels = law.window(0, N)
    closing = T[N]

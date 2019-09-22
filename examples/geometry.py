"""Modello examples based on geometry."""
from modello import InstanceDummy, Modello
from sympy import sqrt


class RightAngleTriangle(Modello):
    """
    Use Pythagorus' theorm to model a right angled triangle.

    >>> T = RightAngleTriangle("T", a=3, b=4)
    >>> T.c
    5
    >>> T = RightAngleTriangle("T", b=4, c=5)
    >>> T.a
    3
    """

    a = InstanceDummy("a", real=True, positive=True)
    b = InstanceDummy("b", real=True, positive=True)
    c = sqrt(a ** 2 + b ** 2)

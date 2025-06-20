"""Example showing nested models.

>>> box = Box("B", base={"a": 3, "b": 4}, height=12)
>>> box.base.c
5
>>> box.diagonal
13
"""
from modello import InstanceDummy, Modello
from examples.geometry import RightAngleTriangle
from sympy import sqrt


class Box(Modello):
    """Simple box using a RightAngleTriangle as the base."""

    base = RightAngleTriangle
    height = InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c ** 2 + height ** 2)


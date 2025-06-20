"""Example demonstrating nested models with a Line composed of Points.

>>> line = Line("L", start={"x": 0, "y": 0}, end={"x": 3, "y": 4})
>>> line.length
5
"""
from modello import InstanceDummy, Modello
from sympy import sqrt


class Point(Modello):
    """Simple 2D point."""

    x = InstanceDummy("x", real=True)
    y = InstanceDummy("y", real=True)


class Line(Modello):
    """Line consisting of two points."""

    start = Point
    end = Point
    length = sqrt((end.x - start.x) ** 2 + (end.y - start.y) ** 2)

"""Small annotated declarations and snapshot helpers shared by GUI tests."""

from xml.etree import ElementTree

from sympy import Product

from modello import Index, InstanceDummy, Modello, Vector, component, vector
from modello.gui import gui


class Panel(Modello):
    """A rectangular panel."""

    width = gui(InstanceDummy("width", positive=True), description="Width in metres.", symbol="w")
    height = InstanceDummy("height", positive=True)
    area = width * height


class Estimate(Modello):
    panel = component(Panel)
    price = gui(InstanceDummy("price", positive=True), symbol="p", hidden=True)
    total = panel.area * price


def rows(data, key="fields"):
    return {row["path"]: row for row in data[key]}


NS = {"m": "http://www.w3.org/1998/Math/MathML"}


class Window(Modello):
    samples = gui(Vector("x", size=3, real=True), title="Observations", symbol="x", unit="m")
    i = Index("i")
    total = samples.sum(i)
    growth = Product(1 + samples[i], (i, 0, 2))
    changes = gui(vector(samples[i] - samples[i - 1], (i, 1, 2)), symbol="Δx")


def mathml_text(markup):
    return "".join(ElementTree.fromstring(markup).itertext())

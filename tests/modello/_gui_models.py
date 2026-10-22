"""Small annotated declarations and snapshot helpers shared by GUI tests."""

from modello import InstanceDummy, Modello, component
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


def rows(data):
    return {row["path"]: row for row in data["fields"]}

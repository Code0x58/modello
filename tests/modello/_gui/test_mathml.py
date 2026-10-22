"""MathML preserves literal labels, navigable identities, and grouped expressions."""

from xml.etree import ElementTree

from sympy import Rational

from modello import InstanceDummy, Modello
from modello.gui import Explorer, FieldInfo
from tests.modello._gui_models import Estimate, rows


def test_mathml_is_valid_xml_and_escapes_metadata():
    data = Explorer(Estimate, metadata={"panel.width": FieldInfo(symbol='<script>"&')}).snapshot(raw=True)
    for equation in data["equations"]:
        root = ElementTree.fromstring(equation["mathml"])
        assert root.tag == "{http://www.w3.org/1998/Math/MathML}math"
        assert not root.findall(".//script")
    panel = next(e for e in data["equations"] if e["path"] == "panel.area")
    parsed = ElementTree.fromstring(panel["mathml"])
    width = next(element for element in parsed.iter() if element.get("data-path") == "panel.width")
    assert width.text == '<script>"&'
    assert 'data-path="panel.width"' in panel["mathml"]


def test_percent_math_groups_the_whole_expression():
    class Rate(Modello):
        weight = InstanceDummy("weight", real=True)
        result = weight / 20 + Rational(3, 100)

    data = rows(Explorer(Rate, metadata={"result": FieldInfo(number_format="percent")}).snapshot())
    markup = data["result"]["value_mathml"]
    assert "<mo>(</mo>" in markup and "<mo>)</mo>" in markup
    ElementTree.fromstring(markup)

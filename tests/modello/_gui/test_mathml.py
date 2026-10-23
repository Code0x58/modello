"""MathML preserves literal labels, navigable identities, and grouped expressions."""

from xml.etree import ElementTree

from sympy import Derivative, Eq, Function, IndexedBase, Integral, Limit, Ne, Piecewise, Product, Rational, Sum, Symbol

from modello import Index, InstanceDummy, Modello, Vector, vector
from modello._gui.mathml import EquationPrinter
from modello.gui import Explorer, FieldInfo
from tests.modello._gui_models import NS, Estimate, Window, mathml_text, rows


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


def test_raw_vector_formulas_and_symbols_are_valid_navigable_mathml():
    data = Explorer(Window).snapshot(raw=True)
    equations = rows(data, "equations")
    vectors = rows(data, "vectors")
    assert set(equations) == {"total", "growth", "changes"}
    assert "∑" in mathml_text(equations["total"]["mathml"])
    assert "∏" in mathml_text(equations["growth"]["mathml"])
    assert "fori∈{1,2}" in mathml_text(equations["changes"]["mathml"])
    assert ElementTree.fromstring(vectors["samples"]["symbol_mathml"]).find("m:mover", NS) is not None
    assert vectors["samples"]["shape"] == [3]
    assert vectors["samples"]["entries"] == ["samples[0]", "samples[1]", "samples[2]"]
    assert rows(data)["samples[1]"]["parent"] == "samples"
    assert rows(data)["samples[1]"]["unit"] == "m"
    for item in data["fields"] + data["vectors"]:
        for key in ("symbol_mathml", "value_mathml", "exact_mathml"):
            root = ElementTree.fromstring(item[key])
            assert root.tag.endswith("math")
    selected = ElementTree.fromstring(equations["total"]["mathml"]).find(".//m:msub", NS)
    assert selected.get("data-path") == "samples"


def test_fixed_length_window_renders_its_symbolic_origin():
    class ShiftedWindow(Modello):
        offset = InstanceDummy("offset", integer=True)
        i = Index("i")
        values = vector(i**2, (i, offset, offset + 2))

    data = Explorer(ShiftedWindow).snapshot(raw=True)
    equation = rows(data, "equations")["values"]["mathml"]
    root = ElementTree.fromstring(equation)
    assert "fori∈{offset,offset+1,offset+2}" in "".join(root.itertext())
    assert rows(data, "vectors")["values"]["shape"] == [3]


def test_multiple_sum_and_product_limits_are_all_rendered():
    x, y = Symbol("x"), Symbol("y")
    printer = EquationPrinter({})
    for operator, glyph in ((Sum, "∑"), (Product, "∏")):
        root = ElementTree.fromstring(printer.expression(operator(x + y, (x, 0, 2), (y, 1, 3))))
        limits = root.findall(".//m:munderover", NS)
        assert len(limits) == 2
        assert ["".join(limit[1].itertext()) for limit in limits] == ["y=1", "x=0"]
        assert ["".join(limit[0].itertext()) for limit in limits] == [glyph, glyph]


def test_integral_differentials_keep_the_order_of_integration():
    x, y = Symbol("x"), Symbol("y")
    root = ElementTree.fromstring(EquationPrinter({}).expression(Integral(x + y, (x, 0, y), (y, 0, 1))))
    assert [child.text for child in root[0][-4:]] == ["d", "x", "d", "y"]
    assert len(root.findall(".//m:msubsup", NS)) == 2


def test_derivative_powers_belong_on_the_variables_in_the_denominator():
    x = Symbol("x")
    root = ElementTree.fromstring(EquationPrinter({}).expression(Derivative(Function("f")(x), (x, 2))))
    denominator = root.find("m:mfrac/m:mrow[2]", NS)
    assert denominator[0].text == "d"
    assert denominator[1].tag.endswith("msup")
    assert denominator[1][0].text == "x"
    assert denominator[1][1].text == "2"


def test_one_sided_limit_renders_its_direction():
    x = Symbol("x")
    markup = EquationPrinter({}).expression(Limit(1 / x, x, 0, dir="-"))
    assert "limx→0-" in mathml_text(markup)


def test_piecewise_uses_cases_with_navigable_conditions_and_an_explicit_default():
    x = Symbol("x", real=True)
    printer = EquationPrinter({x: ("input", "x")})
    expression = Piecewise((x**2, x < 0), (x, True))
    root = ElementTree.fromstring(printer.expression(expression))
    assert root[0][0].text == "{"
    cases = root.find("m:mrow/m:mtable", NS)
    assert cases.get("columnalign") == "left left"
    assert len(cases) == 2
    assert [len(row) for row in cases] == [2, 2]
    assert "".join(cases[0][1].itertext()) == "ifx<0"
    assert "".join(cases[1][1].itertext()) == "otherwise"
    assert cases[0][1].find(".//m:mi", NS).get("data-path") == "input"
    conditional = ElementTree.fromstring(printer.expression(Piecewise((x**2, x < 0), (x, x > 1))))
    assert "otherwise" not in "".join(conditional.itertext())


def test_relations_are_xml_compatible_with_their_mathematical_operator():
    x = Symbol("x", real=True)
    comparisons = ((x < 0, "<"), (x > 0, ">"), (x <= 0, "≤"), (x >= 0, "≥"), (Eq(x, 0), "="), (Ne(x, 0), "≠"))
    for expression, glyph in comparisons:
        root = ElementTree.fromstring(EquationPrinter({}).expression(expression))
        assert "".join(root.itertext()) == f"x{glyph}0"


def test_literal_function_names_are_xml_text():
    x = Symbol("x")
    name = 'response<&"'
    root = ElementTree.fromstring(EquationPrinter({}).expression(Function(name)(x)))
    assert root.find("m:mrow/m:mi", NS).text == name


def test_native_indexed_vectors_and_multivariate_calculus_are_well_formed():
    x, y = Symbol("x"), Symbol("y")
    array = IndexedBase("a", shape=(3,))
    printer = EquationPrinter({})
    assert ElementTree.fromstring(printer.expression(array)).find("m:mover", NS) is not None
    assert ElementTree.fromstring(printer.expression(array[x])).find("m:msub/m:mi", NS).text == "a"
    derivative = ElementTree.fromstring(printer.expression(Derivative(Function("f")(x, y), x, y)))
    assert "∂" in "".join(derivative.itertext())
    for expression in (Integral(x, x), Integral(x, (x, y)), Limit(1 / x, x, 0, dir="+")):
        ElementTree.fromstring(printer.expression(expression))


def test_temporary_vector_formulas_render_as_their_selected_scalar_expression():
    class Scaled(Modello):
        samples = Vector("x", size=3, real=True)
        total = (2 * samples).sum()

    data = Explorer(Scaled).snapshot(raw=True)
    markup = data["equations"][0]["mathml"]
    assert "∑" in mathml_text(markup)
    assert " for " not in mathml_text(markup)
    assert "mtable" not in markup
    assert "samples" in mathml_text(markup)


def test_multiplication_uses_a_visible_dot_between_literal_quantity_names():
    a, b = Symbol("opening"), Symbol("growth")
    rendered = EquationPrinter({}).expression(a * b)
    assert "·" in mathml_text(rendered)


def test_iteration_notation_has_explicit_spacing_in_mathml():
    data = Explorer(Window).snapshot(raw=True)
    equation = rows(data, "vectors")["changes"]["constraints"][0]["mathml"]
    root = ElementTree.fromstring(equation)
    spaces = root.findall(".//m:mspace", NS)
    assert len(spaces) >= 2
    assert all(float(space.get("width").removesuffix("em")) > 0 for space in spaces)
    assert root.find(".//m:mtext", NS).text == "for"

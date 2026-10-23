"""Relative and continuous source equations remain distinct from sampled values."""

from xml.etree import ElementTree

from examples.gui_timeseries import DEMOS
from modello.gui import Explorer
from tests.modello._gui_models import NS, mathml_text, rows


def test_relative_equation_retains_its_predecessor_and_boundary_and_is_the_query_target():
    model, inputs = DEMOS["relative"]
    data = Explorer(model, inputs=inputs).snapshot(raw=True)
    equations = rows(data, "equations")
    law = ElementTree.fromstring(equations["law"]["mathml"])
    indices = ["".join(node[1].itertext()) for node in law.findall(".//m:msub", NS) if node[0].text == "T"]
    assert "n" in indices and "0" in indices
    assert any("n" in index and "1" in index for index in indices)
    assert "≥1" in "".join(law.itertext())
    definition = next(item for item in rows(data)["closing"]["constraints"] if item["kind"] == "definition")
    assert definition["equation_path"] == "law"
    assert "law" not in rows(data)


def test_continuous_equation_and_conditional_moments_have_mathematical_markup():
    model, inputs = DEMOS["diffusion"]
    explorer = Explorer(model, inputs=inputs)
    data = explorer.snapshot(raw=True)
    equations = rows(data, "equations")
    assert "dW" in mathml_text(equations["process"]["mathml"])
    assert "drift" in mathml_text(equations["log_drift"]["mathml"])
    mean = mathml_text(equations["mean"]["mathml"])
    assert "𝔼" in mean and "|" in mean
    assert "S" in mean and "τ" in mean
    for path, letter, glyph in (("mean", "E", "𝔼"), ("variance", "V", "𝕍")):
        formula = ElementTree.fromstring(equations[path]["mathml"])
        operators = [operator for operator in formula.findall(".//m:mi", NS) if operator.text == glyph]
        assert len(operators) == 1
        assert not formula.findall('.//m:mi[@mathvariant="double-struck"]', NS)
        fences = [node for node in formula.iter() if node.get("fence") == "true"]
        assert len(fences) == 2
        assert all(node.get("stretchy") == "true" for node in fences)
        definition = next(c for c in explorer.problem.constraints if c.path == (path,))
        assert rf"\mathbb{{{letter}}}" in explorer.problem.latex(definition.right)
    for item in data["equations"]:
        ElementTree.fromstring(item["mathml"])

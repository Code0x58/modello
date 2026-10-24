"""Relative and continuous source equations remain distinct from sampled values."""

from xml.etree import ElementTree

from sympy import log

from modello import Coordinate, InstanceDummy, Modello, Series
from modello.contrib.stochastic import ItoProcess
from modello.gui import Explorer, FieldInfo
from tests.modello._gui_models import NS, mathml_text, rows
from tests.modello._models import DrivenSequence


def test_relative_equation_retains_its_predecessor_and_boundary_and_is_the_query_target(relative_inputs):
    data = Explorer(DrivenSequence, inputs=relative_inputs).snapshot(raw=True)
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
    class Brownian(Modello):
        t = Coordinate("t", real=True)
        S = Series("S", t, positive=True)
        growth = InstanceDummy("growth", real=True)
        volatility = InstanceDummy("volatility", nonnegative=True)
        opening = InstanceDummy("opening", positive=True)
        duration = InstanceDummy("duration", nonnegative=True)
        process = ItoProcess(S[t], growth * S[t], volatility * S[t], t)
        log_drift = process.drift(log(S[t]))
        mean = process.mean(opening, duration)
        variance = process.variance(opening, duration)

    explorer = Explorer(Brownian, metadata={"duration": FieldInfo(symbol="τ")})
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

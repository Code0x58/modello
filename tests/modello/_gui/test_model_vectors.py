"""Model collections present typed members without exposing storage columns."""

from xml.etree import ElementTree

from examples.gui_timeseries import DEMOS
from modello import InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.gui import Explorer, FieldInfo, ModelInfo, create_app
from tests.modello._gui_models import mathml_text, rows


def test_model_vector_members_and_state_recurrence_use_mathml_and_bracketed_paths():
    model, inputs = DEMOS["sequence"]
    data = Explorer(model, inputs=inputs).snapshot({"terminal.cash": 908})
    fields = rows(data)
    assert fields["fills[1].price"]["value"] == "110"
    assert fields["states[2].cash"]["value"] == "908"
    assert fields["terminal.cash"]["value"] == "908"
    assert not any("column_" in row["path"] for row in [*data["fields"], *data["vectors"]])
    groups = {row["path"]: row for row in data["components"]}
    assert groups["fills"]["shape"] == [2]
    assert groups["states"]["shape"] == [3]
    assert groups["states[2]"]["parent"] == "states"
    assert groups["fills[1]"]["parent"] == "fills"
    assert groups["states.steps"]["collapsed"] is True
    assert "mover" in groups["states"]["symbol_mathml"]
    law = next(row for row in data["equations"] if row["path"] == "states")
    markup = ElementTree.fromstring(law["mathml"])
    assert "FillPosting.after" in mathml_text(law["mathml"])
    assert "n−1" in mathml_text(law["mathml"])
    assert "msub" in law["mathml"]
    fences = [node for node in markup.iter() if node.get("fence") == "true"]
    assert len(fences) == 3
    assert all(node.get("stretchy") == "true" for node in fences)


def test_sequence_demo_resizes_to_zero_and_restores_scenarios_after_shortening():
    model, inputs = DEMOS["sequence"]
    client = create_app(model, inputs=inputs).test_client()
    shorter = {"fill_quantities": "[2]", "prices": "[100]", "fees": "[1]"}
    empty = {"fill_quantities": "[]", "prices": "[]", "fees": "[]"}
    for proposal, size, cash in [(shorter, 1, "799"), (empty, 0, "1000")]:
        response = client.post("/api/explore", json={"bindings": proposal})
        assert response.status_code == 200
        assert rows(response.json)["N"]["value"] == str(size)
        assert rows(response.json)["terminal.cash"]["value"] == cash
        assert any(scenario["bindings"].get("prices[1]") == "110" for scenario in response.json["scenarios"])
    restored = client.post("/api/explore", json={"bindings": {"prices[1]": "110"}})
    assert restored.status_code == 200
    assert rows(restored.json)["N"]["value"] == "2"
    assert rows(restored.json)["terminal.cash"]["value"] == "908"


def test_model_vector_element_metadata_uses_the_same_bracketed_paths_as_overrides():
    @ModelInfo(fields={"value": FieldInfo(title="Declared member value")})
    class Element(Modello):
        value = InstanceDummy("value", real=True)

    class Model(Modello):
        items = model_vector(Element, size=2)

    data = Explorer(Model, metadata={"items[1].value": FieldInfo(title="Second value")}).snapshot(raw=True)
    assert rows(data)["items[0].value"]["title"] == "Declared member value"
    assert rows(data)["items[1].value"]["title"] == "Second value"


def test_nested_collections_without_scalar_fields_still_specialize():
    class Empty(Modello):
        pass

    class Empties(Modello):
        N = InstanceDummy("N", integer=True, nonnegative=True)
        items = model_vector(Empty, size=N)

    class Parent(Modello):
        child = component(Empties)

    supplied = Empties("source", items=[Empty("a"), Empty("b")])
    copied = Parent("parent", child=supplied)
    assert len(copied.child.items) == copied.child.N == 2
    assert len(copied.child.items[0]._modello_bound_fields) == 0


def test_chained_recurrence_markup_names_the_selected_prior_boundary():
    from examples.model_sequences import AccountSequence
    from examples.portfolio.models import FillPosting

    class Chained(AccountSequence):
        continuation = model_recurrence(FillPosting, initial=AccountSequence.states[-1], size=0)

    data = Explorer(Chained, inputs={"N": 2}).snapshot(raw=True)
    equation = next(row for row in data["equations"] if row["path"] == "continuation")
    assert "continuation[0] = states[2]" in equation["text"]
    assert "<msub><mi>states</mi><mn>2</mn></msub>" in equation["mathml"]
    assert "∅" in equation["mathml"]

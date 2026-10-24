"""Model collections present typed members without exposing storage columns."""

import json
from xml.etree import ElementTree

from examples.gallery.catalog import EXAMPLES
from examples.portfolio.models import AccountSequence
from modello import Index, InstanceDummy, Modello, component, model_recurrence, model_vector
from modello.gui import Explorer, FieldInfo, ModelInfo, Scenario, create_app
from tests.modello._gui_models import mathml_text, rows


def test_model_vector_members_and_state_recurrence_use_mathml_and_bracketed_paths():
    example = EXAMPLES["sequence"]
    model, inputs = example.model, example.inputs
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
    assert not any("steps" in path or "before" in path or "after" in path for path in groups)
    assert not data["vectors"]
    assert "mover" in groups["states"]["symbol_mathml"]
    law = next(row for row in data["equations"] if row["path"] == "law")
    markup = ElementTree.fromstring(law["mathml"])
    assert "FillPosting" not in mathml_text(law["mathml"])
    assert "Qn" in mathml_text(law["mathml"])
    assert "Cn" in mathml_text(law["mathml"])
    assert "n−1" in mathml_text(law["mathml"])
    assert "msub" in law["mathml"]
    fences = [node for node in markup.iter() if node.get("fence") == "true"]
    assert len(fences) == 1
    assert all(node.get("stretchy") == "true" for node in fences)
    notional = next(row for row in data["equations"] if row["path"] == "fills[0].notional")
    assert "fills" not in mathml_text(notional["mathml"])
    assert "v0" in mathml_text(notional["mathml"])
    assert "u0" in mathml_text(notional["mathml"])
    assert "p0" in mathml_text(notional["mathml"])
    assert "fills[0].notional" in notional["mathml"]  # Links retain the API path.
    assert not any(eq["path"].startswith("states[") for eq in data["equations"])
    assert fields["states[1].cash"]["constraints"][0]["equation_path"] == "law"


def test_sequence_demo_resizes_to_zero_and_restores_scenarios_after_shortening():
    example = EXAMPLES["sequence"]

    @ModelInfo(
        scenarios=(
            Scenario(
                title="Original second price",
                description="Restore the original horizon.",
                bindings={"fills[1].price": 110},
            ),
        )
    )
    class WithOriginalCoordinate(AccountSequence):
        pass

    client = create_app(WithOriginalCoordinate, inputs=example.inputs).test_client()
    shorter = {"fills": json.dumps([{"quantity": "2", "price": "100", "fee": "1"}])}
    empty = {"fills": "[]"}
    for proposal, size, cash in [(shorter, 1, "799"), (empty, 0, "1000")]:
        response = client.post("/api/explore", json={"bindings": proposal})
        assert response.status_code == 200
        assert rows(response.json)["N"]["value"] == str(size)
        assert rows(response.json)["terminal.cash"]["value"] == cash
        assert any(scenario["bindings"].get("fills[1].price") == "110" for scenario in response.json["scenarios"])
    restored = client.post("/api/explore", json={"bindings": {"fills[1].price": "110"}})
    assert restored.status_code == 200
    assert rows(restored.json)["N"]["value"] == "2"
    assert rows(restored.json)["terminal.cash"]["value"] == "908"


def test_model_records_keep_only_observed_cells_and_preserve_editor_arithmetic():
    example = EXAMPLES["sequence"]
    model, inputs = example.model, example.inputs
    client = create_app(model, inputs=inputs).test_client()
    bindings = {"fills": json.dumps([{"quantity": "2", "price": "10**20 * 10**20", "fee": "1"}])}
    response = client.post("/api/explore", json={"bindings": bindings})
    assert response.status_code == 200
    execution = next(row for row in response.json["components"] if row["path"] == "fills")
    assert execution["observations"][0]["price"] == "10**20 * 10**20"
    assert execution["observations"][0]["notional"] is None
    target = client.post("/api/explore", json={"bindings": {"terminal.cash": "908"}}).json
    accounts = next(row for row in target["components"] if row["path"] == "states")
    assert accounts["observations"][-1]["cash"] is None  # A terminal target stays attached to terminal.
    invalid = {"fills": json.dumps([{"price": "__import__('os').getcwd()"}])}
    assert client.post("/api/explore", json={"bindings": invalid}).status_code == 422
    assert client.post("/api/explore", json={"bindings": {}}).status_code == 200


def test_unobserved_model_records_resize_with_the_related_execution_window():
    example = EXAMPLES["sequence"]
    model, original = example.model, example.inputs
    inputs = {"fills": original["fills"], "states": [{}, {}, {}]}
    client = create_app(model, inputs=inputs).test_client()
    added = {"fills": json.dumps([{"quantity": "2", "price": "100", "fee": "1"}, {}, {}])}
    response = client.post("/api/explore", json={"bindings": added})
    assert response.status_code == 200
    assert rows(response.json)["N"]["value"] == "3"
    states = next(component for component in response.json["components"] if component["path"] == "states")
    assert states["shape"] == [4]


def test_model_vector_element_metadata_uses_the_same_bracketed_paths_as_overrides():
    @ModelInfo(fields={"value": FieldInfo(title="Declared member value")})
    class Element(Modello):
        value = InstanceDummy("value", real=True)

    class Model(Modello):
        items = model_vector(Element, size=2)

    data = Explorer(Model, metadata={"items[1].value": FieldInfo(title="Second value")}).snapshot(raw=True)
    assert rows(data)["items[0].value"]["title"] == "Declared member value"
    assert rows(data)["items[1].value"]["title"] == "Second value"


def test_nested_member_laws_use_distinct_public_paths_instead_of_storage_columns():
    @ModelInfo(fields={"quantity": FieldInfo(symbol="q")})
    class Item(Modello):
        quantity = InstanceDummy("quantity", real=True)

    class Row(Modello):
        holdings = model_vector(Item, size=2)

    class Panel(Modello):
        rows = model_vector(Row, size=3)
        n = Index("n")
        law = model_recurrence(
            rows[n],
            **{
                "holdings[0].quantity": rows[n - 1].holdings[0].quantity + 1,
                "holdings[1].quantity": rows[n - 1].holdings[1].quantity + 10,
            },
        )

    for metadata, expected in (
        ({}, ("holdings[0].quantity", "holdings[1].quantity")),
        (
            {
                "rows[].holdings[0].quantity": FieldInfo(symbol="q₀"),
                "rows[].holdings[1].quantity": FieldInfo(symbol="q₁"),
            },
            ("q₀", "q₁"),
        ),
    ):
        data = Explorer(Panel, metadata=metadata).snapshot(raw=True)
        law = next(equation for equation in data["equations"] if equation["path"] == "law")
        for text in (law["text"], mathml_text(law["mathml"])):
            assert "column_" not in text
            assert all(label in text for label in expected)
        assert "column_" not in " ".join(field["path"] for field in data["fields"])


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


def test_member_schema_constraints_are_retained_when_a_law_also_defines_that_field():
    class State(Modello):
        value = InstanceDummy("value", real=True)
        twice = 2 * value

    class Model(Modello):
        states = model_vector(State, size=2)
        n = Index("n")
        law = model_recurrence(states[n], value=states[n - 1].value + 1, twice=0)

    data = Explorer(Model).snapshot(raw=True)
    equations = [eq for eq in data["equations"] if eq["path"] == "states[1].twice"]
    assert len(equations) == 2
    assert any("0" in mathml_text(eq["mathml"]) for eq in equations)
    assert any("2" in mathml_text(eq["mathml"]) for eq in equations)

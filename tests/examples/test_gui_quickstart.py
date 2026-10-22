"""The GUI quickstart supports forward, inverse, and free exploration."""

from modello.gui import create_app
from tests.modello._gui_models import rows


def test_quickstart_supports_forward_inverse_and_free_exploration():
    from examples.gui_quickstart import ReturnModel

    client = create_app(ReturnModel, inputs={"capital": 100_000}).test_client()
    initial = client.post("/api/explore", json={}).json
    assert rows(initial)["expected_return"]["kind"] == "free"
    assert initial["status"] == "conditional"
    bindings = initial["scenarios"][0]["bindings"]
    forward = client.post("/api/explore", json={"bindings": bindings}).json
    assert rows(forward)["gain"]["value"] == "8000"
    changed = client.post("/api/explore", json={"bindings": {**bindings, "capital": "200000"}}).json
    assert rows(changed)["gain"]["value"] == "16000"
    released = client.post("/api/explore", json={"bindings": {"expected_return": None}}).json
    assert rows(released)["expected_return"]["kind"] == "free"
    inverse = client.post("/api/explore", json={"bindings": {"gain": "8000"}}).json
    assert rows(inverse)["expected_return"]["display_value"] == "8%"
    assert rows(inverse)["expected_return"]["kind"] == "derived"

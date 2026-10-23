"""Resized observation windows remain stateless, reversible proposals."""

from sympy import Rational

from examples.gui_timeseries import DEMOS
from modello.gui import create_app


def test_state_vector_lengths_resize_the_model_and_restore_the_original_window():
    model, inputs = DEMOS["relative"]
    client = create_app(model, inputs=inputs).test_client()
    for text, size in [("[1, 2, 4, 8, 16, 32, 64]", 6), ("[1, 2, 4]", 2), ("[1]", 0)]:
        response = client.post("/api/explore", json={"bindings": {"levels": text}})
        assert response.status_code == 200
        data = response.json
        fields = {row["path"]: row for row in data["fields"]}
        assert fields["N"]["value"] == str(size)
        assert data["bindings"] == {"levels": text}
        levels = next(row for row in data["vectors"] if row["path"] == "levels")
        assert levels["shape"] == [size + 1]
        assert levels["resizable"] is True
        assert levels["initial_shape"] == [5]
        assert levels["initial_input"] == "[2, 2, 1, 5/2, 5/4]"
        assert levels["window_changed"] is True
        assert levels["input_state"] == "overridden"
    restored = client.post("/api/explore", json={"bindings": {}}).json
    assert next(row for row in restored["fields"] if row["path"] == "N")["value"] == "4"
    assert inputs["levels"] == [2, 2, 1, Rational(5, 2), Rational(5, 4)]


def test_unknown_states_still_supply_a_length_and_observed_innovations_still_constrain_them():
    model, inputs = DEMOS["relative"]
    client = create_app(model, inputs=inputs).test_client()
    unknown = client.post("/api/explore", json={"bindings": {"levels": "[?, ?, ?]"}})
    assert unknown.status_code == 200
    assert next(row for row in unknown.json["fields"] if row["path"] == "N")["value"] == "2"
    conflict = client.post("/api/explore", json={"bindings": {"levels": "[1, 2, 3]", "innovations": "[10, 10]"}})
    assert conflict.status_code == 422
    assert client.post("/api/explore", json={"bindings": {"levels": "[]"}}).status_code == 422
    assert client.post("/api/explore", json={"bindings": {}}).status_code == 200


def test_math_font_and_license_are_served_locally():
    model, inputs = DEMOS["relative"]
    client = create_app(model, inputs=inputs).test_client()
    font = client.get("/assets/fonts/STIXTwoMath-Regular.woff2")
    assert font.status_code == 200
    assert font.data.startswith(b"wOF2")
    assert b"SIL OPEN FONT LICENSE" in client.get("/assets/fonts/OFL.txt").data

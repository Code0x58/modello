"""Resized observation windows remain stateless, reversible proposals."""

import pytest
from sympy import Tuple

from modello.gui import create_app
from tests.modello._models import DrivenSequence


@pytest.mark.parametrize(
    "text, size",
    [("[1, 2, 4, 8, 16, 32, 64]", 6), ("[1, 2, 4]", 2), ("[1]", 0)],
    ids=["longer", "shorter", "history-only"],
)
def test_state_vector_lengths_resize_the_model_and_restore_the_original_window(relative_client, text, size):
    response = relative_client.post("/api/explore", json={"bindings": {"levels": text}})
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
    restored = relative_client.post("/api/explore", json={"bindings": {}}).json
    assert next(row for row in restored["fields"] if row["path"] == "N")["value"] == "4"


def test_unknown_states_still_supply_a_length_and_observed_innovations_still_constrain_them(relative_client):
    unknown = relative_client.post("/api/explore", json={"bindings": {"levels": "[?, ?, ?]"}})
    assert unknown.status_code == 200
    assert next(row for row in unknown.json["fields"] if row["path"] == "N")["value"] == "2"
    conflict = relative_client.post(
        "/api/explore", json={"bindings": {"levels": "[1, 2, 3]", "innovations": "[10, 10]"}}
    )
    assert conflict.status_code == 422
    assert relative_client.post("/api/explore", json={"bindings": {"levels": "[]"}}).status_code == 422
    assert relative_client.post("/api/explore", json={"bindings": {}}).status_code == 200


def test_math_font_and_license_are_served_locally(relative_client):
    font = relative_client.get("/assets/fonts/STIXTwoMath-Regular.woff2")
    assert font.status_code == 200
    assert font.data.startswith(b"wOF2")
    assert b"SIL OPEN FONT LICENSE" in relative_client.get("/assets/fonts/OFL.txt").data


def test_sympy_tuple_observations_use_the_same_shape_validation_as_python_sequences():
    client = create_app(DrivenSequence, inputs={"retention": 1, "levels": Tuple(1, 3, 6)}).test_client()
    response = client.post("/api/explore", json={"bindings": {"levels": "[1, 3, 6, 10]"}})
    assert response.status_code == 200
    assert next(row for row in response.json["fields"] if row["path"] == "N")["value"] == "3"

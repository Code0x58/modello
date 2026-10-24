"""The actual frontend retains coherent proposals after bulk and scalar edits."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from sympy import Rational

from modello import Modello, Vector
from modello.gui import FieldInfo, create_app
from tests.modello._models import DrivenSequence


def test_frontend_vector_proposal_state(tmp_path: Path, pytestconfig: pytest.Config) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("The optional frontend state check requires Node.js")

    class Window(Modello):
        samples = Vector("x", size=3, real=True)
        total = samples.sum()
        doubled = 2 * samples

    client = create_app(Window, inputs={"samples": [1, None, 3]}).test_client()

    def propose(bindings):
        response = client.post("/api/explore", json={"bindings": bindings})
        assert response.status_code == 200
        return response.json

    initial = propose({})
    bulk = propose({"samples": "[1, ?, 3]", "total": "6"})
    edited = propose({**bulk["bindings"], "samples[1]": "2"})
    restored = propose({"total": "6"})
    rejected = client.post("/api/explore", json={"bindings": {"total": "6", "samples[1]": "4"}})
    assert rejected.status_code == 422
    from examples.gallery.catalog import EXAMPLES

    example = EXAMPLES["simple"]
    fundamental_client = create_app(example.model, inputs=example.inputs).test_client()
    fundamental = fundamental_client.post("/api/explore", json={"bindings": {}}).json
    assert fundamental is not None
    conflict = fundamental_client.post("/api/explore", json={"bindings": fundamental["scenarios"][4]["bindings"]})
    assert conflict.status_code == 422
    law_client = create_app(
        DrivenSequence,
        inputs={"retention": Rational(1, 2), "levels": [2, 2, 1, Rational(5, 2), Rational(5, 4)]},
        metadata={"law": FieldInfo(title="Relative recurrence and boundary")},
    ).test_client()
    laws = law_client.post("/api/explore", json={"bindings": {}})
    assert laws.status_code == 200
    law_edits = {}
    for name, text in {
        "longer": "[2, 3, 5, 8, 13, 21, 34]",
        "entry_edit": "[2, 3, 5, 8, 13, 21, 35]",
        "entry_restore": "[2, 3, 5, 8, 13, 21, ?]",
    }.items():
        response = law_client.post("/api/explore", json={"bindings": {"levels": text}})
        assert response.status_code == 200
        law_edits[name] = response.json
    fixtures = tmp_path / "state.json"
    example = EXAMPLES["sequence"]
    model, inputs = example.model, example.inputs
    sequence_client = create_app(model, inputs=inputs).test_client()
    sequence = sequence_client.post("/api/explore", json={"bindings": {}})
    assert sequence.status_code == 200
    data = sequence.get_json()
    assert data is not None
    records = next(component["observations"] for component in data["components"] if component["path"] == "fills")
    added = [*records, {}]
    cell = [*records, {"price": "110"}]
    sequence_edits = {}
    for name, observations in {"added": added, "cell": cell, "removed": records[:1], "empty": []}.items():
        response = sequence_client.post(
            "/api/explore", json={"bindings": {"N": None, "fills": json.dumps(observations)}}
        )
        assert response.status_code == 200, response.json
        sequence_edits["sequence_" + name] = response.json
    fixtures.write_text(
        json.dumps(
            dict(
                initial=initial,
                bulk=bulk,
                edited=edited,
                restored=restored,
                rejected=rejected.json,
                fundamental=fundamental,
                conflict=conflict.json,
                laws=laws.json,
                sequence=sequence.json,
                **sequence_edits,
                **law_edits,
            )
        )
    )
    result = subprocess.run(
        [
            node,
            str(pytestconfig.rootpath / "tests/modello/_gui/test_explorer.cjs"),
            str(pytestconfig.rootpath / "modello/_gui/static/explorer.js"),
            str(fixtures),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

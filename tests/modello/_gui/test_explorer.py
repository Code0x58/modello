"""The actual frontend retains coherent proposals after bulk and scalar edits."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from modello import Modello, Vector
from modello.gui import create_app


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
    fixtures = tmp_path / "state.json"
    fixtures.write_text(
        json.dumps(dict(initial=initial, bulk=bulk, edited=edited, restored=restored, rejected=rejected.json))
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

"""Replay artifacts preserve model branches and distinguish kinds of evidence."""

import json
import subprocess
import sys

import pytest

from examples.research_models._reporting import _provenance, write_reports
from examples.research_models._study import Computation, Study, Variant
from examples.research_models.study import MODELS, build_collection, write_collection
from modello import InstanceDummy
from modello.contrib.research import Method, ResearchCase, ResearchModel


@pytest.fixture(scope="module")
def compact_collection(tmp_path_factory):
    directory = tmp_path_factory.mktemp("research_collection")
    payload = write_collection(directory, compact=True)
    return directory, payload


def _check(report, name):
    return next(item for item in report["checks"] if item["id"] == name)


def test_all_model_subdirectories_export_evidence_results_figures_and_provenance(compact_collection):
    directory, payload = compact_collection
    assert set(payload["studies"]) == set(MODELS)
    assert payload["selection"] == list(MODELS)
    saved = json.loads((directory / "collection.json").read_text())
    assert saved["compact_selection"] and saved["numerical_references"]
    assert saved["provenance"]["source_sha256"]
    assert "examples/_research_checks.py" in saved["provenance"]["source_sha256"]
    for name, study in payload["studies"].items():
        output = directory / name
        assert (output / "report.html").read_text().startswith("<!doctype html>")
        assert (output / "results.csv").read_text().startswith("variant,computation,quantity,value,kind")
        assert json.loads((output / "report.json").read_text())["provenance"] == saved["provenance"]
        assert list(output.glob("evidence-*.html"))
        assert {"overview.svg", "overview.png"} == set(study["figures"])
        for filename in study["figures"]:
            assert (output / "figures" / filename).stat().st_size > 1000
    assert "deserialized or executed" in (directory / "report.html").read_text()


def test_bsm_keeps_exact_identities_and_numerical_quadrature_separate(compact_collection):
    _, payload = compact_collection
    variant = payload["studies"]["black-scholes"]["variants"]["affine_with_dividends"]
    exact = variant["computations"]["identities"]
    numerical = variant["computations"]["quadrature"]
    assert set(exact["data"]["residuals"].values()) == {"0"}
    assert exact["kind"] == "exact scalar checks"
    assert numerical["kind"] == "numerical observation"
    assert numerical["data"]["call_error"] < 1e-9
    assert (
        _check(variant["blocked"]["dividend_free_shortcut"]["evidence"], "dividend_free_parity")["verdict"]
        == "strictly false"
    )


def test_feller_failure_does_not_erase_successful_bond_valuation(compact_collection):
    _, payload = compact_collection
    variant = payload["studies"]["cir-bond"]["variants"]["feller_not_satisfied"]
    assert _check(variant["computations"]["price"]["evidence"], "affine_valuation")["verdict"] == "strictly true"
    boundary = _check(variant["blocked"]["boundary"]["evidence"], "unattainable_zero")
    assert boundary["verdict"] == "unresolved"
    assert any(
        item["id"].endswith("/feller_condition") and item["verdict"] == "strictly false"
        for item in boundary["premises"]
    )


def test_piecewise_regularity_and_nested_certificates_are_exported(compact_collection):
    directory, payload = compact_collection
    variant = payload["studies"]["perpetual-put"]["variants"]["smooth_fit"]
    certificate = variant["computations"]["scalar_certificate"]["data"]
    assert _check(certificate["regularity"], "global_c1")["verdict"] == "strictly true"
    assert _check(certificate["regularity"], "global_c2")["verdict"] == "strictly false"
    assert certificate["one_sided"]["delta_left"] == certificate["one_sided"]["delta_right"] == "-1"
    assert (directory / "perpetual-put" / "evidence-smooth_fit-scalar_certificate-data-regularity.html").is_file()
    assert "untrusted_stopping" in variant["blocked"]


def test_symbolic_report_writer_escapes_labels_and_never_needs_optional_dependencies(tmp_path):
    class Tiny(ResearchModel):
        value = InstanceDummy("value", real=True)

    case = ResearchCase.for_model(Tiny).observe(value=2)
    result = case.run(Method("scalar result", lambda current: current.expression("value")))
    study = Study(
        "tiny", "<script>bad title</script>", "a < b", (), {"<branch>": Variant(case, {"result": Computation(result)})}
    )
    payload = write_reports((study,), tmp_path, numerical=False, compact=True)
    html = (tmp_path / "tiny" / "report.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;branch&gt;" in html
    assert payload["studies"]["tiny"]["variants"]["<branch>"]["computations"]["result"]["data"] == "2"
    assert payload["studies"]["tiny"]["figures"] == []
    assert payload["selection"] == ["tiny"]
    assert "--models tiny" in (tmp_path / "report.html").read_text()


def test_collection_imports_are_lightweight_and_module_entrypoint_is_guarded():
    program = (
        "import sys; import examples.research_models; import examples.research_models.study; "
        "import examples.research_models.__main__; "
        "assert not {'numpy','scipy','matplotlib'} & sys.modules.keys(); "
        "assert 'examples.research_models.black_scholes' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_invalid_model_selections_fail_before_running_unrelated_studies():
    for selected in ((), ("unknown-model",)):
        with pytest.raises(ValueError, match="Select one or more"):
            build_collection(models=selected, numerical=False)


def test_git_is_optional_in_collection_provenance(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("Git is not installed")

    monkeypatch.setattr(subprocess, "run", missing_git)
    provenance = _provenance()
    assert provenance["git_revision"] is provenance["working_tree_dirty"] is None
    assert provenance["source_sha256"]

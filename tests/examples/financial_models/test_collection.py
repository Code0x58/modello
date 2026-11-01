"""Replay artifacts preserve model branches and distinguish kinds of evidence."""

import json
import re
import shlex
import subprocess
import sys
from dataclasses import replace

import pytest

from examples.financial_models._reporting import _provenance, write_reports
from examples.financial_models._study import Blocked, Computation, Study, Variant
from examples.financial_models.study import MODELS, build_collection, main, write_collection
from modello import InstanceDummy
from modello.contrib.verification import Method, VerificationContext, VerificationModel, positive


@pytest.fixture(scope="module")
def compact_collection(tmp_path_factory):
    directory = tmp_path_factory.mktemp("financial_collection")
    payload = write_collection(directory, compact=True)
    return directory, payload


def _verify(report, name):
    return next(item for item in report["checks"] if item["id"] == name)


def test_all_model_subdirectories_export_evidence_results_figures_and_provenance(compact_collection, monkeypatch):
    directory, payload = compact_collection
    assert set(payload["studies"]) == set(MODELS)
    assert payload["selection"] == list(MODELS)
    saved = json.loads((directory / "collection.json").read_text())
    assert saved["compact_selection"] and saved["numerical_references"]
    assert saved["provenance"]["source_sha256"]
    assert "examples/_verification_checks.py" in saved["provenance"]["source_sha256"]
    for name, study in payload["studies"].items():
        output = directory / name
        assert (output / "report.html").read_text().startswith("<!doctype html>")
        assert (output / "results.csv").read_text().startswith("variant,computation,quantity,value,kind")
        assert json.loads((output / "report.json").read_text())["provenance"] == saved["provenance"]
        assert list(output.glob("evidence-*.html"))
        assert {"overview.svg", "overview.png"} == set(study["figures"])
        for filename in study["figures"]:
            assert (output / "figures" / filename).stat().st_size > 1000
    html = (directory / "report.html").read_text()
    assert "deserialized or executed" in html
    command = re.search(r"<pre>(python -m examples\.financial_models[^<]*)</pre>", html)
    assert command is not None
    calls = []
    monkeypatch.setattr(
        "examples.financial_models.study.write_collection", lambda output, **settings: calls.append((output, settings))
    )
    main(shlex.split(command.group(1))[3:])
    assert len(calls) == 1 and calls[0][0].name == "OUTPUT_DIRECTORY"
    assert calls[0][1] == {"models": list(MODELS), "numerical": True, "compact": True}


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
        _verify(variant["blocked"]["dividend_free_shortcut"]["evidence"], "dividend_free_parity")["verdict"]
        == "strictly false"
    )


def test_feller_failure_does_not_erase_successful_bond_valuation(compact_collection):
    _, payload = compact_collection
    variant = payload["studies"]["cir-bond"]["variants"]["feller_not_satisfied"]
    assert _verify(variant["computations"]["price"]["evidence"], "affine_valuation")["verdict"] == "strictly true"
    boundary = _verify(variant["blocked"]["boundary"]["evidence"], "unattainable_zero")
    assert boundary["verdict"] == "unresolved"
    assert any(
        item["id"].endswith("/feller_condition") and item["verdict"] == "strictly false"
        for item in boundary["premises"]
    )


def test_piecewise_regularity_and_nested_certificates_are_exported(compact_collection):
    directory, payload = compact_collection
    variant = payload["studies"]["perpetual-put"]["variants"]["smooth_fit"]
    certificate = variant["computations"]["scalar_certificate"]["data"]
    assert _verify(certificate["regularity"], "global_c1")["verdict"] == "strictly true"
    assert _verify(certificate["regularity"], "global_c2")["verdict"] == "strictly false"
    assert certificate["one_sided"]["delta_left"] == certificate["one_sided"]["delta_right"] == "-1"
    assert (directory / "perpetual-put" / "evidence-smooth_fit-scalar_certificate-data-regularity.html").is_file()
    assert "untrusted_stopping" in variant["blocked"]


@pytest.mark.parametrize("key", ("tiny", "markowitz"))
def test_symbolic_report_writer_escapes_labels_and_never_needs_optional_dependencies(tmp_path, key):
    class Tiny(VerificationModel):
        value = InstanceDummy("value", real=True)

    case = VerificationContext.for_model(Tiny).observe(value=2)
    result = case.run(Method("scalar result", lambda current: current.expression("value")))
    study = Study(
        key, "<script>bad title</script>", "a < b", (), {"<branch>": Variant(case, {"result": Computation(result)})}
    )
    payload = write_reports((study,), tmp_path, numerical=False, compact=True)
    html = (tmp_path / key / "report.html").read_text()
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "&lt;branch&gt;" in html
    assert payload["studies"][key]["variants"]["<branch>"]["computations"]["result"]["data"] == "2"
    assert payload["studies"][key]["figures"] == []
    assert payload["selection"] == [key]
    index = (tmp_path / "report.html").read_text()
    assert "python -m examples.financial_models" not in index
    assert "constructing Python code" in index


@pytest.mark.parametrize("collision", ("variant", "computation", "nested", "blocked", "combined", "case", "long"))
def test_collection_links_keep_distinct_evidence_when_output_names_collide(tmp_path, collision):
    class Tiny(VerificationModel):
        value = InstanceDummy("value", real=True)

    cases = tuple(
        VerificationContext.for_model(Tiny).observe(value=value).with_obligations(positive("value", name="sign"))
        for value in (1, -1)
    )
    results = tuple(case.run(Method("read", lambda current: current.expression("value"))) for case in cases)
    case = cases[0]
    good, bad = results
    labels = ("cap <= 10%", "cap >= 10%")
    expected = ("strictly true", "strictly false")
    if collision == "computation":
        variants = {"branch": Variant(case, {labels[0]: Computation(good), labels[1]: Computation(bad)})}
    elif collision in {"nested", "blocked"}:
        reports = {labels[0]: good.report, labels[1]: bad.report} if collision == "nested" else {"blocked": good.report}
        computations = {"read": Computation(replace(good, data=reports))}
        blocked = {} if collision == "nested" else {"read-data": Blocked(bad.report, "Unestablished sign")}
        variants = {"branch": Variant(case, computations, blocked)}
        expected = ("strictly true", "strictly true", "strictly false")
    elif collision == "combined":
        variants = {"a-b": Variant(case, {"c": Computation(good)}), "a": Variant(cases[1], {"b-c": Computation(bad)})}
    else:
        if collision == "case":
            labels = ("BRANCH", "branch")
        elif collision == "long":
            labels = ("branch" * 30 + "left", "branch" * 30 + "right")
        name = "read" * 40 if collision == "long" else "read"
        variants = {
            label: Variant(branch, {name: Computation(result)}) for label, branch, result in zip(labels, cases, results)
        }
    study = Study("tiny", "Distinct evidence", "Colliding labels", (), variants)
    write_reports((study,), tmp_path, numerical=False, compact=True)
    output = tmp_path / "tiny"
    html = (output / "report.html").read_text()
    links = re.findall(r'href="(evidence-[^"]+\.html)"', html)
    assert len(links) == len({name.casefold() for name in links}) == len(expected)
    assert len(list(output.glob("evidence-*.html"))) == len(expected)
    for filename, verdict in zip(links, expected):
        assert f"<strong>{verdict}</strong>" in (output / filename).read_text()
    write_reports((study,), tmp_path, numerical=False, compact=True)
    assert (output / "report.html").read_text() == html


def test_collection_imports_are_lightweight_and_module_entrypoint_is_guarded():
    program = (
        "import sys; import examples.financial_models; import examples.financial_models.study; "
        "import examples.financial_models.__main__; "
        "assert not {'numpy','scipy','matplotlib'} & sys.modules.keys(); "
        "assert 'examples.financial_models.black_scholes' not in sys.modules"
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

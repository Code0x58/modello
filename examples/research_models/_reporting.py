"""Escaped offline collection reports; JSON is descriptive and never reloaded."""

from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from hashlib import sha256
from html import escape
from importlib import metadata
from pathlib import Path
from typing import Any

from sympy import Basic

from modello import ConstraintKind
from modello.contrib.research import ResearchCase, ResearchReport

from ._study import Study, Variant


def encode(value):
    """Normalize immutable result records without copying MappingProxyType."""
    if isinstance(value, ResearchReport):
        return value.to_dict()
    if isinstance(value, Basic):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): encode(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [encode(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"The audit export does not support {type(value).__name__}")


def _slug(value: str):
    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")[:120] or "record"


def _provenance() -> dict[str, Any]:
    import modello.contrib.research as research

    root = Path(__file__).resolve().parents[2]
    revision: subprocess.CompletedProcess[str] | None
    status: subprocess.CompletedProcess[str] | None
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        revision = status = None
    versions: dict[str, str | None] = {}
    for package in ("modello", "sympy", "numpy", "scipy", "matplotlib"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    shared = Path(__file__).parent.parent / "_research_checks.py"
    sources = {"examples/_research_checks.py": sha256(shared.read_bytes()).hexdigest()}
    for prefix, directory in (
        ("examples/research_models", Path(__file__).parent),
        ("modello/contrib/research", Path(research.__file__).parent),
    ):
        for path in sorted(directory.glob("*.py")):
            sources[f"{prefix}/{path.name}"] = sha256(path.read_bytes()).hexdigest()
    return {
        "git_revision": revision.stdout.strip() if revision is not None and revision.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout) if status is not None and status.returncode == 0 else None,
        "python": platform.python_version(),
        "versions": versions,
        "source_sha256": sources,
    }


def _case_record(case: ResearchCase):
    return {
        "model": case.problem.model.__name__,
        "name": case.problem.name,
        "assumptions": [str(condition) for condition in case.assumptions],
        "changes": list(case.changes),
        "inputs": [
            {
                "path": ".".join(item.path),
                "origin": str(item.kind),
                "value": str(item.right),
                "input_group": item.input,
            }
            for item in case.problem.constraints
            if item.kind is not ConstraintKind.DEFINITION
        ],
    }


def _reports(value, path="data"):
    if isinstance(value, ResearchReport):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _reports(item, f"{path}-{key}")
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _reports(getattr(value, field.name), f"{path}-{field.name}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _reports(item, f"{path}-{index}")


def _facts(value, path=""):
    if isinstance(value, (ResearchReport, ResearchCase)):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _facts(item, f"{path}.{key}" if path else str(key))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _facts(getattr(value, field.name), f"{path}.{field.name}" if path else field.name)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _facts(item, f"{path}[{index}]")
    else:
        yield path or "value", encode(value)


def _summary(value):
    facts = list(_facts(value))
    preferred = [
        (path, item)
        for path, item in facts
        if "." not in path and "[" not in path and path not in {"interpretation", "kind", "statement", "source"}
    ]
    return "; ".join(f"{path}={item}" for path, item in (preferred or facts)[:7])


def _page(title, body):
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>"
        "body{max-width:75rem;margin:2rem auto;padding:0 1rem;font:17px/1.55 system-ui;color:#182a3b}"
        "h1,h2,h3{line-height:1.25}article{border-top:1px solid #ccd6df;padding:1rem 0}"
        "a{color:#075a8c}code,pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "table{width:100%;border-collapse:collapse}td,th{text-align:left;vertical-align:top;"
        "padding:.6rem;border-bottom:1px solid #ccd6df;overflow-wrap:anywhere}"
        ".kind{font-size:.85em;color:#43586a}.blocked{border-left:4px solid #a65225;padding-left:1rem}"
        "img{width:100%;height:auto}figure{margin:2rem 0}details{margin:1rem 0}"
        f"</style></head><body><h1>{escape(title)}</h1>{body}</body></html>"
    )


def _counts(report):
    return ", ".join(
        f"{amount} {verdict}" for verdict, amount in Counter(item.verdict for item in report.checks).items()
    )


def _variant(variant: Variant, label: str, directory: Path, filenames: set[str]):
    def evidence(name, suffix, report):
        stem = f"evidence-{_slug(label)}-{_slug(name)}-{_slug(suffix)}"[:200]
        filename, index = f"{stem}.html", 2
        while filename.casefold() in filenames:
            filename = f"{stem}-{index}.html"
            index += 1
        filenames.add(filename.casefold())
        (directory / filename).write_text(report.to_html(title=f"{label}: {name}: {suffix}"), encoding="utf-8")
        return filename

    encoded: dict[str, Any] = {
        "case": _case_record(variant.case),
        "computations": {},
        "blocked": {},
        "observations": encode(variant.observations),
        "notes": list(variant.notes),
    }
    rows: list[str] = []
    fact_rows: list[tuple[str, str, str, Any, str]] = []
    for name, computation in variant.computations.items():
        result = computation.result
        links = []
        for suffix, report in (("premises", result.report), *_reports(result.data)):
            filename = evidence(name, suffix, report)
            links.append(f'<a href="{filename}">{escape(suffix)}</a>')
        encoded["computations"][name] = {
            "method": result.method,
            "kind": computation.kind,
            "settings": encode(result.settings),
            "data": encode(result.data),
            "evidence": result.report.to_dict(),
        }
        rows.append(
            f"<tr><td>{escape(name)}<br><span class=kind>{escape(computation.kind)}</span></td>"
            f"<td>{escape(_summary(result.data))}</td><td>{escape(_counts(result.report))}<br>"
            f"{' · '.join(links)}</td></tr>"
        )
        fact_rows.extend((label, name, path, value, computation.kind) for path, value in _facts(result.data))
    rejected = []
    for name, blocked in variant.blocked.items():
        filename = evidence(name, "blocked", blocked.report)
        encoded["blocked"][name] = {"message": blocked.message, "evidence": blocked.report.to_dict()}
        rejected.append(
            f"<p class=blocked><strong>{escape(name)} blocked.</strong> {escape(blocked.message)} "
            f'<a href="{filename}">Inspect the evidence</a>.</p>'
        )
    notes = "".join(f"<li>{escape(note)}</li>" for note in (*variant.case.changes, *variant.notes))
    inputs = escape(json.dumps(encoded["case"], indent=2, ensure_ascii=False))
    observations = ""
    if variant.observations:
        observations = (
            "<h3>Supplementary checks and observations</h3>"
            f"<pre>{escape(json.dumps(encode(variant.observations), indent=2))}</pre>"
        )
        for name, observation in variant.observations.items():
            kind = observation.get("kind", "reported datum") if isinstance(observation, Mapping) else "reported datum"
            fact_rows.extend((label, name, path, value, kind) for path, value in _facts(observation))
    table = (
        "<table><tr><th>Computation / evidence kind</th><th>Selected results</th><th>Checks and full evidence</th></tr>"
        + "".join(rows)
        + "</table>"
        if rows
        else ""
    )
    html = (
        f"<article><h2>{escape(label)}</h2><ul>{notes}</ul>{table}{''.join(rejected)}{observations}"
        f"<details><summary>Case inputs and assumption changes</summary><pre>{inputs}</pre></details></article>"
    )
    return encoded, html, fact_rows


def write_reports(studies: tuple[Study, ...], directory: Path, *, numerical: bool, compact: bool):
    """Write each model to its own subdirectory and one collection index."""
    directory.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "numerical_references": numerical,
        "compact_selection": compact,
        "selection": [study.key for study in studies],
        "provenance": _provenance(),
        "studies": {},
    }
    cards = []
    used_keys = set()
    for study in studies:
        key = _slug(study.key)
        if key in used_keys:
            raise ValueError(f"Two studies have the same output name: {key}")
        used_keys.add(key)
        output = directory / key
        output.mkdir(parents=True, exist_ok=True)
        contents, facts, variants = [], [], {}
        filenames: set[str] = set()
        for label, variant in study.variants.items():
            data, html, rows = _variant(variant, label, output, filenames)
            variants[label] = data
            contents.append(html)
            facts.extend(rows)
        figures: tuple[str, ...] = ()
        if numerical:
            from ._plots import write_figure

            figures = write_figure(study, output / "figures")
        data = {
            "schema_version": 1,
            "model": key,
            "provenance": payload["provenance"],
            "numerical_references": numerical,
            "compact_selection": compact,
            "title": study.title,
            "description": study.description,
            "sources": list(study.sources),
            "notes": list(study.notes),
            "variants": variants,
            "figures": list(figures),
        }
        payload["studies"][key] = data
        (output / "report.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("variant", "computation", "quantity", "value", "kind"))
            writer.writerows(facts)
        source_links = " · ".join(
            f'<a href="{escape(url, quote=True)}">Source {index}</a>'
            if url.startswith(("https://", "http://"))
            else escape(url)
            for index, url in enumerate(study.sources, 1)
        )
        notes = "".join(f"<li>{escape(note)}</li>" for note in study.notes)
        previews = "".join(
            f'<figure><img src="figures/{name}" alt="{escape(study.title)} scientific comparison">'
            f"<figcaption>{escape(study.description)}</figcaption></figure>"
            for name in figures
            if name.endswith(".svg")
        )
        body = (
            '<p><a href="../report.html">All models</a> · <a href="report.json">JSON audit</a> · '
            '<a href="results.csv">Results CSV</a></p>'
            f"<p>{escape(study.description)}</p><p>{source_links}</p><ul>{notes}</ul>{previews}{''.join(contents)}"
        )
        (output / "report.html").write_text(_page(study.title, body), encoding="utf-8")
        amount = sum(len(variant.computations) for variant in study.variants.values())
        blocked = sum(len(variant.blocked) for variant in study.variants.values())
        cards.append(
            f'<article><h2><a href="{key}/report.html">{escape(study.title)}</a></h2>'
            f"<p>{escape(study.description)}</p><p>{len(study.variants)} branches; "
            f"{amount} completed computations; {blocked} blocked computations.</p></article>"
        )
    (directory / "collection.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    selection = "Representative branches" if compact else "Full branch collection"
    mode = (
        "Exact checks with independent numerical observations and figures"
        if numerical
        else "Symbolic replay using SymPy"
    )
    command = (
        "python -m examples.research_models --output OUTPUT_DIRECTORY"
        + " --models "
        + " ".join(study.key for study in studies)
        + (" --compact" if compact else "")
        + (" --symbolic-only" if not numerical else "")
    )
    body = (
        f"<p>{selection}. {mode}.</p><p>Each sub-example separates mathematical validity, "
        "optional economic restrictions, and method contracts. Exact identities, trusted theorem premises, "
        "numerical comparisons, and blocked computations remain distinct in the evidence.</p>"
        f"{''.join(cards)}<h2>Replay and provenance</h2><pre>{escape(command)}</pre>"
        '<p><a href="collection.json">Complete JSON audit, source fingerprints, versions and solver settings</a></p>'
        f"<p>Source revision: <code>{escape(str(payload['provenance']['git_revision']))}</code>; "
        f"working tree dirty: {payload['provenance']['working_tree_dirty']}.</p>"
        "<p>The JSON files are descriptive exports. Replay uses the Python study; "
        "exported expressions are never deserialized or executed.</p>"
    )
    (directory / "report.html").write_text(_page("Constrained research model collection", body), encoding="utf-8")
    return payload

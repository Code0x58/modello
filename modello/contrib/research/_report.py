"""Offline, escaped audit reports; no expression or Python deserialization."""

from __future__ import annotations

import json
import platform
from hashlib import sha256
from html import escape
from importlib import metadata

from sympy import Basic, Symbol, mathml

from ._records import Evidence, ResearchReport


def _labelled(report: ResearchReport, value):
    if isinstance(value, Basic):
        return value.xreplace({symbol: Symbol(label) for symbol, label in report.labels})
    return value


def _text(report, value):
    return None if value is None else str(_labelled(report, value))


def _evidence(report: ResearchReport, evidence: Evidence):
    claim = evidence.resolved_claim or evidence.requirement.claim
    declared = evidence.requirement.claim
    labels = {str(symbol): label for symbol, label in report.labels}

    def step_text(value):
        for symbol, label in sorted(labels.items(), key=lambda item: -len(item[0])):
            value = value.replace(symbol, label)
        return value

    return {
        "id": evidence.id,
        "property": claim.property,
        "verdict": evidence.verdict,
        "declared_subject": _text(report, declared.subject),
        "declared_coordinate": _text(report, declared.wrt),
        "declared_domain": _text(report, declared.domain),
        "expected_template": _text(report, evidence.expected_template),
        "subject": str(claim.subject) if isinstance(claim.subject, str) else _text(report, claim.subject),
        "expression": _text(report, evidence.expression),
        "coordinate": _text(report, evidence.coordinate),
        "domain": str(claim.domain),
        "order": claim.order,
        "at": _text(report, claim.at),
        "expected": _text(report, claim.expected),
        "target_bound": _text(report, claim.bound),
        "checked_upper_bound": _text(report, evidence.hint),
        "condition": _text(report, evidence.condition),
        "condition_strength": evidence.condition_strength,
        "consumer": evidence.requirement.consumer,
        "reason": evidence.requirement.reason,
        "source": evidence.requirement.source,
        "kind": evidence.kind,
        "steps": [step_text(step) for step in evidence.steps],
        "dependencies": list(evidence.dependencies),
        "witness": [
            {"symbol": _text(report, symbol), "value": _text(report, value)} for symbol, value in evidence.witness
        ],
        "premises": [_evidence(report, item) for item in evidence.premises],
    }


def report_dict(report: ResearchReport):
    payload = {
        "schema_version": 1,
        "model": report.model,
        "assumptions_consistent": report.consistent,
        "assumptions": [_text(report, value) for value in report.assumptions],
        "issues": list(report.issues),
        "checks": [_evidence(report, item) for item in report.checks],
    }
    payload["evidence_sha256"] = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload["environment"] = {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in ("modello", "sympy")},
    }
    return payload


def _math(report, expression):
    try:
        rendered = mathml(_labelled(report, expression), printer="presentation")
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{rendered}</math>'
    except (TypeError, ValueError, NotImplementedError):
        return f"<pre>{escape(str(expression))}</pre>"


def report_html(report: ResearchReport, title: str):
    data = report_dict(report)

    def render(item, level=2):
        source = escape(item["source"])
        source_link = f'<a href="{source}">{source}</a>' if source.startswith(("https://", "http://")) else source
        steps = "".join(f"<li>{escape(step)}</li>" for step in item["steps"])
        witness = "".join(f"<li>{escape(point['symbol'])} = {escape(point['value'])}</li>" for point in item["witness"])
        premises = "".join(render(premise, min(level + 1, 6)) for premise in item["premises"])
        return (
            f"<article><h{level}>{escape(item['id'])}</h{level}>"
            f"<p><strong>{escape(item['verdict'])}</strong> · {escape(item['kind'])}</p>"
            f"<p>{escape(item['reason'])} Used by: <code>{escape(item['consumer'])}</code>.</p>"
            f"<p>Declared subject: <code>{escape(str(item['declared_subject']))}</code>; "
            f"property: {escape(item['property'])}.</p>"
            f"<p>Expression: <code>{escape(item['expression'])}</code></p>"
            f"<p>Coordinate: {escape(str(item['coordinate']))}; domain: {escape(item['domain'])}.</p>"
            f"<p>Condition ({escape(item['condition_strength'])}): <code>{escape(str(item['condition']))}</code></p>"
            f"<p>Expected form: <code>{escape(str(item['expected']))}</code>; "
            f"target bound: <code>{escape(str(item['target_bound']))}</code>; "
            f"checked upper bound: <code>{escape(str(item['checked_upper_bound']))}</code>.</p>"
            f"<p>Expected template: <code>{escape(str(item['expected_template']))}</code>.</p>"
            f"<ol>{steps}</ol><ul>{witness}</ul><p>{source_link}</p>{premises}</article>"
        )

    assumptions = "".join(_math(report, value) for value in report.assumptions) or "<p>No additional assumptions.</p>"
    checks = "".join(render(item) for item in data["checks"])
    issues = "".join(f"<li>{escape(issue)}</li>" for issue in report.issues)
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>"
        "body{max-width:64rem;margin:2rem auto;padding:0 1rem;font:17px/1.55 system-ui;"
        "color:#182a3b;overflow-wrap:anywhere}"
        "article{border-top:1px solid #cbd4db;padding:1rem 0}article article{margin-left:1rem}"
        "code,pre{white-space:pre-wrap;overflow-wrap:anywhere}math{overflow-x:auto}"
        "h1,h2,h3{line-height:1.25}a{color:#075a8c}li{margin:.4rem 0}"
        f"</style><h1>{escape(title)}</h1><p>Model: {escape(report.model)}. "
        f"Assumptions consistent: {report.consistent}.</p><h2>Assumptions</h2>{assumptions}"
        f"<ul>{issues}</ul>{checks}<footer>Evidence fingerprint: "
        f"<code>{data['evidence_sha256']}</code></footer></html>"
    )

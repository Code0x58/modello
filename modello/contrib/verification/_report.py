"""Offline, escaped audit reports; no expression or Python deserialization."""

from __future__ import annotations

import json
import platform
import re
from collections import defaultdict
from hashlib import sha256
from html import escape
from importlib import metadata

from sympy import Basic, Symbol
from sympy.printing.mathml import MathMLPresentationPrinter
from sympy.printing.str import StrPrinter

from ._records import Evidence, VerificationReport


class _TextPrinter(StrPrinter):
    def __init__(self, labels):
        super().__init__()
        self.labels = dict(labels)
        spellings = defaultdict(set)
        for symbol, label in labels:
            spellings[str(symbol)].add(label)
        self._step_labels = {text: next(iter(names)) for text, names in spellings.items() if len(names) == 1}
        self._step_pattern = re.compile(
            r"(?<![\w.])(?:"
            + "|".join(re.escape(text) for text in sorted(self._step_labels, key=len, reverse=True))
            + r")(?![\w.])"
        )

    def text(self, value):
        if value is None:
            return None
        return self.doprint(value) if isinstance(value, Basic) else str(value)

    def step(self, value):
        return (
            self._step_pattern.sub(lambda match: self._step_labels[match.group()], value)
            if self._step_labels
            else value
        )

    def _print(self, expression, **kwargs):
        if isinstance(expression, Symbol) and expression in self.labels:
            return self.labels[expression]
        return super()._print(expression, **kwargs)


class _MathPrinter(MathMLPresentationPrinter):
    def __init__(self, labels):
        super().__init__()
        self.labels = dict(labels)

    def _print(self, expression, **kwargs):
        if isinstance(expression, Symbol) and expression in self.labels:
            expression = Symbol(self.labels[expression])
        return super()._print(expression, **kwargs)


def _evidence(printer: _TextPrinter, evidence: Evidence):
    claim = evidence.resolved or evidence.obligation
    declared = evidence.obligation
    return {
        "id": evidence.id,
        "property": claim.property,
        "verdict": evidence.verdict,
        "declared_subject": printer.text(declared.subject),
        "declared_coordinate": printer.text(declared.wrt),
        "declared_domain": printer.text(declared.domain),
        "expected_template": printer.text(evidence.expected_template),
        "subject": printer.text(claim.subject),
        "expression": printer.text(evidence.expression),
        "coordinate": printer.text(evidence.coordinate),
        "domain": printer.text(claim.domain),
        "order": claim.order,
        "at": printer.text(claim.at),
        "expected": printer.text(claim.expected),
        "target_bound": printer.text(claim.bound),
        "bound_hint": printer.text(evidence.hint),
        "source_operands": [printer.text(value) for value in evidence.source_operands],
        "condition": printer.text(evidence.condition),
        "condition_strength": evidence.condition_strength,
        "consumer": evidence.obligation.consumer,
        "reason": evidence.obligation.reason,
        "source": evidence.obligation.source,
        "kind": evidence.kind,
        "steps": [printer.step(step) for step in evidence.steps],
        "dependencies": list(evidence.dependencies),
        "witness": [
            {"symbol": printer.text(symbol), "value": printer.text(value)} for symbol, value in evidence.witness
        ],
        "premises": [_evidence(printer, item) for item in evidence.premises],
    }


def report_dict(report: VerificationReport):
    printer = _TextPrinter(report.labels)
    payload = {
        "schema_version": 2,
        "model": report.model,
        "assumptions_consistent": report.consistent,
        "assumptions": [printer.text(value) for value in report.assumptions],
        "issues": list(report.issues),
        "evidence": [_evidence(printer, item) for item in report.evidence],
    }
    payload["evidence_sha256"] = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    payload["environment"] = {
        "python": platform.python_version(),
        **{name: metadata.version(name) for name in ("modello", "sympy")},
    }
    return payload


def _math(report, expression):
    try:
        rendered = _MathPrinter(report.labels).doprint(expression)
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">{rendered}</math>'
    except (TypeError, ValueError, NotImplementedError):
        return f"<pre>{escape(_TextPrinter(report.labels).text(expression))}</pre>"


def report_html(report: VerificationReport, title: str):
    data = report_dict(report)

    def render(item, level=2):
        source = escape(item["source"])
        source_link = f'<a href="{source}">{source}</a>' if source.startswith(("https://", "http://")) else source
        steps = "".join(f"<li>{escape(step)}</li>" for step in item["steps"])
        witness = "".join(f"<li>{escape(point['symbol'])} = {escape(point['value'])}</li>" for point in item["witness"])
        premises = "".join(render(premise, min(level + 1, 6)) for premise in item["premises"])
        observations = (
            "<p>Source operands: "
            + "; ".join(f"<code>{escape(value)}</code>" for value in item["source_operands"])
            + ".</p>"
            if item["source_operands"]
            else ""
        )
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
            f"bound hint: <code>{escape(str(item['bound_hint']))}</code>.</p>"
            f"<p>Expected template: <code>{escape(str(item['expected_template']))}</code>.</p>"
            f"{observations}"
            f"<ol>{steps}</ol><ul>{witness}</ul><p>{source_link}</p>{premises}</article>"
        )

    assumptions = "".join(_math(report, value) for value in report.assumptions) or "<p>No additional assumptions.</p>"
    evidence = "".join(render(item) for item in data["evidence"])
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
        f"<ul>{issues}</ul>{evidence}<footer>Evidence fingerprint: "
        f"<code>{data['evidence_sha256']}</code></footer></html>"
    )

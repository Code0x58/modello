"""Reproduce the exponential/logistic study and write its audit artifacts."""

from __future__ import annotations

import csv
import json
import subprocess
from hashlib import sha256
from html import escape
from importlib import metadata
from pathlib import Path

from examples._market_making_models import (
    SOURCE,
    CARAMarketMaker as CARAMarketMaker,
    LogisticIntensity as LogisticIntensity,
    study_case,
)
from modello.contrib.research import ResearchError


def compare(*, capacity=5, points=101):
    """Return cases, checked numerical results, and independent comparisons."""
    import numpy as np

    from examples._market_making_numerics import exponential_reference, run_quotes

    results = {}
    for name, logistic in (("exponential", False), ("logistic", True)):
        case = study_case(logistic=logistic, capacity=capacity)
        result = run_quotes(case, points=points)
        refined = run_quotes(case, points=points, rtol=1e-10, atol=1e-12, max_step=0.005)
        data, tighter = result.data, refined.data
        quote_errors = [
            abs(left - right)
            for side in ("bid", "ask")
            for row, refined_row in zip(getattr(data, side), getattr(tighter, side), strict=True)
            for left, right in zip(row, refined_row, strict=True)
            if left is not None and right is not None
        ]
        convergence = {
            "max_theta_difference": float(np.max(np.abs(np.array(data.theta) - np.array(tighter.theta)))),
            "max_quote_difference": max(quote_errors),
            "reference_rtol": 1e-10,
            "reference_atol": 1e-12,
            "reference_max_step": 0.005,
        }
        if not logistic:
            reference = exponential_reference(case, data.times)
            convergence["matrix_exponential_theta_error"] = float(np.max(np.abs(np.array(data.theta) - reference)))
        parameters = {
            path: str(case.expression(path))
            for path in (
                "horizon",
                "capacity",
                "step",
                "gamma",
                "risk.sigma",
                "terminal.eta",
                "bid.intensity.scale",
                "bid.intensity.decay",
                "ask.intensity.scale",
                "ask.intensity.decay",
            )
        }
        results[name] = {"case": case, "result": result, "parameters": parameters, "convergence": convergence}
    return results


def _provenance():
    import modello.contrib.research as research

    root = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        revision = status = None
    source_files = [
        Path(__file__),
        *Path(research.__file__).parent.glob("*.py"),
        Path(__file__).with_name("_market_making_models.py"),
        Path(__file__).with_name("_market_making_numerics.py"),
        Path(__file__).with_name("_research_checks.py"),
        Path(__file__).with_name("market_making.py"),
    ]
    return {
        "git_revision": revision.stdout.strip() if revision is not None and revision.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout) if status is not None and status.returncode == 0 else None,
        "versions": {name: metadata.version(name) for name in ("modello", "sympy", "numpy", "scipy", "matplotlib")},
        "source_sha256": {
            str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest() for path in sorted(source_files)
        },
    }


def _figures(results, directory):
    import matplotlib

    with matplotlib.rc_context({"svg.hashsalt": "modello-market-making", "font.size": 10}):
        _draw_figures(results, directory)


def _draw_figures(results, directory):
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from scipy.special import expit

    directory.mkdir(parents=True, exist_ok=True)

    def subplots(height, *, sharey=False):
        figure = Figure(figsize=(10, height), layout="constrained")
        FigureCanvasAgg(figure)
        return figure, figure.subplots(1, 2, sharey=sharey)

    def save(figure, name):
        figure.savefig(directory / name, bbox_inches="tight", metadata={"Date": None})
        figure.savefig((directory / name).with_suffix(".png"), bbox_inches="tight", dpi=140)
        figure.clear()

    colors = {"exponential": "#136eaa", "logistic": "#c65b22"}
    distances = np.linspace(-1.5, 3, 301)
    figure, axes = subplots(3.5)
    axes[0].plot(distances, 2 * np.exp(-distances), color=colors["exponential"], label="Exponential")
    axes[0].plot(distances, 4 * expit(-2 * distances), color=colors["logistic"], label="Logistic")
    axes[0].set(xlabel="Quote distance", ylabel="Trade arrival rate", title="Matched rate and slope at zero")
    axes[0].legend()
    axes[1].plot(distances, np.ones_like(distances), color=colors["exponential"], label="Exponential")
    axes[1].plot(distances, 1 - np.exp(-2 * distances), color=colors["logistic"], label="Logistic")
    axes[1].axhline(2, color="#555555", linestyle=":", label="Required uniform bound")
    axes[1].set(xlabel="Quote distance", ylabel="Curvature ratio", title="Both suprema equal 1")
    axes[1].legend()
    save(figure, "intensities.svg")
    figure, axes = subplots(4)
    minimum = min(min(row) for value in results.values() for row in value["result"].data.theta)
    maximum = max(max(row) for value in results.values() for row in value["result"].data.theta)
    for axis, (name, value) in zip(axes, results.items(), strict=True):
        data = value["result"].data
        image = axis.pcolormesh(
            data.times, data.inventory, np.array(data.theta).T, shading="nearest", vmin=minimum, vmax=maximum
        )
        axis.set_xlim(data.times[0], data.times[-1])
        axis.set(xlabel="Time", ylabel="Inventory", title=name.capitalize())
    figure.colorbar(image, ax=axes, label="Continuation value theta")
    save(figure, "values.svg")
    figure, axes = subplots(4, sharey=True)
    for axis, (name, value) in zip(axes, results.items(), strict=True):
        data = value["result"].data
        for index, color in zip(
            (0, len(data.times) // 2, len(data.times) - 1), ("#136eaa", "#c65b22", "#697b36"), strict=True
        ):
            axis.plot(
                data.inventory,
                [np.nan if x is None else x for x in data.bid[index]],
                color=color,
                label=f"Bid, t={data.times[index]:g}",
            )
            axis.plot(
                data.inventory,
                [np.nan if x is None else x for x in data.ask[index]],
                color=color,
                linestyle="--",
                label=f"Ask, t={data.times[index]:g}",
            )
        axis.set(xlabel="Inventory", ylabel="Quote distance", title=name.capitalize())
        axis.legend(fontsize=8)
    save(figure, "quotes.svg")
    exponential, logistic = (results[name]["result"].data for name in ("exponential", "logistic"))
    difference = np.array(logistic.theta) - np.array(exponential.theta)
    quote_difference = np.array(
        [
            [np.nan if a is None or b is None else a - b for a, b in zip(left, right, strict=True)]
            for left, right in zip(logistic.bid, exponential.bid, strict=True)
        ]
    )
    figure, axes = subplots(4)
    for axis, values, label in zip(
        axes, (difference, quote_difference), ("Theta difference", "Bid-distance difference"), strict=True
    ):
        bound = max(float(np.nanmax(np.abs(values))), 1e-12)
        image = axis.pcolormesh(
            logistic.times,
            logistic.inventory,
            np.ma.masked_invalid(values.T),
            shading="nearest",
            cmap="RdBu_r",
            vmin=-bound,
            vmax=bound,
        )
        axis.set_xlim(logistic.times[0], logistic.times[-1])
        axis.set(xlabel="Time", ylabel="Inventory", title=label)
        figure.colorbar(image, ax=axis, label="Logistic minus exponential")
    save(figure, "differences.svg")


def write_study(directory: Path, *, capacity=5, points=101):
    """Write descriptive JSON, HTML, CSV, and offline SVG figures."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        results = compare(capacity=capacity, points=points)
    except ResearchError as error:
        (directory / "report.json").write_text(error.report.to_json())
        (directory / "report.html").write_text(error.report.to_html(title=str(error)))
        raise
    payload = {
        "schema_version": 1,
        "source": SOURCE,
        "provenance": _provenance(),
        "comparison": "Matched rate and slope at zero quote distance",
        "cases": {},
    }
    for name, value in results.items():
        result, data = value["result"], value["result"].data
        payload["cases"][name] = {
            "parameters": value["parameters"],
            "evidence": result.report.to_dict(),
            "settings": dict(result.settings),
            "diagnostics": data.diagnostics,
            "convergence": value["convergence"],
            "times": data.times,
            "inventory": data.inventory,
            "theta": data.theta,
            "bid": data.bid,
            "ask": data.ask,
        }
        (directory / f"evidence-{name}.html").write_text(result.report.to_html(title=f"{name.capitalize()} evidence"))
    with (directory / "theta.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "time", "inventory", "theta"))
        for name, value in results.items():
            data = value["result"].data
            writer.writerows(
                (name, time, q, data.theta[i][j])
                for i, time in enumerate(data.times)
                for j, q in enumerate(data.inventory)
            )
    with (directory / "quotes.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("family", "time", "inventory", "bid_distance", "ask_distance"))
        for name, value in results.items():
            data = value["result"].data
            writer.writerows(
                (name, time, q, data.bid[i][j], data.ask[i][j])
                for i, time in enumerate(data.times)
                for j, q in enumerate(data.inventory)
            )
    _figures(results, directory / "figures")
    (directory / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    rows = ""
    for name, value in results.items():
        convergence = value["convergence"]
        diagnostics = value["result"].data.diagnostics
        rows += (
            f'<tr><td><a href="evidence-{name}.html">{name.capitalize()}</a></td>'
            f"<td>{convergence['max_theta_difference']:.3g}</td>"
            f"<td>{convergence['max_quote_difference']:.3g}</td>"
            f"<td>{diagnostics['time_difference_defects'][1]:.3g}</td>"
            f"<td>{diagnostics['max_root_residual']:.3g}</td></tr>"
        )
    figures = "".join(
        f'<figure><img src="figures/{name}.svg" alt="{label}"><figcaption>{label}</figcaption></figure>'
        for name, label in (
            ("intensities", "The two intensity families and their curvature ratios."),
            ("values", "Time-dependent continuation values on the same scale."),
            ("quotes", "Bid and ask distances at three times; unavailable sides are omitted."),
            ("differences", "Changes caused by substituting logistic intensity."),
        )
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Market-making research study</title><style>"
        "body{max-width:72rem;margin:2rem auto;padding:0 1rem;font:17px/1.6 system-ui;color:#182a3b}"
        "img{width:100%;height:auto}figure{margin:2rem 0}table{width:100%;border-collapse:collapse}"
        "td,th{text-align:left;padding:.6rem;border-bottom:1px solid #cbd4db}code{overflow-wrap:anywhere}"
        "a{color:#075a8c}</style><h1>Market-making research study</h1>"
        f'<p>A single-asset CARA model following <a href="{SOURCE}">Guéant, Model A</a>. '
        "Time is continuous and inventory changes in lots. Both intensity families have the same "
        "arrival rate and slope at zero distance.</p><p>The linked evidence reports retain the assumptions, "
        "checked curvature bounds, theorem premises, and source expressions. Numerical convergence "
        "checks below measure approximation error; they do not prove global mathematical properties.</p>"
        "<h2>Numerical checks</h2><p>Differences compare the default run with tighter tolerances. "
        "HJB defects use independent central time differences. The exponential case also has a "
        "matrix-exponential reference, recorded in the JSON report.</p>"
        "<table><tr><th>Case / evidence</th><th>Theta difference</th><th>Quote difference</th>"
        f"<th>HJB defect</th><th>Quote root residual</th></tr>{rows}</table>"
        f"{figures}<h2>Reproduce and inspect</h2>"
        '<p><a href="report.json">Complete JSON audit</a> · <a href="theta.csv">Values CSV</a> · '
        '<a href="quotes.csv">Quotes CSV</a></p>'
        "<pre>uv run --locked --extra research python -m examples.market_making --study --output OUTPUT_DIRECTORY</pre>"
        f"<p>Source revision: <code>{escape(str(payload['provenance']['git_revision']))}</code>. "
        f"Working tree dirty: {payload['provenance']['working_tree_dirty']}. "
        "Exact source fingerprints and dependency versions are in the JSON audit.</p></html>"
    )
    (directory / "report.html").write_text(html)
    return payload

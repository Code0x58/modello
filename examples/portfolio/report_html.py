"""Self-contained account tables and interactive SVG timelines over typed reports."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from html import escape

from sympy import Symbol

from examples.compute.named_map import NamedMap
from examples.compute.series import Series
from examples.portfolio.performance import Comparison
from examples.portfolio.reporting import AccountReport

COLORS = ("#2563a6", "#db7927", "#7d58a3", "#238578", "#a44554", "#556578")
METHOD = (
    "Opening holdings form an unchanged reference book. Signal sleeves contain signed incremental shares "
    "from explicitly linked fills; fees follow fill allocations, and instrument income follows sleeve shares "
    "at the income event. Unassigned trades and cash income stay unattributed. This is an accounting "
    "decomposition under a declared allocation rule, rather than an estimate of a signal's causal effect."
)


def table(headers: tuple[str, ...], rows: Iterable[tuple[str, ...]], caption: str) -> str:
    head = "".join(f'<th scope="col">{escape(value)}</th>' for value in headers)
    body = "".join(
        "<tr>" + f'<th scope="row">{escape(row[0])}</th>' + "".join(f"<td>{escape(x)}</td>" for x in row[1:]) + "</tr>"
        for row in rows
    )
    return (
        f'<div class="table-scroll"><table><caption>{escape(caption)}</caption>'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def chart(title: str, unit: str, lines: NamedMap[Series[datetime]]) -> str:
    """Use actual elapsed time, separate unit axes, and hoverable observation markers."""
    observations = tuple((at, value) for series in lines.values() for at, value in zip(series.times, series.values))
    if not observations:
        return f'<section class="chart"><h3>{escape(title)}</h3><p>No observations in this window.</p></section>'
    first = min(at.timestamp() for at, _ in observations)
    last = max(at.timestamp() for at, _ in observations)
    low = min(value for _, value in observations)
    high = max(value for _, value in observations)
    padding = (high - low) * 0.12 if high != low else max(abs(high) * 0.05, 0.01)
    low -= padding
    high += padding

    def x(at: datetime) -> float:
        return 75 + 610 * (at.timestamp() - first) / (last - first) if last > first else 380

    def y(value: float) -> float:
        return 205 - 170 * (value - low) / (high - low)

    marks = []
    for i in range(5):
        value = low + (high - low) * i / 4
        at_y = y(value)
        marks.append(
            f'<line x1="75" x2="685" y1="{at_y:.2f}" y2="{at_y:.2f}" class="gridline"/>'
            f'<text x="63" y="{at_y + 4:.2f}" text-anchor="end">{value:.4g}</text>'
        )
    ticks = sorted({at for at, _ in observations})
    for at in dict.fromkeys((ticks[0], ticks[len(ticks) // 2], ticks[-1])):
        label = at.strftime("%d %b %H:%M") if last - first < 86400 else at.strftime("%d %b %Y")
        anchor = "start" if at == ticks[0] else "end" if at == ticks[-1] else "middle"
        marks.append(f'<text x="{x(at):.2f}" y="232" text-anchor="{anchor}">{label}</text>')
    legend = []
    for i, (name, series) in enumerate(lines.items()):
        color = COLORS[i % len(COLORS)]
        points = " ".join(f"{x(at):.2f},{y(value):.2f}" for at, value in zip(series.times, series.values))
        markers = "".join(
            f'<circle cx="{x(at):.2f}" cy="{y(value):.2f}" r="4"><title>'
            f"{escape(name)} · {at.isoformat()} · {value:.7g} {escape(unit)}</title></circle>"
            for at, value in zip(series.times, series.values)
        )
        marks.append(
            f'<g data-line="{i}" stroke="{color}" fill="{color}">'
            f'<polyline points="{points}" fill="none" stroke-width="2.5"/>{markers}</g>'
        )
        legend.append(
            f'<button type="button" data-toggle="{i}" aria-pressed="true" style="--line:{color}">'
            f'<span class="swatch"></span>{escape(name)}</button>'
        )
    return (
        f'<section class="chart"><h3>{escape(title)}</h3><p class="unit">{escape(unit)} · UTC</p>'
        f'<svg viewBox="0 0 720 255" role="img" aria-label="{escape(title, quote=True)}">'
        + "".join(marks)
        + '</svg><div class="legend">'
        + "".join(legend)
        + "</div></section>"
    )


def cumulative(series: Series[datetime], start: datetime) -> Series[datetime]:
    return Series((start, *series.times), (0.0, *series.cumulative_sum().values))


def profit_table(paths: NamedMap[NamedMap[Series[datetime]]], caption: str) -> str:
    return table(
        ("Source", "Holding", "Trading", "Income", "Fees", "Net P&L"),
        (
            (name, *(f"{sum(columns[field].values):,.2f}" for field in ("holding", "trading", "income", "fees", "net")))
            for name, columns in paths.items()
        ),
        caption,
    )


def render_html(report: AccountReport, comparison: Comparison[datetime], *, context: str = "") -> str:
    """Render existing measurements; presentation does not perform accounting or infer attribution."""
    path = report.path
    normalized = path.unit_nav.map(lambda value: 100 * value / path.unit_nav.values[0])
    if comparison.currency != "USD" or comparison.normalized["portfolio"] != normalized:
        raise ValueError("The performance comparison must match the account's USD unit-NAV path")
    performance = comparison.portfolio
    start = path.nav.times[0]
    parts = [
        '<header><p class="eyebrow">Modello / portfolio research</p><h1>Account, signals &amp; utility</h1>',
        f"<p>{escape(path.snapshots[0].account)} · USD · {start.date()} — {path.nav.times[-1].date()}</p>",
        f'<p class="notice">{performance.return_count} returns over {performance.elapsed_days:g} calendar days.</p>',
    ]
    if context:
        parts.append(f'<p class="notice">{escape(context)}</p>')
    parts.append('</header><main><section class="metrics">')
    for label, value in (
        ("Closing NAV", f"USD {path.nav.values[-1]:,.2f}"),
        ("Investment P&L", f"USD {sum(report.account['net'].values):,.2f}"),
        ("Account total return", f"{performance.total_return:.4%}"),
        ("Benchmark return", f"{comparison.benchmark.total_return:.4%}"),
        ("Account minus benchmark", f"{comparison.return_difference * 100:+.4f} pp"),
    ):
        parts.append(f'<div class="metric"><span>{escape(label)}</span><strong>{escape(value)}</strong></div>')
    parts.append("</section><h2>Account overview</h2>")
    parts.append(
        table(
            ("Mark (UTC)", "NAV", "Cash", "Unit NAV", "Interval P&L", "External capital", "Residual"),
            (
                (
                    snapshot.at.strftime("%Y-%m-%d %H:%M"),
                    f"{snapshot.nav:,.2f}",
                    f"{snapshot.cash:,.2f}",
                    f"{snapshot.unit_nav:.4f}",
                    *(
                        f"{report.account[field].values[i - 1]:,.2f}" if i else "—"
                        for field in ("net", "external_flows")
                    ),
                    f"{report.account['residual'].values[i - 1]:.2f}" if i else "—",
                )
                for i, snapshot in enumerate(path.snapshots)
            ),
            "Account marks; cash and P&L in USD. External capital is separated from investment return.",
        )
    )
    parts.append('<div class="charts">')
    parts.append(chart("Account and benchmark", "Rebased total-return index (100 at opening)", comparison.normalized))
    parts.append(chart("Account balances", "USD", NamedMap({"NAV": path.nav, "Cash": path.cash})))
    parts.append("</div>")
    if not performance.annualization_eligible:
        parts.append('<p class="note">CAGR and annualized Sharpe: unavailable below one full calendar year.</p>')
    else:

        def display(value: float | None, spec: str) -> str:
            return "Unavailable" if value is None else format(value, spec)

        parts.append(
            table(
                ("Metric", "Account", "Benchmark"),
                (
                    ("CAGR", display(performance.cagr, ".4%"), display(comparison.benchmark.cagr, ".4%")),
                    (
                        "Annualized Sharpe",
                        display(performance.sharpe_ratio, ".4f"),
                        display(comparison.benchmark.sharpe_ratio, ".4f"),
                    ),
                ),
                "Annualized performance",
            )
        )
    parts.append(
        f'<p class="note">Account index: {escape(comparison.portfolio_name)}. '
        f"Benchmark: {escape(comparison.benchmark_name)}. "
        f"Sharpe reference: {escape(comparison.risk_free_name)}. "
        "Account returns include income and booked fees; benchmark costs follow the supplied index definition.</p>"
    )
    parts.append("<h2>P&amp;L by signal sleeve</h2>")
    parts.append(f'<p class="measure">{escape(METHOD)}</p>')
    parts.append(profit_table(report.sources, "Attribution totals in USD; fees are shown as positive costs."))
    parts.append(
        table(
            ("Order", "Decision (UTC)", "Instrument", "Signal", "Observed value", "Allocation"),
            (
                (
                    order_id,
                    attribution.order.decision_at.strftime("%Y-%m-%d %H:%M"),
                    attribution.order.member,
                    signal,
                    f"{report.signals[signal][attribution.order.member].asof(attribution.order.decision_at):.6g}",
                    f"{weight:.0%}",
                )
                for order_id, attribution in report.attributions.items()
                for signal, weight in attribution.weights.items()
                if weight
            ),
            "Declared order-to-signal lineage; observations available at the decision time.",
        )
    )
    parts.append(
        chart("Cumulative investment P&L", "USD", report.sources.map(lambda columns: cumulative(columns["net"], start)))
    )
    parts.append("<h2>Instruments</h2>")
    parts.append(
        table(
            ("Member", "Shares", "Price", "Value", "Weight", "Holding P&L", "Trading P&L", "Income", "Fees", "Net P&L"),
            (
                (
                    name,
                    f"{instrument.position.quantity.values[-1]:,.4f}",
                    f"{instrument.position.price.values[-1]:,.2f}",
                    f"{instrument.position.value.values[-1]:,.2f}",
                    f"{instrument.position.weight.values[-1]:.2%}",
                    *(
                        f"{sum(instrument.profit[field].values):,.2f}"
                        for field in ("holding", "trading", "income", "fees", "net")
                    ),
                )
                for name, instrument in report.instruments.items()
            ),
            "Closing positions and full-window P&L in USD. Price is USD per share.",
        )
    )
    if report.instruments:
        parts.append('<label class="picker">Instrument <select id="instrument">')
        parts.extend(f'<option value="{i}">{escape(name)}</option>' for i, name in enumerate(report.instruments))
        parts.append("</select></label>")
    for i, (name, instrument) in enumerate(report.instruments.items()):
        parts.append(f'<section class="instrument-panel" data-instrument="{i}"{" hidden" if i else ""}>')
        parts.append(f"<h3>{escape(name)} / attribution</h3>")
        parts.append(profit_table(instrument.sources, "Instrument P&L by source, USD"))
        parts.append('<div class="charts">')
        parts.append(chart(f"{name} / price", "USD per share", NamedMap({name: instrument.position.price})))
        parts.append(
            chart(
                f"{name} / share decomposition",
                "Shares (signed for virtual sleeves)",
                NamedMap([("actual", instrument.position.quantity), *instrument.quantities.items()]),
            )
        )
        parts.append(
            chart(
                f"{name} / cumulative P&L",
                "USD",
                instrument.sources.map(lambda columns: cumulative(columns["net"], start)),
            )
        )
        observations = NamedMap((signal, values[name]) for signal, values in report.signals.items() if name in values)
        parts.append(chart(f"{name} / observed signals", "Signal value", observations))
        parts.append("</div></section>")
    parts.append("<h2>Cash utility &amp; risk aversion</h2>")
    parts.append(
        '<p class="measure">Utility is evaluated at account NAV, including external capital. '
        "These are supplied preferences, rather than preferences fitted to the P&amp;L. "
        "Absolute risk aversion is −U″(W)/U′(W); relative risk aversion is −W·U″(W)/U′(W). "
        "Utility levels across families depend on normalization.</p>"
    )
    parts.append(
        table(
            ("Definition", "U(NAV)", "U′(NAV)", "U″(NAV)", "Absolute (1/USD)", "Relative"),
            (
                (
                    utility.definition.name,
                    *(
                        f"{utility.values[field].values[-1]:.7g}"
                        for field in (
                            "utility",
                            "marginal_utility",
                            "curvature",
                            "absolute_risk_aversion",
                            "relative_risk_aversion",
                        )
                    ),
                )
                for utility in report.utilities.values()
            ),
            "Risk preferences evaluated at closing NAV; relative risk aversion is dimensionless.",
        )
    )
    parts.append('<div class="charts">')
    for field, title, unit in (
        ("absolute_risk_aversion", "Absolute risk aversion", "1 / USD"),
        ("relative_risk_aversion", "Relative risk aversion", "Dimensionless"),
    ):
        parts.append(chart(title, unit, report.utilities.map(lambda utility: utility.values[field])))
    parts.append("</div><details><summary>Inspect the symbolic reporting definitions</summary>")
    rows = []
    for label, plan, fields in (
        ("Instrument", report.instrument_plan, ("holding", "trading", "net")),
        ("Account", report.account_plan, ("net", "residual")),
    ):
        replacements = {
            field.expression: Symbol(".".join(field.path)) for field in plan.walk() if field.role == "input"
        }
        for name in fields:
            rows.append((f"{label} / {name}", str(getattr(plan.source, name).xreplace(replacements))))
    for name, utility in report.utilities.items():
        rows.extend(
            (f"{name} / {field}", str(expression)) for field, expression in utility.definition.expressions.items()
        )
    parts.append(table(("Model output", "Expression"), rows, "Expressions compiled and evaluated by this report"))
    parts.append(
        '</details><p class="note">Timeline lines connect sampled observations; intraperiod changes are not shown. '
        "Toggle a legend to inspect a series, or hover a point for its timestamp and value.</p></main>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Modello — account research report</title><style>"
        + STYLE
        + "</style></head><body>"
        + "".join(parts)
        + "<script>"
        + SCRIPT
        + "</script></body></html>"
    )


def report_data(report: AccountReport) -> dict[str, object]:
    """Serialize report observations and definitions, preserving names and exact time axes."""

    def columns(values: NamedMap[Series[datetime]]) -> dict[str, object]:
        return {
            name: {"times": tuple(at.isoformat() for at in series.times), "values": series.values}
            for name, series in values.items()
        }

    return {
        "attribution_method": METHOD,
        "orders": {
            order_id: {
                "member": attribution.order.member,
                "decision_at": attribution.order.decision_at.isoformat(),
                "signal_weights": dict(attribution.weights),
            }
            for order_id, attribution in report.attributions.items()
        },
        "account": columns(report.account),
        "sources": {name: columns(values) for name, values in report.sources.items()},
        "instruments": {
            name: {
                "profit": columns(instrument.profit),
                "sources": {source: columns(values) for source, values in instrument.sources.items()},
                "quantities": columns(instrument.quantities),
            }
            for name, instrument in report.instruments.items()
        },
        "utilities": {
            name: {
                "definition": str(utility.definition.expression),
                "expressions": {key: str(value) for key, value in utility.definition.expressions.items()},
                "values": columns(utility.values),
            }
            for name, utility in report.utilities.items()
        },
    }


STYLE = """
:root{color-scheme:light;font-family:system-ui,sans-serif;color:#1a2e43;background:#f3f5f7}
*{box-sizing:border-box}body{margin:0}header,main{max-width:1280px;margin:auto;padding:30px}
header{padding-bottom:12px}.eyebrow{font-size:12px;letter-spacing:.15em;text-transform:uppercase;color:#48627b}
h1{font-size:36px;letter-spacing:-.04em;margin:12px 0}h2{font-size:23px;margin:40px 0 14px}h3{font-size:16px}
p{line-height:1.6}.notice{padding:12px 16px;border-left:3px solid #d99238;background:#fff6e5;border-radius:3px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.metric{background:white;
border:1px solid #dce2e8;border-radius:8px;padding:18px}.metric span{font-size:12px;color:#52697d;display:block}
.metric strong{display:block;font-size:23px;margin-top:8px;font-variant-numeric:tabular-nums}
.table-scroll{overflow-x:auto;border:1px solid #dce2e8;border-radius:8px;background:white;margin:16px 0 24px}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}caption{text-align:left;
font-size:12px;color:#52697d;padding:13px 15px;border-bottom:1px solid #e4e9ed}td,th{padding:12px 14px;
text-align:right;white-space:nowrap;border-bottom:1px solid #eef1f4}th:first-child{text-align:left}
thead{background:#edf2f6}
tbody th{font-weight:500}.charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}
.chart{border:1px solid #dce2e8;border-radius:8px;background:#fff;padding:16px;margin:12px 0;min-width:0}
.chart h3{margin:0 0 5px}.chart .unit{font-size:11px;color:#52697d;margin:0}
.chart svg{display:block;width:100%;height:auto}svg text{font-size:12px;fill:#52697d}
.gridline{stroke:#e6ecf0;stroke-width:1}.legend{display:flex;gap:8px;flex-wrap:wrap}
.legend button{font:inherit;font-size:12px;color:#274057;background:#f3f6f9;border:1px solid #dce2e8;border-radius:5px;
padding:6px 9px;cursor:pointer}.legend button[aria-pressed=false]{opacity:.5}.swatch{display:inline-block;width:10px;
height:10px;border-radius:2px;background:var(--line);margin-right:7px}.note{font-size:12px;color:#52697d}
.measure{max-width:1000px;font-size:14px}.picker{display:inline-flex;gap:12px;align-items:center;margin:4px 0}
select{font:inherit;padding:8px 14px;border:1px solid #aab9c6;background:white;border-radius:6px}details{margin:28px 0}
summary{cursor:pointer;font-size:14px;font-weight:600}button:focus-visible,select:focus-visible,summary:focus-visible{
outline:3px solid #418acc;outline-offset:2px}[hidden]{display:none!important}
@media(max-width:760px){header,main{padding:18px}.charts{grid-template-columns:1fr}h1{font-size:29px}}
"""

SCRIPT = """
const picker = document.getElementById('instrument');
if (picker) picker.addEventListener('change', () => {
  document.querySelectorAll('.instrument-panel').forEach(panel => {
    panel.hidden = panel.dataset.instrument !== picker.value;
  });
});
document.querySelectorAll('[data-toggle]').forEach(button => {
  button.addEventListener('click', () => {
    const visible = button.getAttribute('aria-pressed') !== 'true';
    button.setAttribute('aria-pressed', String(visible));
    const series = button.closest('.chart').querySelector('[data-line="' + button.dataset.toggle + '"]');
    series.style.display = visible ? '' : 'none';
  });
});
"""

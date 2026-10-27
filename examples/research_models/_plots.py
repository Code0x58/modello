"""Scientific figures for replayed studies, without importing plotting at import.

The writer samples the bound declarations and uses already computed decisions.
The sampled curves are illustrations of those declarations, not new evidence.
"""

from __future__ import annotations

from math import erf, erfc, isfinite
from pathlib import Path
from typing import Any

from ._study import Study, Variant

_COLORS = ("#2563eb", "#d97706", "#0f766e", "#7c3aed", "#be185d", "#475569")
_SUPPORTED = frozenset({"black-scholes", "participation-execution", "markowitz", "cir-bond", "perpetual-put"})


class _Unresolved(ValueError):
    """A branch has no fully specified curve or decision to display."""


def _number(value: Any) -> float:
    if getattr(value, "free_symbols", ()):
        raise _Unresolved("A plotted quantity still has free parameters")
    try:
        answer = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise _Unresolved("A plotted quantity is not a finite real scalar") from error
    if not isfinite(answer):
        raise _Unresolved("A plotted quantity is not finite")
    return answer


def _scalar(variant: Variant, path: str) -> float:
    return _number(variant.case.expression(path).doit())


def _label(value: str) -> str:
    return value.replace("_", " ").replace("no short", "long only").capitalize()


def _rows(study: Study, preferred: tuple[str, ...]) -> list[tuple[str, Variant]]:
    ordered = [name for name in preferred if name in study.variants]
    ordered.extend(name for name in study.variants if name not in ordered)
    return [(name, study.variants[name]) for name in ordered if study.variants[name].computations]


def _result(variant: Variant, *names: str) -> Any:
    for name in names:
        if name in variant.computations:
            return variant.computations[name].result.data
    raise _Unresolved("This variant has no relevant successful computation")


def _legend(axis: Any, **settings: Any) -> None:
    if axis.get_legend_handles_labels()[0]:
        options = {"frameon": False, "fontsize": 8.5, **settings}
        axis.legend(**options)


class _Plot:
    def __init__(self, figure: Any, numpy: Any, lambdify: Any, derivative: Any) -> None:
        self.figure = figure
        self.axes = tuple(figure.subplots(1, 2))
        self.np, self.lambdify, self.derivative = numpy, lambdify, derivative
        self.functions = {
            name: numpy.vectorize(function, otypes=[float]) for name, function in (("erf", erf), ("erfc", erfc))
        }
        self.used = 0

    def evaluate(self, expression: Any, coordinate: Any, points: Any) -> Any:
        expression = expression.doit()
        if expression.free_symbols - {coordinate}:
            raise _Unresolved("The plotted curve has unresolved free parameters")
        try:
            evaluate = self.lambdify(coordinate, expression, modules=[self.functions, "numpy"])
            with self.np.errstate(all="ignore"):
                values = self.np.asarray(evaluate(points))
            if self.np.iscomplexobj(values):
                if self.np.any(values.imag != 0):
                    raise _Unresolved("The plotted curve is not real on its sample domain")
                values = values.real
            values = self.np.broadcast_to(values, points.shape).astype(float)
        except (TypeError, ValueError, NameError, OverflowError) as error:
            raise _Unresolved("The bound curve cannot be sampled on this domain") from error
        if not self.np.all(self.np.isfinite(values)):
            raise _Unresolved("The plotted curve is not finite on its sample domain")
        return values

    def curve(self, variant: Variant, field: str, coordinate: str, points: Any, *, substitutions=None) -> Any:
        variable = variant.case.expression(coordinate, wrt=coordinate)
        expression = variant.case.expression(field, wrt=variable)
        if substitutions:
            expression = expression.subs(substitutions)
        return self.evaluate(expression, variable, points)

    def finish(self, study: Study) -> None:
        self.figure.suptitle(study.title, fontsize=15, fontweight="bold", x=0.055, ha="left")
        for axis in self.axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color="#e2e8f0", linewidth=0.7)
            axis.set_axisbelow(True)
        if not self.used:
            for axis in self.axes:
                axis.text(
                    0.5,
                    0.5,
                    "No fully specified successful variant\nto plot for this study.",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="#64748b",
                )


def _black_scholes(study: Study, plot: _Plot) -> None:
    rows = []
    for name, variant in _rows(study, ("no_dividends", "constant", "affine", "affine_with_dividends")):
        try:
            spot, strike = (_scalar(variant, f"pricing.{path}") for path in ("spot", "strike"))
            _scalar(variant, "pricing.carry.rate_total")
            _scalar(variant, "pricing.carry.yield_total")
            if min(spot, strike, _scalar(variant, "pricing.carry.variance_total")) <= 0:
                continue
            rows.append((name, variant, spot, strike))
        except _Unresolved:
            continue
    rows = rows[:4]
    left, right = plot.axes
    for axis, title in ((left, "European call"), (right, "European put")):
        axis.set(title=title, xlabel="Spot S", ylabel="Option value")
    if not rows:
        return
    anchors = [value for row in rows for value in row[2:]]
    spots = plot.np.linspace(min(anchors) * 0.35, max(anchors) * 1.65, 401)
    strikes = set()
    ceiling = 0.0
    reference_seen = False
    for index, (name, variant, spot, strike) in enumerate(rows):
        try:
            curves = [plot.curve(variant, f"pricing.{option}", "pricing.spot", spots) for option in ("call", "put")]
            values = [_scalar(variant, f"pricing.{option}") for option in ("call", "put")]
        except _Unresolved:
            continue
        label = f"{_label(name)} · t={_scalar(variant, 'now'):g}"
        for axis, curve, value in zip(plot.axes, curves, values):
            axis.plot(spots, curve, color=_COLORS[index], label=label, linewidth=2)
            axis.scatter([spot], [value], color=_COLORS[index], s=28, zorder=4)
            ceiling = max(ceiling, float(plot.np.max(curve)))
        numerical = variant.computations.get("quadrature")
        if numerical is not None:
            for axis, field in zip(plot.axes, ("quadrature_call", "quadrature_put")):
                axis.scatter(
                    [spot],
                    [_number(getattr(numerical.result.data, field))],
                    marker="x",
                    color="#0f172a",
                    s=55,
                    zorder=5,
                    label="Payoff quadrature" if not reference_seen else None,
                )
            reference_seen = True
        strikes.add(strike)
        plot.used += 1
    for axis in plot.axes:
        for strike in strikes:
            axis.axvline(strike, color="#94a3b8", linewidth=0.8, linestyle="--")
        axis.set_xlim(spots[0], spots[-1])
        if ceiling:
            axis.set_ylim(0, ceiling * 1.08)
        _legend(axis, loc="upper left" if axis is left else "upper right")


def _participation(study: Study, plot: _Plot) -> None:
    preferred = ("capped_deadline", "free", "capped", "deadline", "surcharge_replacement", "risk_neutral_free", "both")
    rows = []
    for name, variant in _rows(study, preferred):
        try:
            data = _result(variant, "analytic", "numerical")
            quantity = _scalar(variant, "quantity")
            for path in ("volume", "eta", "phi", "gamma", "sigma"):
                _scalar(variant, path)
            rate = None if data.rate is None else _number(data.rate)
            duration = None if data.duration is None else _number(data.duration)
            lower = _scalar(variant, "limits.lower")
            upper = _scalar(variant, "limits.cap") if _scalar(variant, "limits.cap_active") == 1 else None
            rows.append((name, variant, data, quantity, rate, duration, lower, upper))
        except _Unresolved:
            continue
    rows = rows[:6]
    left, right = plot.axes
    left.set(
        title="Cost across constant participation rates",
        xlabel="Participation rate ρ",
        ylabel="Execution objective J(ρ)",
    )
    right.set(title="Deterministic inventory at selected rates", xlabel="Time", ylabel="Remaining inventory")
    if not rows:
        return
    anchors = [value for row in rows for value in (row[4], row[6], row[7]) if value is not None and value > 0]
    if not anchors:
        # Q/V has the units of time; a unit-time candidate supplies a finite
        # display scale only. It is not reported as an optimizer.
        anchors = [rows[0][3] / _scalar(rows[0][1], "volume")]
    rates = plot.np.geomspace(min(anchors) / 4, max(anchors) * 2.5, 401)
    objectives, paths, decisions = set(), set(), set()
    unattained = False
    for index, (name, variant, data, quantity, rate, duration, lower, upper) in enumerate(rows):
        try:
            curve = plot.curve(variant, "objective", "rate", rates)
            fingerprint = tuple(plot.np.round(curve, 11))
            color = _COLORS[index % len(_COLORS)]
            if fingerprint not in objectives:
                left.plot(rates, curve, color=color, linewidth=2, label=_label(name))
                objectives.add(fingerprint)
            if rate is not None and duration is not None:
                cost = _number(data.cost)
                decision = (round(rate, 12), round(cost, 12))
                if decision not in decisions:
                    left.scatter([rate], [cost], color=color, s=35, zorder=4)
                    if len(rows) > 1:
                        left.annotate(
                            _label(name),
                            (rate, cost),
                            xytext=(5, 32 if len(_label(name)) > 18 else 6 + index * 4),
                            textcoords="offset points",
                            fontsize=7.5,
                        )
                    decisions.add(decision)
                times = plot.np.linspace(0, duration, 201)
                rate_symbol = variant.case.expression("rate", wrt="rate")
                inventory = plot.curve(variant, "inventory", "time", times, substitutions={rate_symbol: data.rate})
                signature = (round(rate, 12), round(duration, 12), round(quantity, 12))
                if signature not in paths:
                    right.plot(times, inventory, color=color, linewidth=2, label=f"{_label(name)} · ρ={rate:.3g}")
                    paths.add(signature)
                numerical = variant.computations.get("numerical")
                if numerical is not None and numerical.result.data.rate is not None:
                    left.scatter(
                        [_number(numerical.result.data.rate)],
                        [_number(numerical.result.data.cost)],
                        marker="x",
                        s=55,
                        color="#0f172a",
                        zorder=5,
                        label="Numerical minimum" if index == 0 else None,
                    )
            else:
                unattained = True
            if len(rows) == 1:
                if lower > 0:
                    left.axvspan(rates[0], lower, color="#cbd5e1", alpha=0.35)
                if upper is not None:
                    left.axvspan(upper, rates[-1], color="#cbd5e1", alpha=0.35)
                if lower > 0:
                    left.axvline(lower, color="#64748b", linestyle="--", linewidth=1, label="Deadline bound")
                if upper is not None:
                    left.axvline(upper, color="#94a3b8", linestyle=":", linewidth=1.3, label="Participation cap")
            plot.used += 1
        except _Unresolved:
            continue
    left.set_xscale("log")
    left.set_xlim(rates[0], rates[-1])
    if unattained:
        left.text(
            0.02,
            0.02,
            "ρ=0 is excluded; a risk-neutral infimum may be unattained.",
            transform=left.transAxes,
            fontsize=8,
        )
    if not paths:
        right.text(
            0.5,
            0.5,
            "No attained participation rate;\nno finite completion path is selected.",
            transform=right.transAxes,
            ha="center",
            va="center",
            color="#64748b",
        )
    right.axhline(0, color="#94a3b8", linewidth=0.8)
    right.set_ylim(bottom=0)
    _legend(left, loc="upper center")
    _legend(right, loc="upper right")


def _markowitz(study: Study, plot: _Plot) -> None:
    rows = []
    preferred = ("target_and_cap", "target", "cap", "no_short", "shorts_allowed", "flat", "interior", "target_removed")
    for name, variant in _rows(study, preferred):
        try:
            data = _result(variant, "allocation")
            weight, other, variance = map(_number, (data.weight, data.other_weight, data.variance))
            moments = tuple(_scalar(variant, f"moments.{path}") for path in ("first", "second", "cross"))
            rows.append((name, variant, data, weight, other, variance, moments))
        except _Unresolved:
            continue
    rows = rows[:5]
    left, right = plot.axes
    left.set(
        title="Variance along the fully invested budget line", xlabel="Asset 1 weight w", ylabel="Portfolio variance"
    )
    right.set(title="Certified allocation weights", ylabel="Weight")
    if not rows:
        return
    weights = [row[3] for row in rows]
    start, end = min(0.0, *weights) - 0.15, max(1.0, *weights) + 0.15
    samples = plot.np.linspace(start, end, 401)
    families: dict[tuple[float, ...], str] = {}
    plotted = []
    for name, variant, data, weight, other, variance, moments in rows:
        try:
            curve = plot.curve(variant, "variance", "weight", samples)
        except _Unresolved:
            continue
        if moments not in families:
            color = _COLORS[len(families)]
            families[moments] = color
            first, second, cross = moments
            left.plot(samples, curve, color=color, linewidth=2, label=f"v1={first:g}, v2={second:g}, cov={cross:g}")
        color = families[moments]
        left.scatter([weight], [variance], color=color, s=40, edgecolor="white", linewidth=0.7, zorder=4)
        if len(rows) == 1 and hasattr(data.feasible, "start"):
            lower = _number(data.feasible.start) if data.feasible.start.is_finite else start
            upper = _number(data.feasible.end) if data.feasible.end.is_finite else end
            left.axvspan(max(start, lower), min(end, upper), color=color, alpha=0.1, label="Feasible interval")
        plotted.append((name, variant, data, weight, other))
        plot.used += 1
    if not plotted:
        return
    positions = plot.np.arange(len(plotted))
    first = plot.np.array([row[3] for row in plotted])
    second = plot.np.array([row[4] for row in plotted])
    right.bar(positions, first, color=_COLORS[0], width=0.65, label="Asset 1")
    right.bar(
        positions,
        second,
        bottom=plot.np.maximum(first, 0) * (second >= 0),
        color=_COLORS[1],
        width=0.65,
        label="Asset 2",
    )
    active_names = {
        "minimum_return": "return floor",
        "concentration_cap": "cap",
        "first_nonnegative": "w1 ≥ 0",
        "second_nonnegative": "w2 ≥ 0",
    }
    labels = []
    reference_seen = False
    for position, (name, variant, data, weight, other) in enumerate(plotted):
        active = ", ".join(active_names.get(item, item.replace("_", " ")) for item in data.active)
        labels.append(_label(name) + (f"\nactive: {active}" if active else "\ninterior"))
        right.annotate(
            f"{weight:.1%} / {other:.1%}",
            (position, max(weight, other, 1)),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
        if getattr(data.minimizers, "is_Interval", False) or type(data.minimizers).__name__ == "Interval":
            labels[-1] += "\none minimizer shown"
        reference = variant.observations.get("independent_minimum")
        if reference is not None:
            left.scatter(
                [reference["weight"]],
                [reference["variance"]],
                marker="x",
                color="#0f172a",
                s=55,
                zorder=5,
                label="Numerical minimum" if not reference_seen else None,
            )
            reference_seen = True
    right.set_xticks(positions, labels, fontsize=8)
    right.set_xlim(-0.75, len(plotted) - 0.25)
    right.axhline(0, color="#475569", linewidth=0.9)
    right.margins(y=0.2)
    left.set_xlim(start, end)
    left.set_ylim(bottom=0)
    _legend(left, loc="upper center")
    _legend(right, loc="upper right")


def _cir(study: Study, plot: _Plot) -> None:
    rows = []
    preferred = ("feller_satisfied", "feller_not_satisfied", "feller_failed", "zero_initial_rate", "terminal")
    for name, variant in _rows(study, preferred):
        try:
            _result(variant, "price", "verification", "numerical")
            _scalar(variant, "kappa")
            horizon, initial, theta, sigma, margin = (
                _scalar(variant, path) for path in ("horizon", "initial_rate", "theta", "sigma", "feller_margin")
            )
            rows.append((name, variant, horizon, initial, theta, sigma, margin))
        except _Unresolved:
            continue
    rows = rows[:5]
    left, right = plot.axes
    left.set(
        title="Bond value across current short rates", xlabel="Current short rate r", ylabel="Value of unit payment"
    )
    right.set(title="Riccati coefficient B across remaining time", xlabel="Remaining time τ", ylabel="B(τ)")
    if not rows:
        return
    rates = plot.np.linspace(0, max(max(row[3], row[4]) for row in rows) * 3, 301)
    times = plot.np.linspace(0, max(1.0, *(row[2] for row in rows)) * 1.1, 301)
    price_families: dict[tuple[float, ...], str] = {}
    b_families: dict[tuple[float, ...], str] = {}
    lowest_price = 1.0
    reference_seen = False
    for index, (name, variant, horizon, initial, theta, sigma, margin) in enumerate(rows):
        try:
            prices = plot.curve(variant, "curve", "rate", rates)
            coefficients = plot.curve(variant, "b", "time", times)
            current_price = _scalar(variant, "value")
            b_horizon = _scalar(variant, "b_at_horizon")
        except _Unresolved:
            continue
        color = _COLORS[index]
        regime = "Feller holds" if margin >= 0 else "Feller not satisfied"
        label = f"σ={sigma:g}; {regime}" if horizon else "Terminal payment (τ=0)"
        price_key, b_key = tuple(plot.np.round(prices, 12)), tuple(plot.np.round(coefficients, 12))
        if price_key not in price_families:
            left.plot(rates, prices, color=color, linewidth=2, label=label)
            price_families[price_key] = color
        if b_key not in b_families:
            right.plot(times, coefficients, color=color, linewidth=2, label=f"σ={sigma:g}; {regime}")
            b_families[b_key] = color
        left.scatter([initial], [current_price], color=price_families[price_key], s=28, zorder=4)
        lowest_price = min(lowest_price, float(plot.np.min(prices)))
        right.scatter([horizon], [b_horizon], color=b_families[b_key], s=28, zorder=4)
        numerical = variant.computations.get("numerical")
        if numerical is not None:
            for axis, point, value in (
                (left, initial, numerical.result.data.price),
                (right, horizon, numerical.result.data.b),
            ):
                axis.scatter(
                    [point],
                    [_number(value)],
                    marker="x",
                    color="#0f172a",
                    s=55,
                    zorder=5,
                    label="ODE reference" if not reference_seen else None,
                )
            reference_seen = True
        plot.used += 1
    left.text(
        0.03, 0.03, "Both Feller regimes retain bond valuation.", transform=left.transAxes, color="#475569", fontsize=8
    )
    left.set_ylim(bottom=max(0, lowest_price - max(0.03, (1 - lowest_price) * 0.15)), top=1.04)
    right.set_ylim(bottom=0)
    _legend(left, loc="upper right", frameon=True, framealpha=1, facecolor="white", edgecolor="none")
    _legend(right, loc="lower right")


def _perpetual(study: Study, plot: _Plot) -> None:
    rows = []
    for name, variant in _rows(study, ("smooth_fit", "trusted_stopping", "immediate_exercise")):
        try:
            boundary, strike, initial = (_scalar(variant, path) for path in ("boundary", "strike", "initial_spot"))
            _scalar(variant, "beta")
            if min(boundary, strike, initial) <= 0:
                continue
            rows.append((name, variant, boundary, strike, initial))
        except _Unresolved:
            continue
    rows = rows[:4]
    left, right = plot.axes
    left.set(title="Exercise and continuation value", xlabel="Spot S", ylabel="Put value / immediate payoff")
    right.set(title="Smooth fit at the exercise boundary", xlabel="Spot S", ylabel="Delta ∂V/∂S")
    families: set[tuple[str, float, float]] = set()
    reference_seen = False
    for index, (name, variant, boundary, strike, initial) in enumerate(rows):
        try:
            x = variant.case.expression("moneyness", wrt="moneyness")
            value = variant.case.expression("value", wrt=x).doit()
            exercise = variant.case.expression("exercise", wrt=x).doit()
            spots = plot.np.unique(
                plot.np.r_[plot.np.linspace(boundary * 0.2, max(strike * 1.8, initial * 1.3), 400), boundary]
            )
            ratios = spots / boundary
            values = plot.evaluate(value, x, ratios)
            payoff = plot.np.maximum(plot.evaluate(exercise, x, ratios), 0)
            delta = plot.evaluate(plot.derivative(value, x) / boundary, x, ratios)
            signature = (str(value), boundary, strike)
        except _Unresolved:
            continue
        color = _COLORS[index]
        if signature not in families:
            label = "Declared value" if not families else _label(name)
            left.plot(spots, values, color=color, linewidth=2.2, label=label)
            left.plot(
                spots,
                payoff,
                color="#64748b",
                linestyle="--",
                linewidth=1.3,
                label="Immediate payoff" if not families else None,
            )
            right.plot(spots, delta, color=color, linewidth=2.2)
            for axis in plot.axes:
                axis.axvline(boundary, color="#94a3b8", linestyle="--", linewidth=1)
            left.annotate(
                f"B={boundary:.3g}",
                (boundary, strike - boundary),
                xytext=(7, 10),
                textcoords="offset points",
                fontsize=9,
            )
            right.scatter([boundary], [-1], color=color, s=30, zorder=4)
            certificate = None
            for computation in variant.computations.values():
                data = computation.result.data
                if hasattr(data, "one_sided"):
                    certificate = data
                elif hasattr(data, "certificate") and hasattr(data.certificate, "one_sided"):
                    certificate = data.certificate
            if certificate is not None and not families:
                slopes = certificate.one_sided
                text = (
                    f"Delta joins at {_number(slopes['delta_left']):g}.\n"
                    f"Gamma jumps: {_number(slopes['gamma_left']):g} → {_number(slopes['gamma_right']):.3g}."
                )
                right.text(
                    0.04,
                    0.93,
                    text,
                    transform=right.transAxes,
                    va="top",
                    fontsize=9,
                    color="#334155",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.95, "pad": 3},
                )
            families.add(signature)
        current = plot.evaluate(value, x, plot.np.array([initial / boundary]))[0]
        left.scatter([initial], [current], color=color, s=30, zorder=4)
        numerical = variant.computations.get("barrier_search")
        if numerical is not None:
            left.scatter(
                [initial],
                [_number(numerical.result.data.discounted_payoff)],
                marker="x",
                s=55,
                color="#0f172a",
                zorder=5,
                label="Barrier-search value" if not reference_seen else None,
            )
            reference_seen = True
        plot.used += 1
    left.set_ylim(bottom=0)
    right.set_ylim(-1.12, 0.06)
    _legend(left, loc="upper right")


def write_figure(study: Study, directory: Path) -> tuple[str, ...]:
    """Write one two-panel SVG/PNG figure, without touching notebook figures.

    Successful fully specified variants are sampled from their actual bound
    expressions. Unsupported study keys raise ``ValueError`` before optional
    dependencies are loaded or files are created.
    """
    if study.key not in _SUPPORTED:
        raise ValueError(f"Unsupported study key: {study.key!r}")

    import numpy as np
    from matplotlib import rc_context
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sympy import diff, lambdify

    renderers = {
        "black-scholes": _black_scholes,
        "participation-execution": _participation,
        "markowitz": _markowitz,
        "cir-bond": _cir,
        "perpetual-put": _perpetual,
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "font.weight": "normal",
            "text.usetex": False,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlecolor": "#0f172a",
            "axes.labelweight": "normal",
            "axes.labelcolor": "#334155",
            "axes.edgecolor": "#64748b",
            "axes.grid": False,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.transparent": False,
            "svg.fonttype": "none",
            "svg.hashsalt": f"modello-research-collection-{study.key}",
        }
    ):
        figure = Figure(figsize=(11.5, 4.8), layout="constrained")
        FigureCanvasAgg(figure)
        try:
            plot = _Plot(figure, np, lambdify, diff)
            renderers[study.key](study, plot)
            plot.finish(study)
            figure.savefig(directory / "overview.svg", metadata={"Date": None, "Creator": "Modello research studies"})
            figure.savefig(directory / "overview.png", dpi=160, metadata={"Software": "Modello research studies"})
        finally:
            figure.clear()
    return ("overview.svg", "overview.png")

"""Discover coordinate sections and plot responses without guessing dimensions.

Run ``python -m examples.solution_views``. Optional ``--plot-dir plots`` shows
the safety parabola, reachable area, safety paraboloid, and reachable volume
for an ideal projectile with a fixed launch speed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from itertools import combinations
from math import isfinite
from pathlib import Path
from typing import Generic, Iterator, TypeVar, cast

from sympy import (
    ConditionSet,
    Contains,
    Eq,
    Expr,
    FiniteSet,
    Interval,
    Matrix,
    Ne,
    Poly,
    ProductSet,
    S,
    Tuple,
    lambdify,
    sympify,
)
from sympy.core.relational import Relational
from sympy.plotting.series import ImplicitSeries
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from examples.projectile import Projectile
from modello import (
    ModelField,
    ModelInputError,
    Modello,
    ScalarValue,
    SolutionFamily,
    SymbolicScalar,
    UnresolvedSolutionError,
)
from modello._paths import _path_label, _path_value

M = TypeVar("M", bound=Modello)


def _intervals(family: SolutionFamily[M], *, enclosing: bool = False) -> tuple[Interval, ...] | None:
    """Recognise a real box; optionally enclose rather than resolve conditions."""
    domain = family.parameter_domain
    while enclosing and isinstance(domain, ConditionSet):
        domain = domain.base_set
    factors = domain.args if len(family.parameters) > 1 and isinstance(domain, ProductSet) else (domain,)
    intervals = tuple(Interval(-S.Infinity, S.Infinity) if factor is S.Reals else factor for factor in factors)
    if len(intervals) != len(family.parameters) or any(
        not isinstance(factor, Interval) or factor.is_empty is not False for factor in intervals
    ):
        return None
    return intervals


def _finite_real_points(family: SolutionFamily[M]) -> bool:
    domain = family.parameter_domain
    if not isinstance(domain, FiniteSet):
        return False
    for point in domain:
        values = (point,) if len(family.parameters) == 1 else point
        if not isinstance(values, (tuple, Tuple)) or len(values) != len(family.parameters):
            return False
        if any(value.is_real is not True or value.is_finite is not True for value in values):
            return False
    return True


def _polynomials(family: SolutionFamily[M], expressions: tuple[ScalarValue, ...]) -> tuple[Expr, ...] | None:
    """Accept only rational-coefficient polynomials in these public coordinates."""
    parameters = tuple(field.symbol for field in family.parameters)
    outputs = tuple(family.expression(expression) for expression in expressions)
    try:
        if parameters:
            return tuple(Poly(output, *parameters, domain=QQ).as_expr() for output in outputs)
        if all(output.is_Rational for output in outputs):
            return outputs
    except (PolynomialError, CoercionFailed):
        pass
    return None


def _value(model: M, field: ModelField) -> Expr:
    expression = sympify(_path_value(model, field.path))
    if not isinstance(expression, Expr) or any(
        getattr(expression, predicate) is not True for predicate in ("is_number", "is_real", "is_finite")
    ):
        raise ModelInputError("Plot bounds must be finite real numeric model quantities")
    return expression


@dataclass(frozen=True)
class ModelView(Generic[M]):
    """One coordinate section; its domain, outputs, and plot axes are distinct."""

    controls: tuple[ModelField, ...]
    family: SolutionFamily[M]

    @property
    def control_dimension(self) -> int | None:
        """Return the real coordinate-domain dimension for boxes or finite points."""
        intervals = _intervals(self.family)
        if intervals is not None:
            return len(intervals)
        return 0 if _finite_real_points(self.family) else None

    def image_dimension(self, *expressions: ScalarValue) -> int | None:
        """Certify image dimension for polynomials with rational coefficients, or return None."""
        if not expressions:
            raise ModelInputError("Select at least one output expression")
        outputs = _polynomials(self.family, expressions)
        if outputs is None:
            return None
        if _finite_real_points(self.family):
            return 0
        if _intervals(self.family) is None:
            return None
        parameters = tuple(field.symbol for field in self.family.parameters)
        jacobian = Matrix(outputs).jacobian(parameters)
        return int(DomainMatrix.from_Matrix(jacobian).convert_to(QQ.frac_field(*parameters)).rank())

    def _windows(self, lower: M, upper: M) -> tuple[Interval, ...]:
        intervals = _intervals(self.family, enclosing=True)
        if intervals is None or not 1 <= len(intervals) <= 3:
            raise UnresolvedSolutionError(self.family, "Plotting requires one to three real box coordinates.")
        if not isinstance(lower, self.family.problem.model) or not isinstance(upper, self.family.problem.model):
            raise ModelInputError("Plot bounds must use the family's model type")
        windows = []
        for field, interval in zip(self.family.parameters, intervals, strict=True):
            start, end = _value(lower, field), _value(upper, field)
            if (start < end) is not S.true:
                raise ModelInputError("Each lower control bound must be less than its upper bound")
            window = interval.intersect(Interval(start, end))
            if not isinstance(window, Interval) or any(
                endpoint.is_number is not True or endpoint.is_finite is not True
                for endpoint in (window.start, window.end)
            ):
                raise UnresolvedSolutionError(
                    self.family, "Plot bounds must intersect each control in a numeric interval."
                )
            if not all(isfinite(float(value)) for value in (window.start, window.end)) or float(window.start) >= float(
                window.end
            ):
                raise UnresolvedSolutionError(
                    self.family, "The plot window needs distinct finite floating-point bounds."
                )
            windows.append(window)
        return tuple(windows)

    def plot(
        self,
        output: SymbolicScalar,
        *,
        lower: M,
        upper: M,
        destination: Path,
        samples: int = 17,
        title: str | None = None,
    ) -> None:
        """Plot a bounded polynomial curve, surface, or three-control colour sample."""
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D

        intervals = _intervals(self.family)
        outputs = _polynomials(self.family, (output,))
        if intervals is None or outputs is None or not 1 <= len(intervals) <= 3:
            raise UnresolvedSolutionError(
                self.family, "Plotting requires one to three real box controls and a QQ polynomial."
            )
        if type(samples) is not int or samples < 2:
            raise ModelInputError("Plot samples must be an integer of at least two per control")
        grids = []
        for window in self._windows(lower, upper):
            start_float, end_float = float(window.start), float(window.end)
            left, right = int(window.left_open is S.true), int(window.right_open is S.true)
            grid = np.linspace(start_float, end_float, samples + left + right)
            grids.append(grid[left : len(grid) - right])
        parameters = tuple(field.symbol for field in self.family.parameters)
        coordinates = np.meshgrid(*grids, indexing="ij")
        values = np.broadcast_to(
            np.asarray(lambdify(parameters, outputs[0], "numpy")(*coordinates), dtype=float), coordinates[0].shape
        )
        if not np.isfinite(values).all():
            raise UnresolvedSolutionError(self.family, "The response exceeds the numeric range of this plot.")
        output_label = self.family.format(sympify(output))
        labels = (*(_path_label(field.path) for field in self.family.parameters), output_label)
        figure = plt.figure(figsize=(8, 6))
        try:
            axes = figure.add_subplot(projection="3d" if len(grids) > 1 else None)
            if len(grids) == 1:
                axes.plot(coordinates[0], values)
            else:
                axes3d = cast(Axes3D, axes)
                if len(grids) == 2:
                    axes3d.plot_surface(*coordinates, values, cmap="viridis", alpha=0.9)
                    axes3d.contour(*coordinates, values, zdir="z", offset=values.min(), cmap="viridis", linewidths=0.7)
                else:
                    points = axes3d.scatter(
                        coordinates[0].ravel(),
                        coordinates[1].ravel(),
                        coordinates[2].ravel(),
                        c=values.ravel(),
                        cmap="viridis",
                        s=12,
                    )
                    figure.colorbar(points, ax=axes, label=output_label, shrink=0.7, pad=0.15)
                axes3d.set_zlabel(labels[2])
            axes.set(xlabel=labels[0], ylabel=labels[1], title=title or output_label)
            figure.tight_layout()
            destination.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(destination)
        finally:
            plt.close(figure)

    def plot_region(
        self, *, lower: M, upper: M, destination: Path, samples: int = 31, title: str | None = None
    ) -> None:
        """Draw a planar implicit set or a voxel approximation of a 3D inequality region.

        A voxel's centre satisfies the numerical predicate; its whole cell need
        not. Three-dimensional equalities need a surface parameterisation and
        are rejected rather than filtered from a grid.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.collections import PolyCollection

        windows = self._windows(lower, upper)
        parameters = tuple(field.symbol for field in self.family.parameters)
        predicate = self.family.parameter_domain.contains(Tuple(*parameters)) if len(parameters) > 1 else None
        if predicate is None or predicate.free_symbols - set(parameters) or predicate.has(Contains):
            raise UnresolvedSolutionError(
                self.family, "This region needs an explicit predicate in two or three coordinates."
            )
        if type(samples) is not int or samples < 2:
            raise ModelInputError("Plot samples must be an integer of at least two per control")
        labels = tuple(_path_label(field.path) for field in self.family.parameters)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if len(parameters) == 2:
            if predicate is S.true:
                data = ([(windows[0], windows[1])], "fill")
            else:
                data = ImplicitSeries(
                    predicate,
                    *(
                        (parameter, window.start, window.end)
                        for parameter, window in zip(parameters, windows, strict=True)
                    ),
                    depth=2,
                    adaptive=not isinstance(predicate, Relational),
                    n=samples,
                ).get_data()
            figure, axes = plt.subplots(figsize=(8, 6))
            try:
                if len(data) == 4:
                    a, b, residual, kind = data
                    if kind == "contourf":
                        axes.contourf(a, b, residual, levels=[0, np.inf], colors=["#a0d1d8"])
                    # A signed residual's zero level is the constraint boundary.
                    axes.contour(a, b, residual, levels=[0], colors=["#237d91"], linewidths=2)
                else:
                    polygons = [
                        [(a.start, b.start), (a.start, b.end), (a.end, b.end), (a.end, b.start)] for a, b in data[0]
                    ]
                    axes.add_collection(PolyCollection(polygons, facecolors="#a0d1d8", edgecolors="none"))
                axes.set(
                    xlim=(float(windows[0].start), float(windows[0].end)),
                    ylim=(float(windows[1].start), float(windows[1].end)),
                    xlabel=labels[0],
                    ylabel=labels[1],
                    title=title or self.family.problem.name,
                    aspect="equal",
                )
                figure.tight_layout()
                figure.savefig(destination)
            finally:
                plt.close(figure)
            return
        if predicate.has(Eq, Ne):
            raise UnresolvedSolutionError(
                self.family, "Three-dimensional equality sets need a surface parameterisation."
            )
        edges = [np.linspace(float(window.start), float(window.end), samples + 1) for window in windows]
        centres = np.meshgrid(*[(edge[:-1] + edge[1:]) / 2 for edge in edges], indexing="ij")
        occupied = np.broadcast_to(
            np.asarray(lambdify(parameters, predicate, "numpy")(*centres), dtype=bool), centres[0].shape
        )
        if not occupied.any():
            raise UnresolvedSolutionError(
                self.family, "No feasible voxel centres were sampled; this does not prove emptiness."
            )
        figure = plt.figure(figsize=(8, 6))
        try:
            axes = figure.add_subplot(projection="3d")
            axes.voxels(
                *np.meshgrid(*edges, indexing="ij"),
                occupied,
                facecolors="#4398a7",
                edgecolor=(1, 1, 1, 0.12),
                linewidth=0.2,
            )
            axes.set(xlabel=labels[0], ylabel=labels[1], zlabel=labels[2], title=title or self.family.problem.name)
            axes.set_box_aspect(tuple(float(window.measure) for window in windows))
            axes.view_init(elev=24, azim=-55)
            figure.tight_layout()
            figure.savefig(destination)
        finally:
            plt.close(figure)


@dataclass(frozen=True)
class ModelViews(Generic[M]):
    """Enumerate sections through a typed reference, without eagerly solving all."""

    family: SolutionFamily[M]
    through: M

    def sections(self, max_controls: int = 3) -> Iterator[ModelView[M]]:
        """Vary each combination of up to this many current coordinates."""
        if type(max_controls) is not int or max_controls < 1:
            raise ModelInputError("The maximum control count must be a positive integer")
        for count in range(1, min(len(self.family.parameters), max_controls) + 1):
            for controls in combinations(self.family.parameters, count):
                yield ModelView(
                    controls, self.family.section(*(field.symbol for field in controls), through=self.through)
                )


def main() -> None:
    """Discover the boundary of all throws, then the targets they can reach."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path)
    arguments = parser.parse_args()
    family = Projectile.envelope()
    flight = family.symbols
    x, y, target_height = flight.x, flight.y, flight.target_height
    anchor = family.observe(x=0, y=0, target_height=0).one()
    print("Controls | Control dimension | Image dimension (height) | Envelope height")
    for view in ModelViews(family, through=anchor).sections():
        print(
            ", ".join(_path_label(field.path) for field in view.controls),
            view.control_dimension,
            view.image_dimension(flight.height),
            view.family.format(view.family.expression(flight.height)),
            sep=" | ",
        )
    if arguments.plot_dir:
        lower = family.observe(x=-44, y=-44, target_height=0).one()
        upper = family.observe(x=44, y=44, target_height=24).one()
        reachable = family.where(flight.clearance >= 0, target_height >= 0)
        curve = family.section(x, through=anchor).where(flight.height >= 0)
        area = reachable.section(x, target_height, through=anchor)
        surface = family.section(x, y, through=anchor)
        plots: tuple[tuple[str, str, SolutionFamily[Projectile], SymbolicScalar | None], ...] = (
            ("safety-parabola", "Parabola of safety: the highest reachable target", curve, flight.height),
            ("reachable-area", "Reachable targets in a vertical plane", area, None),
            ("safety-paraboloid", "Safety paraboloid: envelope of all launch directions", surface, flight.height),
            ("reachable-volume", "Reachable targets above ground (voxel approximation)", reachable, None),
        )
        gallery = []
        for name, title, selected, response in plots:
            view = ModelView(selected.parameters, selected)
            destination = arguments.plot_dir / f"{name}.png"
            if response is None:
                view.plot_region(
                    lower=lower,
                    upper=upper,
                    destination=destination,
                    samples=401 if len(view.controls) == 2 else 31,
                    title=title,
                )
            else:
                view.plot(response, lower=lower, upper=upper, destination=destination, samples=65, title=title)
            gallery.append(
                f'<figure><img src="{name}.png" alt="{escape(title)}" loading="lazy">'
                f"<figcaption>{escape(title)}</figcaption></figure>"
            )
        (arguments.plot_dir / "index.html").write_text(
            f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>One projectile model, four views</title>
<style>body{{font:16px system-ui;margin:2rem}}figure{{margin:0}}img{{width:100%}}
main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr))}}</style>
<h1>One projectile model, four views</h1>
<p><code>height = {escape(family.format(family.expression(flight.height)))}</code></p>
<p>A ball leaves ground level at 20 m/s under uniform gravity of 10 m/s², without air resistance.
SymPy differentiates the motion model to find the highest reachable target at each
horizontal position. This envelope is the parabola of safety in a vertical plane and
a paraboloid in space; the region between it and the ground contains every reachable target.</p>
<p>The envelope combines all launch directions; it is not one ball's trajectory.
All coordinates are in metres. The surface continues below ground outside the reachable region.
The implicit plots and occupied voxels are numerical illustrations, not exact set boundaries.</p>
<main>{"".join(gallery)}</main></html>""",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

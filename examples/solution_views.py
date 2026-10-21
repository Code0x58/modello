"""Discover coordinate sections and plot responses without guessing dimensions.

Run ``python -m examples.solution_views``. Optional ``--plot-dir plots`` exports
curves, surfaces, and three-control samples with Matplotlib and NumPy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Generic, Iterator, TypeVar

from sympy import (
    Expr,
    FiniteSet,
    Interval,
    Matrix,
    Poly,
    ProductSet,
    Rational,
    S,
    Tuple,
    lambdify,
    sympify,
)
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed, PolynomialError

from modello import (
    ModelField,
    ModelInputError,
    Modello,
    ScalarValue,
    SolutionFamily,
    SymbolicScalar,
    UnresolvedSolutionError,
    inspect_model,
)

M = TypeVar("M", bound=Modello)


def _intervals(family: SolutionFamily[M]) -> tuple[Interval, ...] | None:
    """Recognise a nonempty real Cartesian box, without resolving conditions."""
    domain = family.parameter_domain
    if len(family.parameters) == 1:
        factors = (domain,)
    elif isinstance(domain, ProductSet) and len(domain.args) == len(family.parameters):
        factors = domain.args
    else:
        return None
    intervals = tuple(Interval(-S.Infinity, S.Infinity) if factor is S.Reals else factor for factor in factors)
    if not intervals or any(not isinstance(factor, Interval) or factor.is_empty is not False for factor in intervals):
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
    value: object = model
    for name in field.path:
        value = getattr(value, name)
    expression = sympify(value)
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
        """Certify real polynomial image dimension, rather than rank at an anchor.

        A polynomial map with rational coefficients on a nonempty real interval
        box has image dimension equal to its Jacobian's rational-function rank.
        Conditional domains and other expressions retain an unknown result.
        """
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

    def plot(
        self,
        output: SymbolicScalar,
        *,
        lower: M,
        upper: M,
        destination: Path,
        samples: int = 17,
    ) -> None:
        """Sample a polynomial response inside a bounded real coordinate box.

        One or two controls produce a curve or response surface. With three,
        the control axes carry samples coloured by the output: a four-dimensional
        response graph, not a claim that the output fills a volume.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        intervals = _intervals(self.family)
        outputs = _polynomials(self.family, (output,))
        if intervals is None or outputs is None or not 1 <= len(intervals) <= 3:
            raise UnresolvedSolutionError(
                self.family, "Plotting requires one to three real box controls and a QQ polynomial."
            )
        if type(samples) is not int or samples < 2:
            raise ModelInputError("Plot samples must be an integer of at least two per control")
        if not isinstance(lower, self.family.problem.model) or not isinstance(upper, self.family.problem.model):
            raise ModelInputError("Plot bounds must use the family's model type")
        grids = []
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
            start_float, end_float = float(window.start), float(window.end)
            if not np.isfinite((start_float, end_float)).all() or start_float >= end_float:
                raise UnresolvedSolutionError(
                    self.family, "The plot window needs distinct finite floating-point bounds."
                )
            left, right = int(window.left_open is S.true), int(window.right_open is S.true)
            grid = np.linspace(start_float, end_float, samples + left + right)
            grids.append(grid[left : len(grid) - right])
        parameters = tuple(field.symbol for field in self.family.parameters)
        coordinates = np.meshgrid(*grids, indexing="ij") if len(grids) > 1 else [grids[0]]
        values = np.broadcast_to(
            np.asarray(lambdify(parameters, outputs[0], "numpy")(*coordinates), dtype=float), coordinates[0].shape
        )
        if not np.isfinite(values).all():
            raise UnresolvedSolutionError(self.family, "The response exceeds the numeric range of this plot.")
        labels = tuple(".".join(field.path) for field in self.family.parameters)
        output_label = self.family.format(sympify(output))
        figure = plt.figure(figsize=(8, 6))
        try:
            axes = figure.add_subplot(projection="3d") if len(grids) > 1 else figure.add_subplot()
            if len(grids) == 1:
                axes.plot(coordinates[0], values)
                axes.set_ylabel(output_label)
            elif len(grids) == 2:
                axes.plot_surface(*coordinates, values, cmap="viridis")
                axes.set_ylabel(labels[1])
                axes.set_zlabel(output_label)
            else:
                points = axes.scatter(
                    *(coordinate.ravel() for coordinate in coordinates), c=values.ravel(), cmap="viridis", s=12
                )
                axes.set_ylabel(labels[1])
                axes.set_zlabel(labels[2])
                figure.colorbar(points, ax=axes, label=output_label, shrink=0.7, pad=0.15)
            axes.set_xlabel(labels[0])
            axes.set_title(f"{output_label}: {'three-control samples' if len(grids) == 3 else 'coordinate section'}")
            figure.tight_layout()
            destination.parent.mkdir(parents=True, exist_ok=True)
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
    """Inspect seven execution sections; optionally export nonconstant responses."""
    from examples.solution_families import Execution

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-dir", type=Path)
    arguments = parser.parse_args()
    problem = inspect_model(Execution, "execution views", reference=100)
    execution = problem.symbols
    family = problem.solutions(execution.quantity, execution.impact, execution.fee)
    anchor = Execution("anchor", quantity=5, reference=100, impact=Rational(1, 2), fee=1)
    lower = Execution("lower", quantity=1, reference=100, impact=0, fee=0)
    upper = Execution("upper", quantity=10, reference=100, impact=1, fee=2)
    gallery = []
    print("Controls | Control dimension | Image dimension (price, cost) | Cost")
    for view in ModelViews(family, through=anchor).sections():
        names = ", ".join(".".join(field.path) for field in view.controls)
        print(
            names,
            view.control_dimension,
            view.image_dimension(execution.price, execution.cost),
            view.family.format(view.family.expression(execution.cost)),
            sep=" | ",
        )
        if arguments.plot_dir:
            for name, output in (("price", execution.price), ("cost", execution.cost)):
                if view.image_dimension(output) == 0:
                    continue
                filename = "-".join((*["-".join(field.path) for field in view.controls], name)) + ".png"
                view.plot(output, lower=lower, upper=upper, destination=arguments.plot_dir / filename)
                title = escape(f"{name} · varying {names}")
                gallery.append(
                    f'<figure><img src="{escape(filename, quote=True)}" alt="{title}" loading="lazy">'
                    f"<figcaption>{title}</figcaption></figure>"
                )
    if arguments.plot_dir:
        (arguments.plot_dir / "index.html").write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width"><title>Execution model views</title>'
            "<style>body{font:16px system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#243047}"
            ".gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:1rem}"
            "figure{margin:0;border:1px solid #dce3ed;border-radius:8px;padding:1rem}"
            "img{width:100%;height:auto}figcaption{font-weight:600}</style>"
            "<h1>Execution model views</h1><p>Vary selected controls around one fixed reference. "
            "Reference: quantity 5, impact ½, fee 1, reference price 100. "
            "Windows: quantity [1, 10], impact [0, 1], fee [0, 2]. "
            "Plots sample bounded windows of the feasible sections. Three-control plots use colour "
            'for the output; they do not show a four-dimensional graph directly.</p><div class="gallery">'
            + "".join(gallery)
            + "</div></html>",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()

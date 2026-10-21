# mypy: warn-unused-ignores, disallow-untyped-defs
"""Public typing checks, including errors that mypy must continue to reject."""

from collections import UserDict
from decimal import Decimal
from fractions import Fraction

from sympy import Symbol

from modello import InstanceDummy, Modello, ModelProblem, ModelSolveError, ScalarValue, component, inspect_model
from modello.diagnostics import ComponentView, FieldValue, ModelField, ProblemAnalysis, SolutionSet


class Point(Modello):
    """A child whose attributes and methods must have instance types."""

    x = InstanceDummy("x")
    y = InstanceDummy("y")

    def label(self) -> str:
        """Return a label through a typed instance method."""
        return "point"


class Line(Modello):
    """Two typed components of the same class."""

    start = component(Point)
    end = component(Point)


class Box(Modello):
    """An unrelated model for the incompatible-attribute check."""


class Drawing(Modello):
    """Exercise inferred types through more than one component level."""

    line = component(Line)
    extent = line.end.x - line.start.x


def check_inputs() -> None:
    """Keep model instances and scalar inputs valid, and reject containers."""
    point = Point("P", x=1, y=Fraction(1, 2))
    line = Line("L", start=point, end=Point("E", x=Decimal("2"), y=Symbol("y")))
    drawing = Drawing("D", line=line)
    typed_point: Point = drawing.line.start
    typed_line: Line = drawing.line
    label: str = drawing.line.start.label()
    assert label == "point"
    scalar: ScalarValue = "x + 1"
    Point("P", x=scalar)
    Line("L", start=typed_point, end=typed_line.end)

    # These ignores are assertions: warn-unused-ignores fails if an input starts
    # being accepted, for example through an accidental Any in the input union.
    Line("L", start={"x": 1})  # type: ignore[arg-type]
    Line("L", start={})  # type: ignore[arg-type]
    Line("L", start=UserDict(x=1))  # type: ignore[arg-type]
    Line("L", start={"x": {"nested": 1}})  # type: ignore[arg-type]
    Drawing("D", line={"start": {"x": 1}})  # type: ignore[arg-type]
    Point("P", x=[])  # type: ignore[arg-type]
    Point("P", x=None)  # type: ignore[arg-type]
    drawing.line.start = Box("B")  # type: ignore[assignment]
    _point_class: type[Point] = drawing.line.start  # type: ignore[assignment]


def check_diagnostics() -> None:
    """Debugging uses typed records and the same typed constructor boundary."""
    problem: ModelProblem = inspect_model(Drawing, "case", line=Line("line"))
    field: ModelField = problem.field("line", "start", "x")
    view: ComponentView = problem.at("line")
    report: ProblemAnalysis = problem.analyse(max_checks=1)
    sets: tuple[SolutionSet, ...] = report.solution_sets
    FieldValue(field, 3)
    assert field in view.fields
    assert report.problem is problem
    assert all(result.field in problem.fields for result in sets)
    try:
        Line("case", start=Point("start", x=1))
    except ModelSolveError as error:
        failure: ModelProblem = error.problem
        assert failure.model is Line
    inspect_model(Line, "case", start={"x": 1})  # type: ignore[arg-type]
    inspect_model(Line("instance"), "case")  # type: ignore[arg-type]
    FieldValue(field, {})  # type: ignore[arg-type]
    problem.name = "changed"  # type: ignore[misc]

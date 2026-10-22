# mypy: warn-unused-ignores, disallow-untyped-defs
"""Public typing checks, including errors that mypy must continue to reject."""

from collections import UserDict
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from sympy import Symbol

from modello import (
    Annotation,
    ConstraintKind,
    InstanceDummy,
    Modello,
    ModelProblem,
    ModelSolveError,
    ScalarValue,
    SolutionCountError,
    SolutionFamily,
    SolverBackendError,
    SymbolicCondition,
    annotate,
    annotate_model,
    component,
    inspect_model,
)
from modello.diagnostics import ComponentView, FieldValue, ModelField, ProblemAnalysis, SolutionSet
from modello.gui import FieldInfo, ModelInfo, Scenario, gui


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


@dataclass(frozen=True)
class Description(Annotation):
    text: str


@ModelInfo(title="Annotated drawing")
class AnnotatedDrawing(Modello):
    """Extensions must preserve declared component and decorator types."""

    start = gui(component(Point), title="Start")
    end = annotate(component(Point), Description("End"))
    extent = gui(end.x - start.x, title="Extent")


def check_extensions() -> None:
    """Metadata lookup is typed without weakening model attribute checks."""
    drawing = AnnotatedDrawing("case", start=Point("start", x=2), end=Point("end", x=4))
    point: Point = drawing.start
    label: str = drawing.start.label()
    assert point.label() == label
    info: FieldInfo = FieldInfo.for_model(AnnotatedDrawing)["start"]
    description: Description = Description.for_model(AnnotatedDrawing)["end"]
    annotated: type[AnnotatedDrawing] = annotate_model(AnnotatedDrawing, Description("Drawing"))
    assert info.title == "Start" and description.text == "End" and annotated is AnnotatedDrawing
    drawing.start = Box("B")  # type: ignore[assignment]
    _point_class: type[Point] = drawing.start  # type: ignore[assignment]
    _invalid_label: int = drawing.start.label()  # type: ignore[assignment]
    _invalid_note: int = Description.for_model(AnnotatedDrawing)["end"].text  # type: ignore[assignment]
    annotate_model(object, Description("Invalid"))  # type: ignore[type-var]
    gui(component(Point), "Start")  # type: ignore[call-arg]
    FieldInfo("Start")  # type: ignore[call-arg]
    ModelInfo("Drawing")  # type: ignore[call-arg]
    Scenario("A question", "A description")  # type: ignore[call-arg]


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
    kind: ConstraintKind = problem.constraints[0].kind
    assert kind is ConstraintKind.DEFINITION
    _invalid_kind: ConstraintKind = "input"  # type: ignore[assignment]
    try:
        Line("case", start=Point("start", x=1))
    except ModelSolveError as error:
        failure: ModelProblem = error.problem
        assert failure.model is Line
        if isinstance(error, SolutionCountError):
            count: int = error.solution_count
            returned: int = error.returned_count
            assert count >= 0 and returned >= 0
        elif isinstance(error, SolverBackendError):
            cause: Exception = error.backend_error
            assert cause is error.__cause__
    inspect_model(Line, "case", start={"x": 1})  # type: ignore[arg-type]
    inspect_model(Line("instance"), "case")  # type: ignore[arg-type]
    FieldValue(field, {})  # type: ignore[arg-type]
    problem.name = "changed"  # type: ignore[misc]


def check_solution_families() -> None:
    """Family selection retains the model type, including nested components."""
    problem: ModelProblem[Drawing] = inspect_model(Drawing, "family")
    family: SolutionFamily[Drawing] = problem.solutions()
    _coordinates: tuple[ModelField, ...] = family.parameters
    _cases: tuple[Drawing, ...] = family.cases()
    case: Drawing = family.one()
    point: Point = case.line.start
    assert point.label() == "point"
    symbols: Drawing = problem.symbols
    start: Point = family.symbols.line.start
    problem.solutions(start.x)
    family.expression(symbols.extent)
    family.values(start.x, symbols.extent)
    condition: SymbolicCondition = symbols.extent > 0
    family.where(condition, True)
    family.observe(line=Line("line", start=Point("start", x=2)))
    _section: SolutionFamily[Drawing] = family.section(start.x, through=case)
    family.observe(line={"start": {"x": 2}})  # type: ignore[arg-type]
    family.observe(extent=[])  # type: ignore[arg-type]
    problem.solutions("line.start.x")  # type: ignore[arg-type]
    family.values({"x": 2})  # type: ignore[arg-type]
    family.expression({"x": 2})  # type: ignore[arg-type]
    family.where({"extent": 2})  # type: ignore[arg-type]
    family.where([True])  # type: ignore[arg-type]
    family.where("extent > 0")  # type: ignore[arg-type]
    family.where(1)  # type: ignore[arg-type]
    family.section("line.start.x", through=case)  # type: ignore[arg-type]
    family.section(start.x, through=Line("line"))  # type: ignore[arg-type]
    family.section(start.x, through={})  # type: ignore[arg-type]
    _wrong_symbols: Line = problem.symbols  # type: ignore[assignment]
    _wrong_component: Box = family.symbols.line.start  # type: ignore[assignment]
    _unknown = symbols.line.start.missing  # type: ignore[attr-defined]
    _wrong: Line = family.one()  # type: ignore[assignment]
    _wrong_section: SolutionFamily[Line] = family.section(start.x, through=case)  # type: ignore[assignment]

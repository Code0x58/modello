"""Optional local exploration of nested models (install ``modello[gui]``).

Presentation metadata never changes the algebra. Raw snapshots do not solve;
browser edits are exact scalars parsed without evaluating expression strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field, replace
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence, TypeVar

from sympy import Expr, S, Symbol, factor_terms

from modello import Annotation, Modello, ScalarValue, _solve_problem, annotate, annotate_model, inspect_model
from modello._gui.formatting import display_number
from modello._gui.inputs import parse_scalar
from modello._gui.mathml import EquationPrinter
from modello.diagnostics import Candidate, ConstraintKind, FieldValue, ModelConstraint, ModelField, ModelProblem
from modello.errors import ModelInputError, ModelValidationError, SolverBackendError
from modello.solutions import SolutionFamily

if TYPE_CHECKING:
    from flask import Flask

    from modello._gui.families import FamilyView

__all__ = ["BindingError", "Explorer", "FieldInfo", "ModelInfo", "PlotInfo", "Scenario", "create_app", "gui"]

_ValueT = TypeVar("_ValueT")
_ModelT = TypeVar("_ModelT", bound=Modello)


@dataclass(frozen=True, kw_only=True)
class FieldInfo(Annotation):
    """Display metadata for a scalar or component, without changing its algebra.

    All metadata arguments are keyword-only.
    ``symbol`` is literal MathML text, not LaTeX. ``hidden`` removes the editor
    and observation value while retaining its inputs. Calculated outputs can
    still reveal input values; hiding is a presentation choice. ``unit`` is a
    display label; ``number_format='percent'`` renders fractions as percentages.
    """

    title: str = ""
    description: str = ""
    info: str = ""
    symbol: str = ""
    hidden: bool = False
    unit: str = ""
    number_format: Literal["number", "percent"] = "number"


def gui(
    field: _ValueT,
    *,
    title: str = "",
    description: str = "",
    info: str = "",
    symbol: str = "",
    hidden: bool = False,
    unit: str = "",
    number_format: Literal["number", "percent"] = "number",
) -> _ValueT:
    """Annotate a scalar or component declaration without changing its type.

    Wrap the entire assigned value: ``field = gui(expression, title=...)``
    inside a Modello class body. The class namespace unwraps the declaration
    before later expressions use it; this helper is not an arithmetic wrapper
    outside that scope. Metadata is stored under
    ``Model.__modello_annotations__[FieldInfo]``.
    """
    return annotate(
        field,
        FieldInfo(
            title=title,
            description=description,
            info=info,
            symbol=symbol,
            hidden=hidden,
            unit=unit,
            number_format=number_format,
        ),
    )


@dataclass(frozen=True, kw_only=True)
class PlotInfo:
    """An initial chart of one or two controls, with display windows in model units.

    Windows do not constrain the family. ``view`` selects a flat map or a
    rotatable surface for two controls; one control always gives a curve.
    The browser checks the samples when this exploration is opened.
    """

    controls: tuple[str, ...]
    output: str
    windows: Mapping[str, tuple[ScalarValue, ScalarValue]]
    view: Literal["map", "surface"] = "map"


@dataclass(frozen=True, kw_only=True)
class Scenario:
    """A named exploration that restores initial inputs, then applies bindings.

    All arguments are named. Only visible, editable quantities may be bound.
    ``select`` opens a field in the inspector; it need not be the quantity
    being observed.
    """

    title: str
    description: str
    bindings: Mapping[str, ScalarValue | None] = dataclass_field(default_factory=dict)
    bounds: Mapping[str, tuple[ScalarValue | None, ScalarValue | None]] = dataclass_field(default_factory=dict)
    select: str = ""
    plot: PlotInfo | None = None


@dataclass(frozen=True, kw_only=True)
class ModelInfo(Annotation):
    """Decorate a model with its text and exploration hints.

    Use ``@ModelInfo(title=..., ...)`` above the class. Declare field metadata
    inline with ``gui``. Optional ``fields`` keys are relative dotted paths for
    presentation overrides, including labels on components from other models.
    ``highlights`` selects result cards, ``initial_field`` opens an inspector,
    and ``scenarios`` supplies named explorations from the initial inputs.
    All metadata arguments are keyword-only.
    """

    title: str = ""
    description: str = ""
    info: str = ""
    fields: Mapping[str, FieldInfo] = dataclass_field(default_factory=dict)
    highlights: tuple[str, ...] = ()
    initial_field: str = ""
    scenarios: tuple[Scenario, ...] = ()
    plot: PlotInfo | None = None

    def __call__(self, model: type[_ModelT]) -> type[_ModelT]:
        """Keep the decorated class intact and register its model metadata."""
        return annotate_model(model, self)


@dataclass(frozen=True)
class _Snapshot:
    problem: ModelProblem
    printer: EquationPrinter
    supplied: Mapping[Symbol, Expr]
    initial_inputs: Mapping[Symbol, Expr]
    definitions: Mapping[Symbol, Expr]
    constraints: Mapping[Symbol, list[dict[str, str]]]


class BindingError(ModelValidationError):
    """A rejected edit, with field paths that the interface can highlight."""

    def __init__(self, message: str, paths: tuple[str, ...] = ()) -> None:
        self.paths = paths
        super().__init__(message)


def _instance_problem(instance: Modello, name: str) -> ModelProblem:
    """Reopen original inputs; solved outputs must not become observations."""
    model = type(instance)
    problem = inspect_model(model, name)
    bindings = {
        instance._modello_bound_fields[d].symbol: problem.field(*path).symbol
        for path, d in model._modello_fields.items()
    }
    constraints = list(problem.constraints)
    seen: set[Symbol] = set()
    for path, dummy in model._modello_fields.items():
        value: Any = instance
        for part in path:
            value = getattr(value, part)
        source = instance._modello_bound_fields[dummy]
        field = problem.field(*path)
        if source.copied_kind(value) is ConstraintKind.INPUT and field.symbol not in seen:
            constraints.append(
                ModelConstraint(
                    len(constraints),
                    path,
                    ConstraintKind.INPUT,
                    field.symbol,
                    FieldValue(field, value).value.xreplace(bindings),
                    ".".join(path),
                )
            )
            seen.add(field.symbol)
    return replace(problem, constraints=tuple(constraints))


def _preview(problem: ModelProblem) -> tuple[dict[Symbol, Expr], str]:
    """Audit the shared solver's proposals without selecting an arbitrary branch."""
    try:
        proposals = _solve_problem(problem, check=False)
    except SolverBackendError as error:
        raise BindingError(f"No preview: the symbolic solver raised {type(error.backend_error).__name__}.") from error
    accepted = []
    rejected_paths: set[str] = set()
    for values in proposals:
        audit = problem.check(Candidate(tuple(FieldValue(f, values[f.symbol]) for f in problem.fields)))
        if audit.verdict != "violated":
            accepted.append((values, audit.verdict))
            continue
        rejected_paths.update(
            ".".join(check.value.field.path)
            for check in audit.assumptions
            if check.actual not in (None, check.expected)
        )
        rejected_paths.update(
            ".".join(check.constraint.path) for check in audit.residuals if check.verdict == "violated"
        )
    if len(accepted) != 1:
        message = (
            "Multiple solutions remain. Add an observation to select a unique preview."
            if accepted
            else "The model could not validate this case. Check the highlighted values and assumptions."
        )
        raise BindingError(message, tuple(sorted(rejected_paths)))
    return accepted[0]


class Explorer:
    """Explore a class, instance, recorded problem, or solution family.

    Declare model text with ``@ModelInfo(...)`` and field labels with ``gui``.
    Component declarations are inherited by the layout; ``metadata`` overrides
    field or component labels by full dotted path. ``inputs`` supplies initial
    bindings by dotted path. The original model instance is never mutated.
    """

    def __init__(
        self,
        model: type[Modello] | Modello | ModelProblem | SolutionFamily,
        name: str = "exploration",
        *,
        inputs: Mapping[str, ScalarValue] | None = None,
        metadata: Mapping[str, FieldInfo] | None = None,
    ) -> None:
        from modello._gui.presentation import Layout

        self._source_family = model if isinstance(model, SolutionFamily) else None
        if isinstance(model, SolutionFamily):
            self.problem = model.problem
        elif isinstance(model, ModelProblem):
            self.problem = model
        elif isinstance(model, Modello):
            self.problem = _instance_problem(model, name)
        else:
            self.problem = inspect_model(model, name)
        self._layout = Layout(self.problem, metadata or {})
        self._initial = self._bind(self.problem, inputs or {}, context="inputs")
        self._initial_inputs = {c.left: c.right for c in self._initial.constraints if c.kind is ConstraintKind.INPUT}
        for scenario in self._layout.model_info.scenarios:
            try:
                self._ranges(scenario.bounds)
            except (ModelInputError, ModelValidationError) as error:
                raise ModelInputError(f"Scenario {scenario.title!r}.bounds: {error}") from error
        for plot in (self._layout.model_info.plot, *(s.plot for s in self._layout.model_info.scenarios)):
            if plot is not None:
                # Validate numeric windows at authoring time, without reducing or solving the model.
                for path, endpoints in plot.windows.items():
                    lower, upper = (FieldValue(self._layout.field(path), value).value for value in endpoints)
                    if (
                        any(
                            v.is_number is not True or v.is_real is not True or v.is_finite is not True
                            for v in (lower, upper)
                        )
                        or (lower < upper) is not S.true
                    ):
                        raise ModelInputError(f"PlotInfo.windows: {path!r} needs increasing finite real bounds")

    def _bind(
        self, initial: ModelProblem, bindings: Mapping[str, ScalarValue | None], *, context: str = "bindings"
    ) -> ModelProblem:
        changes: dict[Symbol, tuple[ModelField, Expr | None]] = {}
        for path, raw in bindings.items():
            try:
                field = self._layout.field(path, context=context)
                value = None if raw is None else FieldValue(field, raw).value
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
            if field.symbol in changes and changes[field.symbol][1] != value:
                raise BindingError("Aliases must have the same input value.", (path,))
            if value is not None:
                bad = [
                    key
                    for key, expected in field.symbol.assumptions0.items()
                    if getattr(value, f"is_{key}", None) not in (None, expected)
                ]
                if value.is_finite is False or value.has(S.NaN, S.ComplexInfinity) or bad:
                    requirements = self._layout.assumptions[field.symbol] or ["finite"]
                    requirement = " and ".join(requirements).replace("nonnegative", "zero or greater").replace("_", " ")
                    title = self._layout.metadata[path].title or path
                    raise BindingError(f"{title} must be {requirement}.", (path,))
            changes[field.symbol] = (field, value)
        constraints = [c for c in initial.constraints if c.kind is not ConstraintKind.INPUT or c.left not in changes]
        constraints.extend(
            ModelConstraint(0, field.path, ConstraintKind.INPUT, field.symbol, value, ".".join(field.path))
            for field, value in changes.values()
            if value is not None
        )
        return replace(
            initial, constraints=tuple(replace(c, index=i) for i, c in enumerate(constraints)), candidates=()
        )

    def _browser_bindings(self, bindings: Mapping[str, str | None], *, filters: bool = False) -> dict[str, Expr | None]:
        """Apply the same edit policy used to decide which fields have editors."""
        parsed = {}
        for path, text in bindings.items():
            if (
                path not in self._layout.fields
                or path in self._layout.hidden
                or (not filters and path not in self._layout.editable)
            ):
                raise BindingError("This field cannot be edited.", (path,))
            if text is not None and not isinstance(text, str):
                raise BindingError("Enter input values as text.", (path,))
            try:
                parsed[path] = None if text is None or not text.strip() else parse_scalar(text)
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
        return parsed

    def snapshot(
        self,
        bindings: Mapping[str, ScalarValue | None] | None = None,
        *,
        raw: bool = False,
        bounds: Mapping[str, tuple[ScalarValue | None, ScalarValue | None]] | None = None,
        case: int | None = None,
        point: Mapping[str, ScalarValue] | None = None,
    ) -> dict[str, Any]:
        """Return a JSON-ready view; invalid edits raise without changing state.

        Bindings replace initial inputs, and ``None`` drops an observation
        without removing definitions. Omit a path to restore its initial input
        state. Several bindings are checked together as one proposed case.
        Raw views show definitions without solving. Bound views label an
        undecided audit as a conditional preview.
        """
        problem = self._bind(self._initial, bindings or {})
        family_view = None
        exploration: dict[str, Any] | None = None
        values: dict[Symbol, Expr]
        ranges = self._ranges(bounds or {})
        if raw:
            values, verdict = {}, "raw"
        else:
            try:
                family_view = self._family_view(problem, ranges)
            except ModelInputError as error:
                if self._source_family is not None or ranges or case is not None or point is not None:
                    raise
                # Ordinary GUI previews also support inputs outside the family's exact-scalar scope.
                values, verdict = _preview(problem)
                exploration = {"kind": "unsupported", "reason": str(error)}
            else:
                family_view.select(case=case, point=point)
                values = family_view.values()
                exploration = family_view.summary()
                verdict = (
                    "empty"
                    if family_view.family.is_empty is True
                    else "satisfied"
                    if family_view.reference is not None
                    else "alternatives"
                    if family_view.cases
                    else "unknown"
                )
        definitions = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
        supplied = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.INPUT}
        printer = self._layout.printer()
        constraints: dict[Symbol, list[dict[str, str]]] = {}
        for c in problem.constraints:
            if ".".join(c.path) in self._layout.hidden:
                continue
            entry = {
                "kind": str(c.kind),
                "path": ".".join(c.path),
                "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
                "mathml": printer.equation(c.left, c.right),
            }
            for symbol in c.symbols:
                constraints.setdefault(symbol, []).append(entry)
        context = _Snapshot(problem, printer, supplied, self._initial_inputs, definitions, constraints)
        rows: list[dict[str, Any]] = []
        symbols: list[dict[str, Any]] = []
        for path, field in self._layout.fields.items():
            expression = definitions.get(field.symbol, field.symbol) if raw else values[field.symbol]
            row = self._field_view(path, expression, context)
            if path not in self._layout.hidden:
                row["bounds"] = [
                    self._input_text(path, endpoint) if endpoint is not None else ""
                    for endpoint in ranges.get(field.symbol, (None, None))
                ]
                if family_view is not None:
                    row["possible_values"] = family_view.possible(field)
            (symbols if path in self._layout.hidden else rows).append(row)
        status, message = _preview_status(raw, verdict)
        root = self._layout.metadata[""]
        return {
            "title": root.title or problem.model.__name__,
            "name": problem.name,
            "description": root.description,
            "info": root.info,
            "raw": raw,
            "fields": rows,
            "symbols": symbols,
            "components": [
                {
                    "path": path,
                    "title": self._layout.metadata[path].title or path.split(".")[-1],
                    "description": self._layout.metadata[path].description,
                }
                for path in self._layout.components
                if path not in self._layout.hidden
            ],
            "equations": [dict(equation) for equation in self._layout.equations],
            "status": status,
            "condition": message,
            "exploration": exploration,
            "highlights": list(self._layout.model_info.highlights),
            "initial_field": self._layout.model_info.initial_field,
            "plot": self._plot_info(self._layout.model_info.plot),
            "scenarios": [
                {
                    "title": scenario.title,
                    "description": scenario.description,
                    "select": scenario.select,
                    "plot": self._plot_info(scenario.plot),
                    "bindings": {
                        path: None
                        if value is None
                        else self._input_text(path, FieldValue(self._layout.field(path), value).value)
                        for path, value in scenario.bindings.items()
                    },
                    "bounds": {
                        path: [
                            None
                            if endpoint is None
                            else self._input_text(path, FieldValue(self._layout.field(path), endpoint).value)
                            for endpoint in endpoints
                        ]
                        for path, endpoints in scenario.bounds.items()
                    },
                }
                for scenario in self._layout.model_info.scenarios
            ],
            "hidden_inputs": sum(
                c.kind is ConstraintKind.INPUT and ".".join(c.path) in self._layout.hidden for c in problem.constraints
            ),
        }

    def _plot_info(self, plot: PlotInfo | None) -> dict[str, Any] | None:
        if plot is None:
            return None
        return {
            "controls": list(plot.controls),
            "output": plot.output,
            "windows": {
                path: [self._input_text(path, FieldValue(self._layout.field(path), value).value) for value in endpoints]
                for path, endpoints in plot.windows.items()
            },
            "view": plot.view,
        }

    def _ranges(
        self, bounds: Mapping[str, tuple[ScalarValue | None, ScalarValue | None]]
    ) -> dict[Symbol, tuple[Expr | None, Expr | None]]:
        ranges: dict[Symbol, tuple[Expr | None, Expr | None]] = {}
        for path, endpoints in bounds.items():
            field = self._layout.field(path, context="bounds")
            if path in self._layout.hidden:
                raise BindingError("This field cannot be bounded.", (path,))
            if len(endpoints) != 2:
                raise BindingError("Supply a lower and upper bound; either may be omitted.", (path,))
            lower, upper = (None if endpoint is None else FieldValue(field, endpoint).value for endpoint in endpoints)
            if any(
                value is not None
                and (value.is_number is not True or value.is_real is not True or value.is_finite is not True)
                for value in (lower, upper)
            ):
                raise BindingError("Bounds must be finite real numbers.", (path,))
            if lower is not None and upper is not None and (lower > upper) is S.true:
                raise BindingError("The lower bound must not exceed the upper bound.", (path,))
            if field.symbol in ranges and ranges[field.symbol] != (lower, upper):
                raise BindingError("Aliases must have the same bounds.", (path,))
            if lower is not None or upper is not None:
                ranges[field.symbol] = (lower, upper)
        return ranges

    def _family_view(
        self, problem: ModelProblem, ranges: Mapping[Symbol, tuple[Expr | None, Expr | None]]
    ) -> FamilyView:
        from modello._gui.families import FamilyView

        if self._source_family is not None:
            # Reprepare edited observations with the same identities, preferences, and inherited filters.
            family = SolutionFamily._prepare(problem, self._source_family._preferred, self._source_family.conditions)
        else:
            family = problem.solutions()
        conditions = []
        for symbol, (lower, upper) in ranges.items():
            if lower is not None:
                conditions.append(symbol >= lower)
            if upper is not None:
                conditions.append(symbol <= upper)
        try:
            constrained = family.where(*conditions)
        except TypeError as error:
            raise ModelInputError("Bounds require an ordered real-valued quantity") from error
        return FamilyView(constrained, self._layout)

    def plot(
        self,
        bindings: Mapping[str, ScalarValue | None] | None = None,
        *,
        controls: Sequence[str],
        output: str,
        windows: Mapping[str, Sequence[ScalarValue]],
        bounds: Mapping[str, tuple[ScalarValue | None, ScalarValue | None]] | None = None,
        case: int | None = None,
        point: Mapping[str, ScalarValue] | None = None,
    ) -> dict[str, Any]:
        """Return checked samples of one or two coordinates, or a reference section.

        Windows are display choices, not observations. Omitted coordinates require
        a checked reference. Rejected or undecided samples never establish emptiness.
        """
        view = self._family_view(self._bind(self._initial, bindings or {}), self._ranges(bounds or {}))
        view.select(case=case, point=point)
        return view.plot(controls, output, windows)

    def _input_text(self, path: str, value: Expr) -> str:
        if self._layout.metadata[path].number_format == "percent":
            return self.problem.format(value * 100) + "%"
        return self.problem.format(value)

    def _field_view(self, path: str, expression: Expr, context: _Snapshot) -> dict[str, Any]:
        field = self._layout.fields[path]
        symbol = field.symbol
        info = self._layout.metadata[path]
        hidden = path in self._layout.hidden
        kind = _field_kind(symbol, expression, context.supplied, self._layout.copied)
        if symbol in context.supplied:
            provenance = "Supplied input"
        elif symbol in self._layout.copied:
            provenance = "Recorded component result"
        elif symbol in context.definitions:
            provenance = "Calculated from a model definition"
        elif kind == "free":
            provenance = "Free input"
        else:
            provenance = "Inferred from the other observations"
        scaled = factor_terms(expression * 100 if info.number_format == "percent" else expression)
        parenthesize = info.number_format == "percent" and scaled.is_Add
        number, approximate = ("", False) if hidden else display_number(scaled, percent=info.number_format == "percent")
        return {
            "path": path,
            "name": field.path[-1],
            "depth": len(field.path) - 1,
            "title": info.title or field.path[-1].replace("_", " "),
            "description": info.description,
            "info": info.info,
            "symbol": info.symbol or path,
            "kind": kind,
            "provenance": provenance,
            "unit": info.unit,
            "number_format": info.number_format,
            "value": ("Hidden input" if symbol in context.supplied else "Hidden quantity")
            if hidden
            else context.problem.format(expression),
            "value_state": "hidden"
            if hidden
            else "free"
            if kind == "free"
            else "numeric"
            if expression.is_number is True
            else "symbolic",
            "display_value": number,
            "approximate": approximate,
            "value_mathml": "" if hidden else context.printer.expression(scaled, parenthesize=parenthesize),
            "exact_mathml": "" if hidden else context.printer.expression(scaled, parenthesize=parenthesize),
            "symbol_mathml": context.printer.expression(symbol),
            "input": self._input_text(path, context.supplied[symbol])
            if symbol in context.supplied and not hidden
            else "",
            "initial_input": self._input_text(path, context.initial_inputs[symbol])
            if symbol in context.initial_inputs and not hidden
            else "",
            "input_state": "hidden" if hidden else _input_state(symbol, context.supplied, context.initial_inputs),
            "editable": path in self._layout.editable,
            "assumptions": list(self._layout.assumptions[symbol]),
            "constraints": [] if hidden else list(context.constraints.get(symbol, ())),
            "aliases": []
            if hidden
            else [alias for alias in self._layout.aliases[symbol] if alias not in self._layout.hidden],
            "depends_on": []
            if hidden
            else [
                {"path": name, "title": self._layout.metadata[name].title or item.path[-1].replace("_", " ")}
                for name, item in self._layout.fields.items()
                if item.symbol in expression.free_symbols
                and name not in self._layout.hidden
                and item.symbol != symbol
                and name == self._layout.aliases[item.symbol][0]
            ],
        }


def _field_kind(symbol: Symbol, value: Expr, supplied: Mapping[Symbol, Expr], copied: frozenset[Symbol]) -> str:
    if symbol in supplied:
        return "input"
    if symbol in copied:
        return "fixed"
    return "free" if value == symbol else "derived"


def _input_state(symbol: Symbol, supplied: Mapping[Symbol, Expr], initial: Mapping[Symbol, Expr]) -> str:
    if symbol not in supplied:
        return "dropped" if symbol in initial else "unspecified"
    if symbol not in initial:
        return "added"
    return "initial" if supplied[symbol] == initial[symbol] else "overridden"


def _preview_status(raw: bool, verdict: str) -> tuple[str, str]:
    if raw:
        return "raw", "Definitions only; recorded inputs retained"
    if verdict == "satisfied":
        return "checked", "All equations and domains checked"
    if verdict == "empty":
        return "empty", "No state satisfies all retained observations, filters, and domains"
    if verdict == "alternatives":
        return "alternatives", "Several independently checked cases satisfy the question"
    return "conditional", "Symbolic or undecided: not a validated numeric state"


def create_app(
    model: type[Modello] | Modello | ModelProblem | SolutionFamily,
    name: str = "exploration",
    *,
    inputs: Mapping[str, ScalarValue] | None = None,
    metadata: Mapping[str, FieldInfo] | None = None,
) -> Flask:
    """Create a local Flask app; each browser holds its own trial bindings.

    Serve with ``app.run(host='127.0.0.1', debug=False)``. The bundled frontend
    needs no CDN, JavaScript build, or external math renderer. The prototype
    runs symbolic work synchronously and is intended for small local models.
    """
    try:
        from flask import Flask, jsonify, render_template, request
    except ImportError as error:
        raise ImportError("The browser explorer requires: pip install 'modello[gui]'") from error

    explorer = Explorer(model, name, inputs=inputs, metadata=metadata)
    app = Flask(__name__, template_folder="_gui/templates", static_folder="_gui/static", static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=16_384, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    def arguments(payload: Any, *, plotting: bool = False) -> dict[str, Any]:
        allowed = {"bindings", "bounds", "case", "point"} | ({"controls", "output", "windows"} if plotting else {"raw"})
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise ValueError("Expected observations, bounds, and an optional case or point")
        result: dict[str, Any] = {}
        for name in ("bindings", "point"):
            value = payload.get(name, {} if name == "bindings" else None)
            if value is None and name == "point":
                continue
            if not isinstance(value, dict) or len(value) > len(explorer.problem.fields):
                raise ValueError(f"Invalid {name}")
            if name == "point" and any(not isinstance(v, str) or not v.strip() for v in value.values()):
                raise ValueError("Point coordinates must be exact scalar text")
            result[name] = explorer._browser_bindings(value, filters=name == "point")
        result["bounds"] = {}
        for name in ("bounds", "windows") if plotting else ("bounds",):
            mapping = payload.get(name, {})
            if not isinstance(mapping, dict) or len(mapping) > len(explorer.problem.fields):
                raise ValueError(f"Invalid {name}")
            parsed = {}
            for path, endpoints in mapping.items():
                if not isinstance(endpoints, list) or len(endpoints) != 2:
                    raise ValueError(f"{name}: expected pairs of endpoint values")
                parsed[path] = tuple(
                    explorer._browser_bindings({path: value}, filters=True)[path] for value in endpoints
                )
                if name == "windows" and None in parsed[path]:
                    raise ValueError("Plotting windows need two endpoints")
            result[name] = parsed
        case = payload.get("case")
        if case is not None and (isinstance(case, bool) or not isinstance(case, int)):
            raise ValueError("Case must be an integer index")
        result["case"] = case
        if plotting:
            controls, output = payload.get("controls"), payload.get("output")
            if (
                not isinstance(controls, list)
                or any(not isinstance(path, str) for path in controls)
                or not isinstance(output, str)
            ):
                raise ValueError("Choose controls and an output")
            result.update(controls=controls, output=output)
        else:
            raw = payload.get("raw", False)
            if not isinstance(raw, bool):
                raise ValueError("Invalid raw flag")
            result["raw"] = raw
        return result

    def response(*, plotting: bool = False) -> Any:
        try:
            options = arguments(request.get_json(silent=True), plotting=plotting)
            return jsonify(explorer.plot(**options) if plotting else explorer.snapshot(**options))
        except BindingError as error:
            return jsonify(error=str(error), paths=list(error.paths)), 422
        except (ModelInputError, ModelValidationError) as error:
            return jsonify(error=str(error), paths=[]), 422
        except ValueError as error:
            return jsonify(error=str(error), paths=[]), 400

    @app.post("/api/explore")
    def explore() -> Any:
        return response()

    @app.post("/api/plot")
    def plot() -> Any:
        return response(plotting=True)

    return app

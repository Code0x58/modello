"""Optional local exploration of nested models (install ``modello[gui]``).

Presentation metadata never changes the algebra. Raw snapshots do not solve;
browser edits are exact scalars parsed without evaluating expression strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dataclass_field, replace
from typing import TYPE_CHECKING, Any, Callable, Literal, Mapping, TypeVar, cast

from sympy import Basic, Expr, Symbol, check_assumptions, factor_terms, sympify

from modello import (
    Annotation,
    Modello,
    ScalarValue,
    _solve_problem,
    annotate,
    annotate_model,
    inspect_model,
)
from modello._gui.collections import component_views, update_collection_observations
from modello._gui.formatting import display_number
from modello._gui.inputs import (
    ExplorerValue,
    ModelVectorInput as ModelVectorInput,
    binding_texts,
    parse_model_vector,
    parse_scalar,
    parse_vector,
    request_limit,
)
from modello._gui.mathml import EquationPrinter
from modello._gui.shapes import ShapePlan, _entries
from modello._gui.vectors import vector_input_text, vector_view
from modello._inputs import _validate_vector_values
from modello._model import _has_symbolic_shape
from modello._operators import _definitely_undefined
from modello._paths import _path_label, _path_value
from modello._problem import _FieldPrinter
from modello.diagnostics import Candidate, ConstraintKind, FieldValue, ModelConstraint, ModelField, ModelProblem
from modello.errors import ModelInputError, ModelValidationError, SolverBackendError

if TYPE_CHECKING:
    from flask import Flask

__all__ = ["BindingError", "Explorer", "FieldInfo", "ModelInfo", "ModelVectorInput", "Scenario", "create_app", "gui"]

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
class Scenario:
    """A named exploration that restores initial inputs, then applies bindings.

    All arguments are named. Only visible, editable quantities may be bound.
    ``select`` opens a field in the inspector; it need not be the quantity
    being observed.
    """

    title: str
    description: str
    bindings: Mapping[str, ExplorerValue | None] = dataclass_field(default_factory=dict)
    select: str = ""


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

    def __call__(self, model: type[_ModelT]) -> type[_ModelT]:
        """Keep the decorated class intact and register its model metadata."""
        return annotate_model(model, self)


@dataclass(frozen=True)
class _Snapshot:
    problem: ModelProblem
    printer: EquationPrinter
    text: _FieldPrinter
    supplied: Mapping[Symbol, Expr]
    initial_inputs: Mapping[Symbol, Expr]
    definitions: Mapping[Symbol, Expr]
    constraints: Mapping[Symbol, list[dict[str, str]]]

    def format(self, expression: Basic) -> str:
        return self.text.doprint(expression)


class BindingError(ModelValidationError):
    """A rejected edit, with field paths that the interface can highlight."""

    def __init__(self, message: str, paths: tuple[str, ...] = ()) -> None:
        self.paths = paths
        super().__init__(message)


def _instance_problem(instance: Modello, name: str) -> ModelProblem:
    """Reopen original inputs; solved outputs must not become observations."""
    model = type(instance)
    _validate_vector_values(model, instance)
    problem = inspect_model(model, name)
    bindings = {
        instance._modello_bound_fields[d].symbol: problem.field(*path).symbol
        for path, d in model._modello_definition.fields.items()
    }
    constraints = list(problem.constraints)
    seen: set[Symbol] = set()
    for path, dummy in model._modello_definition.fields.items():
        value = _path_value(instance, path)
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
                    _path_label(path),
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
            _path_label(check.value.field.path)
            for check in audit.assumptions
            if check.actual not in (None, check.expected)
        )
        rejected_paths.update(
            _path_label(check.constraint.path) for check in audit.residuals if check.verdict == "violated"
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
    """Prepare raw or bound views of a class, instance, or recorded problem.

    Declare model text with ``@ModelInfo(...)`` and field labels with ``gui``.
    Component declarations are inherited by the layout; ``metadata`` overrides
    field or component labels by full dotted path. ``inputs`` supplies initial
    bindings by dotted path. The original model instance is never mutated.
    """

    def __init__(
        self,
        model: type[Modello] | Modello | ModelProblem,
        name: str = "exploration",
        *,
        inputs: Mapping[str, ExplorerValue] | None = None,
        metadata: Mapping[str, FieldInfo] | None = None,
    ) -> None:
        self._shape_plan: ShapePlan | None = None
        if isinstance(model, type):
            source = getattr(model, "_modello_shape_source", model)
            if _has_symbolic_shape(source):
                self._shape_plan = ShapePlan(model, inputs or {})
                model = self._shape_plan.initial_model
        resizable = self._shape_plan.axes if self._shape_plan is not None else frozenset()
        self._shape_dimensions = frozenset(self._shape_plan.dimensions) if self._shape_plan is not None else frozenset()
        self._initialize(model, name, inputs or {}, metadata or {}, resizable)

    def _initialize(
        self,
        model: type[Modello] | Modello | ModelProblem,
        name: str,
        inputs: Mapping[str, ExplorerValue],
        metadata: Mapping[str, FieldInfo],
        resizable: frozenset[str],
    ) -> None:
        from modello._gui.presentation import Layout

        if isinstance(model, ModelProblem):
            self.problem = model
        elif isinstance(model, Modello):
            self.problem = _instance_problem(model, name)
        else:
            self.problem = inspect_model(model, name)
        self._metadata = metadata
        self._layout = Layout(self.problem, metadata, resizable=resizable)
        self._input_axes = frozenset(path for path in inputs if path in self._layout.collections)
        self._initial = self._bind(self.problem, inputs, context="inputs")
        self._initial_inputs = {c.left: c.right for c in self._initial.constraints if c.kind is ConstraintKind.INPUT}

    def _finite_context(self, bindings: Mapping[str, object]) -> Explorer:
        if self._shape_plan is None:
            return self
        model, baseline = self._shape_plan.select(bindings)
        target = object.__new__(Explorer)
        target._shape_plan = None
        target._shape_dimensions = frozenset(self._shape_plan.dimensions)
        target._initialize(model, self.problem.name, baseline, self._metadata, self._shape_plan.axes)
        return target

    def _window_history(self, data: dict[str, Any]) -> dict[str, Any]:
        """Describe the original window even when the finite context is shorter."""
        observed = {row["path"] for row in data["fields"] if row["kind"] == "input"}
        for row in data["vectors"]:
            if not row["resizable"]:
                continue
            original = self._layout.vectors[row["path"]]
            entries = [self._initial_inputs.get(symbol) for symbol in original]
            row["initial_shape"] = list(original.shape)
            row["window_changed"] = row["shape"] != row["initial_shape"]
            row["initial_input"] = (
                "["
                + ", ".join("?" if value is None else self._input_text(row["path"], value) for value in entries)
                + "]"
                if any(value is not None for value in entries)
                else ""
            )
            if row["window_changed"] and row["initial_input"]:
                row["input_state"] = "overridden" if observed.intersection(row["entries"]) else "dropped"
        return data

    def _bind(
        self,
        initial: ModelProblem,
        bindings: Mapping[str, ExplorerValue | None],
        *,
        context: str = "bindings",
    ) -> ModelProblem:
        changes: dict[Symbol, tuple[ModelField, Expr | None]] = {}
        for path, scalar in self._layout.expand_bindings(bindings):
            try:
                field = self._layout.field(path, context=context)
                value = None if scalar is None else FieldValue(field, cast(ScalarValue, scalar)).value
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
            if field.symbol in changes and changes[field.symbol][1] != value:
                raise BindingError("Aliases must have the same input value.", (path,))
            if value is not None:
                if (
                    value.is_finite is False
                    or _definitely_undefined(value)
                    or check_assumptions(value, against=field.symbol) is False
                ):
                    requirements = self._layout.assumptions[field.symbol] or ["finite"]
                    requirement = " and ".join(requirements).replace("nonnegative", "zero or greater").replace("_", " ")
                    title = self._layout.metadata[path].title or path
                    raise BindingError(f"{title} must be {requirement}.", (path,))
            changes[field.symbol] = (field, value)
        constraints = [c for c in initial.constraints if c.kind is not ConstraintKind.INPUT or c.left not in changes]
        constraints.extend(
            ModelConstraint(0, field.path, ConstraintKind.INPUT, field.symbol, value, _path_label(field.path))
            for field, value in changes.values()
            if value is not None
        )
        return replace(
            initial, constraints=tuple(replace(c, index=i) for i, c in enumerate(constraints)), candidates=()
        )

    def _browser_bindings(self, bindings: Mapping[str, str | None]) -> dict[str, ExplorerValue | None]:
        """Apply the same edit policy used to decide which fields have editors."""
        parsed: dict[str, ExplorerValue | None] = {}
        for path, text in bindings.items():
            if (
                path not in self._layout.editable
                and path not in self._layout.editable_vectors
                and path not in self._layout.editable_collections
            ):
                raise BindingError("This field cannot be edited.", (path,))
            if text is not None and not isinstance(text, str):
                raise BindingError("Enter input values as text.", (path,))
            try:
                parsed[path] = (
                    None
                    if text is None or not text.strip()
                    else parse_model_vector(text, int(self._layout.collections[path]._modello_collection_size))
                    if path in self._layout.collections
                    else parse_vector(text, len(self._layout.vectors[path]))
                    if path in self._layout.vectors
                    else parse_scalar(text)
                )
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
        return parsed

    def _browser_snapshot(self, bindings: Mapping[str, str | None], *, raw: bool) -> dict[str, Any]:
        """Retain accepted arithmetic instead of seeding an unparseable large literal."""
        if self._shape_plan is not None:
            shape_bindings: dict[str, object] = {}
            for path, text in bindings.items():
                if text is not None and not isinstance(text, str):
                    raise BindingError("Enter input values as text.", (path,))
                if path in self._shape_plan.collections:
                    shape_bindings[path] = None if text is None or not text.strip() else parse_model_vector(text, None)
                elif path in self._shape_plan.vectors:
                    shape_bindings[path] = None if text is None or not text.strip() else parse_vector(text, None)
                elif path in self._shape_plan.dimensions:
                    shape_bindings[path] = None if text is None or not text.strip() else parse_scalar(text)
            return self._window_history(self._finite_context(shape_bindings)._browser_snapshot(bindings, raw=raw))
        parsed = self._browser_bindings(bindings)
        sizes = {path: len(value) for path, value in self._layout.vectors.items()}
        texts = binding_texts(bindings, sizes)
        for path, text in bindings.items():
            if path in self._layout.collections and text:
                for i, record in enumerate(json.loads(text)):
                    for name, value in (record or {}).items():
                        texts[f"{path}[{i}].{name}"] = value
        data = self.snapshot(parsed, raw=raw)
        data["bindings"] = binding_texts(bindings, sizes, preserve=self._layout.resizable)
        rows = {row["path"]: row for row in data["fields"]}
        for row in rows.values():
            if row["kind"] == "input":
                text = next((texts[path] for path in row["aliases"] if path in texts), None)
                if text is not None:
                    row["input"] = text
        for row in data["vectors"]:
            if len(row["entries"]) == row["shape"][0]:
                row["input"] = vector_input_text(row["entries"], rows, "input")
            if row["path"] in self._layout.resizable and bindings.get(row["path"]):
                row["input"] = bindings[row["path"]]
        update_collection_observations(data["components"], rows, self._bind(self._initial, parsed), self._layout.hidden)
        return data

    def snapshot(
        self, bindings: Mapping[str, ExplorerValue | None] | None = None, *, raw: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-ready view; invalid edits raise without changing state.

        Bindings replace initial inputs, and ``None`` drops an observation
        without removing definitions. Omit a path to restore its initial input
        state. Several bindings are checked together as one proposed case.
        Raw views show definitions without solving. Bound views label an
        undecided audit as a conditional preview.
        """
        if self._shape_plan is not None:
            return self._window_history(self._finite_context(bindings or {}).snapshot(bindings, raw=raw))
        problem = self._bind(self._initial, bindings or {})
        values, verdict = ({}, "raw") if raw else _preview(problem)
        definitions = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
        supplied = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.INPUT}
        printer = self._layout.printer()
        text = problem._printer(
            *(c.right for c in problem.constraints), *values.values(), *self._initial_inputs.values()
        )
        constraints: dict[Symbol, list[dict[str, str]]] = {}
        for c in problem.constraints:
            if _path_label(c.path) in self._layout.hidden:
                continue
            entry = {
                "kind": str(c.kind),
                "path": _path_label(c.path),
                "equation_path": self._layout.equation_path(c.path, c.right)
                if c.kind is ConstraintKind.DEFINITION
                else "",
                "text": f"{text.doprint(c.left)} = {text.doprint(c.right)}",
                "mathml": printer.equation(c.left, c.right),
            }
            for symbol in c.symbols:
                constraints.setdefault(symbol, []).append(entry)
        context = _Snapshot(problem, printer, text, supplied, self._initial_inputs, definitions, constraints)
        rows: list[dict[str, Any]] = []
        symbols: list[dict[str, Any]] = []
        for path, field in self._layout.fields.items():
            expression = definitions.get(field.symbol, field.symbol) if raw else values[field.symbol]
            row = self._field_view(path, expression, context)
            (symbols if path in self._layout.hidden else rows).append(row)
        status, message = _preview_status(raw, verdict)
        root = self._layout.metadata[""]
        indexed_rows = {row["path"]: row for row in [*rows, *symbols]}
        vector_rows = [
            vector_view(path, self._layout, context, values, indexed_rows, raw) for path in self._layout.vectors
        ]
        for row in vector_rows:
            row["resizable"] = row["path"] in self._layout.resizable
            if row["path"] in self._layout.hidden:
                row.update(value="Hidden vector", value_state="hidden", value_mathml="", exact_mathml="")
                symbols.append(row)
        return {
            "title": root.title or problem.model.__name__,
            "name": problem.name,
            "description": root.description,
            "info": root.info,
            "raw": raw,
            "fields": rows,
            "symbols": symbols,
            "vectors": [row for row in vector_rows if row["path"] not in self._layout.hidden],
            "components": component_views(
                self._layout, context, indexed_rows, input_axes=self._input_axes, dimensions=self._shape_dimensions
            ),
            "equations": [dict(equation) for equation in self._layout.equations],
            "status": status,
            "condition": message,
            "highlights": list(self._layout.model_info.highlights),
            "initial_field": self._layout.model_info.initial_field,
            "scenarios": [
                {
                    "title": scenario.title,
                    "description": scenario.description,
                    "select": scenario.select,
                    "bindings": self._scenario_bindings(scenario, context),
                }
                for scenario in self._layout.model_info.scenarios
            ],
            "hidden_inputs": sum(
                c.kind is ConstraintKind.INPUT and _path_label(c.path) in self._layout.hidden
                for c in problem.constraints
            ),
        }

    def _scenario_bindings(self, scenario: Scenario, context: _Snapshot) -> dict[str, str | None]:
        result: dict[str, str | None] = {}
        for path, value in scenario.bindings.items():
            if path in self._layout.collections:
                result[path] = (
                    None
                    if value is None
                    else json.dumps(
                        [
                            {
                                name: None if scalar is None else context.format(sympify(scalar))
                                for name, scalar in (record or {}).items()
                            }
                            for record in cast(ModelVectorInput, value)
                        ]
                    )
                )
            elif path in self._layout.resizable:
                result[path] = (
                    None
                    if value is None
                    else "["
                    + ", ".join("?" if entry is None else context.format(sympify(entry)) for entry in _entries(value))
                    + "]"
                )
            else:
                for name, entry in self._layout.expand_bindings({path: value}):
                    if entry is not None and name not in self._layout.fields:
                        # Scenarios restore their original window. Retain editor
                        # text for a coordinate beyond the current shorter one.
                        parent = name.rpartition("[")[0]
                        result[name] = self._input_text(parent, sympify(entry), context.format)
                        continue
                    result[name] = (
                        None
                        if entry is None
                        else self._input_text(
                            name, FieldValue(self._layout.field(name), cast(ScalarValue, entry)).value, context.format
                        )
                    )
        return result

    def _input_text(self, path: str, value: Expr, formatter: Callable[[Basic], str] | None = None) -> str:
        render = self.problem.format if formatter is None else formatter
        if self._layout.metadata[path].number_format == "percent":
            return render(value * 100) + "%"
        return render(value)

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
            "parent": _path_label(field.path[:-1]),
            "title": self._layout.title(path),
            "description": info.description,
            "info": info.info,
            "symbol": info.symbol or path,
            "kind": kind,
            "provenance": provenance,
            "unit": info.unit,
            "number_format": info.number_format,
            "value": ("Hidden input" if symbol in context.supplied else "Hidden quantity")
            if hidden
            else context.format(expression),
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
            "input": self._input_text(path, context.supplied[symbol], context.format)
            if symbol in context.supplied and not hidden
            else "",
            "initial_input": self._input_text(path, context.initial_inputs[symbol], context.format)
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
                {"path": name, "title": self._layout.title(name)}
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
    return "conditional", "Symbolic or undecided: not a validated numeric state"


def create_app(
    model: type[Modello] | Modello | ModelProblem,
    name: str = "exploration",
    *,
    inputs: Mapping[str, ExplorerValue] | None = None,
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
    app.config.update(
        MAX_CONTENT_LENGTH=request_limit(explorer._layout.editable),
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
    )

    @app.errorhandler(413)
    def too_large(_error: Exception) -> Any:
        return jsonify(
            error="The input request is too large. Use shorter values or change fewer entries.", paths=[]
        ), 413

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/explore")
    def explore() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) - {"bindings", "raw"}:
            return jsonify(error="Expected bindings and an optional raw flag.", paths=[]), 400
        bindings, raw = payload.get("bindings", {}), payload.get("raw", False)
        if (
            not isinstance(bindings, dict)
            or not isinstance(raw, bool)
            or len(bindings) > max(256, len(explorer.problem.fields))
        ):
            return jsonify(error="Invalid bindings or raw flag.", paths=[]), 400
        try:
            return jsonify(explorer._browser_snapshot(bindings, raw=raw))
        except BindingError as error:
            return jsonify(error=str(error), paths=list(dict.fromkeys([*bindings, *error.paths]))), 422
        except (ModelInputError, ModelValidationError) as error:
            return jsonify(error=str(error), paths=list(bindings)), 422

    return app

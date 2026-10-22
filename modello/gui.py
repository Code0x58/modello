"""Optional local exploration of nested models (install ``modello[gui]``).

Presentation metadata never changes the algebra. Raw snapshots do not solve;
browser edits are exact scalars parsed without evaluating expression strings.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any, Literal, Mapping

from sympy import Expr, S, Symbol, factor_terms

from modello import Modello, ScalarValue, _solve_problem, inspect_model
from modello._gui.formatting import display_number
from modello._gui.inputs import parse_scalar
from modello._gui.mathml import EquationPrinter
from modello.diagnostics import Candidate, ConstraintKind, FieldValue, ModelConstraint, ModelField, ModelProblem
from modello.errors import ModelInputError, ModelValidationError, SolverBackendError

if TYPE_CHECKING:
    from flask import Flask

__all__ = ["BindingError", "Explorer", "FieldInfo", "ModelInfo", "Scenario", "create_app"]


@dataclass(frozen=True)
class FieldInfo:
    """Display metadata for a scalar or component, without changing its algebra.

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


@dataclass(frozen=True)
class Scenario:
    """A named exploration that restores initial inputs, then applies bindings.

    Only visible, editable quantities may be bound. ``select`` opens a field
    in the inspector; it need not be the quantity being observed.
    """

    title: str
    description: str
    bindings: Mapping[str, ScalarValue | None] = dataclass_field(default_factory=dict)
    select: str = ""


@dataclass(frozen=True)
class ModelInfo:
    """Model-level text and field metadata for a ``__gui__`` declaration.

    Field keys are relative dotted paths. Component classes supply their own
    declarations; a parent's ``fields`` can override them by full path.
    ``highlights`` selects result cards, ``initial_field`` opens an inspector,
    and ``scenarios`` supplies named explorations from the initial inputs.
    """

    title: str = ""
    description: str = ""
    info: str = ""
    fields: Mapping[str, FieldInfo] = dataclass_field(default_factory=dict)
    highlights: tuple[str, ...] = ()
    initial_field: str = ""
    scenarios: tuple[Scenario, ...] = ()


@dataclass(frozen=True)
class _Snapshot:
    problem: ModelProblem
    printer: EquationPrinter
    supplied: Mapping[Symbol, Expr]
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
    """Prepare raw or bound views of a class, instance, or recorded problem.

    Declare model text and field labels with ``__gui__ = ModelInfo(...)``.
    Component declarations are inherited by the layout; ``metadata`` overrides
    field or component labels by full dotted path. ``inputs`` supplies initial
    bindings by dotted path. The original model instance is never mutated.
    """

    def __init__(
        self,
        model: type[Modello] | Modello | ModelProblem,
        name: str = "exploration",
        *,
        inputs: Mapping[str, ScalarValue] | None = None,
        metadata: Mapping[str, FieldInfo] | None = None,
    ) -> None:
        from modello._gui.presentation import Layout

        if isinstance(model, ModelProblem):
            self.problem = model
        elif isinstance(model, Modello):
            self.problem = _instance_problem(model, name)
        else:
            self.problem = inspect_model(model, name)
        self._layout = Layout(self.problem, metadata or {})
        self._initial = self._bind(self.problem, inputs or {})

    def _bind(self, initial: ModelProblem, bindings: Mapping[str, ScalarValue | None]) -> ModelProblem:
        changes: dict[Symbol, tuple[ModelField, Expr | None]] = {}
        for path, raw in bindings.items():
            try:
                field = self._layout.field(path)
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

    def _browser_bindings(self, bindings: Mapping[str, str | None]) -> dict[str, Expr | None]:
        """Apply the same edit policy used to decide which fields have editors."""
        parsed = {}
        for path, text in bindings.items():
            if path not in self._layout.editable:
                raise BindingError("This field cannot be edited.", (path,))
            if text is not None and not isinstance(text, str):
                raise BindingError("Enter input values as text.", (path,))
            try:
                parsed[path] = None if text is None or not text.strip() else parse_scalar(text)
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
        return parsed

    def snapshot(
        self, bindings: Mapping[str, ScalarValue | None] | None = None, *, raw: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-ready view; invalid edits raise without changing state.

        Bindings replace initial inputs, and ``None`` releases an input. Raw
        views show definitions without solving and still identify observations.
        Bound views label an undecided audit as a conditional preview.
        """
        problem = self._bind(self._initial, bindings or {})
        values, verdict = ({}, "raw") if raw else _preview(problem)
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
        context = _Snapshot(problem, printer, supplied, definitions, constraints)
        rows: list[dict[str, Any]] = []
        symbols: list[dict[str, Any]] = []
        for path, field in self._layout.fields.items():
            expression = definitions.get(field.symbol, field.symbol) if raw else values[field.symbol]
            row = self._field_view(path, expression, context)
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
            "highlights": list(self._layout.model_info.highlights),
            "initial_field": self._layout.model_info.initial_field,
            "scenarios": [
                {
                    "title": scenario.title,
                    "description": scenario.description,
                    "select": scenario.select,
                    "bindings": {
                        path: None
                        if value is None
                        else self._input_text(path, FieldValue(self._layout.field(path), value).value)
                        for path, value in scenario.bindings.items()
                    },
                }
                for scenario in self._layout.model_info.scenarios
            ],
            "hidden_inputs": sum(
                c.kind is ConstraintKind.INPUT and ".".join(c.path) in self._layout.hidden for c in problem.constraints
            ),
        }

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

    @app.post("/api/explore")
    def explore() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) - {"bindings", "raw"}:
            return jsonify(error="Expected bindings and an optional raw flag.", paths=[]), 400
        bindings, raw = payload.get("bindings", {}), payload.get("raw", False)
        if not isinstance(bindings, dict) or not isinstance(raw, bool) or len(bindings) > len(explorer.problem.fields):
            return jsonify(error="Invalid bindings or raw flag.", paths=[]), 400
        try:
            return jsonify(explorer.snapshot(explorer._browser_bindings(bindings), raw=raw))
        except BindingError as error:
            return jsonify(error=str(error), paths=list(dict.fromkeys([*bindings, *error.paths]))), 422
        except (ModelInputError, ModelValidationError) as error:
            return jsonify(error=str(error), paths=list(bindings)), 422

    return app

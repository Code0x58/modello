"""Optional local browser exploration of nested models (install ``modello[gui]``).

Presentation metadata never changes the algebra. Raw snapshots do not solve;
interactive bindings are exact scalars parsed without Python or SymPy evaluation.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, replace
from html.entities import html5
from typing import TYPE_CHECKING, Any, Mapping
from xml.dom.minidom import Text

from sympy import E, Eq, Expr, I, Rational, S, Symbol, pi, solve, sqrt
from sympy.printing.mathml import MathMLPresentationPrinter

from modello import Modello, ScalarValue, inspect_model
from modello.diagnostics import Candidate, ConstraintKind, FieldValue, ModelConstraint, ModelProblem
from modello.errors import ModelInputError, ModelValidationError

if TYPE_CHECKING:
    from flask import Flask


@dataclass(frozen=True)
class FieldInfo:
    """Display-only metadata for a scalar, component, or the model at ``''``.

    ``symbol`` is literal text in MathML, not LaTeX. Hidden quantities retain
    their input constraints but have no editor, value, or observation in the
    browser response. This is presentation, not a secrecy boundary: derived
    outputs can reveal their inputs.
    """

    title: str = ""
    description: str = ""
    info: str = ""
    symbol: str = ""
    hidden: bool = False


class BindingError(ModelValidationError):
    """A rejected edit, with field paths that the interface can highlight."""

    def __init__(self, message: str, paths: tuple[str, ...] = ()) -> None:
        self.paths = paths
        super().__init__(message)


def _parse_scalar(text: str) -> Expr:
    """Read a deliberately small arithmetic language, never eval/sympify text."""
    if not text.strip() or len(text) > 160:
        raise ModelInputError("Enter a number or a short exact arithmetic expression.")
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError) as error:
        raise ModelInputError("Use numbers, +, -, *, /, **, sqrt, pi, E, or I.") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ModelInputError("The arithmetic expression is too long.")

    def read(node: ast.AST) -> Expr:
        result = read_node(node)
        if result.is_Rational and max(int(result.p).bit_length(), int(result.q).bit_length()) > 4096:
            raise ModelInputError("The arithmetic result is too large.")
        return result

    def read_node(node: ast.AST) -> Expr:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            token = ast.get_source_segment(text.strip(), node) or ""
            if len(token) > 32 or not math.isfinite(node.value) or abs(node.value) > 10**100:
                raise ModelInputError("Numeric literals must be finite and at most 1e100.")
            exponent = re.search(r"[eE]([+-]?[0-9_]+)$", token)
            if exponent and abs(int(exponent[1].replace("_", ""))) > 1000:
                raise ModelInputError("Decimal exponents must be between -1000 and 1000.")
            return Rational(token.replace("_", ""))
        if isinstance(node, ast.Name) and node.id in ("pi", "E", "I"):
            return {"pi": pi, "E": E, "I": I}[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = read(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = read(node.left), read(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow) and right.is_Rational and abs(right) <= 100:
                return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return sqrt(read(node.args[0]))
        raise ModelInputError("Use numbers, +, -, *, /, ** (power at most 100), sqrt, pi, E, or I.")

    try:
        result = read(tree.body)
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as error:
        raise ModelInputError(str(error)) from error
    if result.is_finite is not True:
        raise ModelInputError("Inputs must be finite scalar values.")
    return result


class _MathPrinter(MathMLPresentationPrinter):
    """Attach real field paths to symbols without modifying the expressions."""

    def __init__(self, problem: ModelProblem, metadata: Mapping[str, FieldInfo]) -> None:
        super().__init__()
        self.labels: dict[Symbol, tuple[str, str]] = {}
        for field in problem.fields:
            path = ".".join(field.path)
            self.labels.setdefault(field.symbol, (path, metadata[path].symbol or path))

    def _print_Symbol(self, expression: Symbol) -> Any:
        node = self.dom.createElement("mi")
        path, label = self.labels.get(expression, ("", expression.name))
        # SymPy replaces Document.createTextNode with an unescaped writer.
        # Field labels are literal user text and require the standard DOM node.
        text = Text()
        text.data = label
        text.ownerDocument = self.dom
        node.appendChild(text)
        if path:
            node.setAttribute("data-path", path)
            node.setAttribute("tabindex", "0")
        return node

    _print_Dummy = _print_Symbol
    _print_InstanceDummy = _print_Symbol
    _print_BoundInstanceDummy = _print_Symbol


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


def _assumptions(symbol: Symbol) -> list[str]:
    """Omit predicates already implied by more informative requirements."""
    requirements = {k: v for k, v in symbol.assumptions0.items() if k != "commutative"}
    for key in sorted(requirements):
        others = {k: v for k, v in requirements.items() if k != key}
        if Symbol("requirements", **others).assumptions0.get(key) == requirements[key]:
            del requirements[key]
    return [key if expected else f"not {key}" for key, expected in sorted(requirements.items())]


def _preview(problem: ModelProblem) -> tuple[dict[Symbol, Expr], str]:
    """Follow construction's directed expansion, then audit its unique preview.

    A solver answer is a proposal. Unknown audits remain conditional, and no
    branch or sampled point is selected to stand in for an ambiguous model.
    """
    fields = tuple(dict.fromkeys(f.symbol for f in problem.fields))
    values = {symbol: symbol for symbol in fields}
    pending = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
    while pending:
        ready = {left: right for left, right in pending.items() if not right.free_symbols & pending.keys()}
        if not ready:
            break
        for left, right in ready.items():
            values[left] = right.xreplace(values)
            del pending[left]
    equations = [Eq(c.left, c.right).xreplace(values) for c in problem.constraints]
    if S.false in equations:
        raise BindingError("These inputs contradict a model equation.")
    equations = [eq for eq in equations if eq is not S.true]
    try:
        proposals = (
            solve(equations, list(dict.fromkeys([*pending, *fields])), dict=True, check=False) if equations else [{}]
        )
    except Exception as error:
        raise BindingError(f"No preview: the symbolic solver raised {type(error).__name__}.") from error
    accepted = []
    rejected_paths: set[str] = set()
    for proposal in proposals:
        resolved = {symbol: value.xreplace(proposal) for symbol, value in values.items()}
        audit = problem.check(Candidate(tuple(FieldValue(f, resolved[f.symbol]) for f in problem.fields)))
        if audit.verdict != "violated":
            accepted.append((resolved, audit.verdict))
        else:
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
            else "No valid preview was established. Check equations, inputs, and declared domains."
        )
        raise BindingError(message, tuple(sorted(rejected_paths)))
    return accepted[0]


class Explorer:
    """Prepare raw or bound views of a class, instance, or recorded problem.

    Model classes may define ``__gui__ = {'field': FieldInfo(...), '': ...}``.
    Nested metadata is inherited from component classes; explicit ``metadata``
    overrides it by full dotted path. ``inputs`` supplies initial bindings by
    dotted path, including nested fields. Original instances are never mutated.
    """

    def __init__(
        self,
        model: type[Modello] | Modello | ModelProblem,
        name: str = "exploration",
        *,
        inputs: Mapping[str, ScalarValue] | None = None,
        metadata: Mapping[str, FieldInfo] | None = None,
    ) -> None:
        if isinstance(model, ModelProblem):
            problem = model
        elif isinstance(model, Modello):
            problem = _instance_problem(model, name)
        else:
            problem = inspect_model(model, name)
        self.problem = problem
        self.metadata: dict[str, FieldInfo] = {}

        def collect(model: type[Modello], prefix: tuple[str, ...]) -> None:
            self.metadata[".".join(prefix)] = FieldInfo(description=model.__doc__ or "")
            for path in model._modello_fields:
                self.metadata.setdefault(".".join(prefix + path), FieldInfo())
            for attr, child in model._modello_namespace.nested_models.items():
                collect(child._modello_model, (*prefix, attr))
            declared = getattr(model, "__gui__", {})
            for path, info in declared.items():
                full_path = ".".join((*prefix, *path.split("."))) if path else ".".join(prefix)
                self.metadata[full_path] = info

        collect(problem.model, ())
        self.metadata.update(metadata or {})
        valid_paths = {"", *(".".join(field.path) for field in problem.fields)}
        valid_paths.update(".".join(field.path[:i]) for field in problem.fields for i in range(1, len(field.path)))
        for path, info in self.metadata.items():
            if path not in valid_paths or not isinstance(info, FieldInfo):
                raise ModelInputError(f"Invalid GUI metadata at {path!r}")
        self._initial = self._bind(problem, inputs or {})
        self._printer = _MathPrinter(problem, self.metadata)

    def _hidden(self, path: str) -> bool:
        paths = [path]
        field = next((f for f in self.problem.fields if ".".join(f.path) == path), None)
        if field is not None:
            paths.extend(".".join(alias.path) for alias in self.problem.aliases(field))
        for candidate in paths:
            parts = candidate.split(".")
            if any(self.metadata[".".join(parts[:i])].hidden for i in range(1, len(parts) + 1)):
                return True
        return False

    def _bind(self, problem: ModelProblem, bindings: Mapping[str, ScalarValue | None]) -> ModelProblem:
        constraints = list(problem.constraints)
        changed: dict[Symbol, Expr | None] = {}
        for path, raw in bindings.items():
            try:
                field = problem.field(*path.split("."))
            except KeyError as error:
                raise BindingError(f"Unknown field: {path}", (path,)) from error
            try:
                value = None if raw is None else FieldValue(field, raw).value
            except ModelInputError as error:
                raise BindingError(str(error), (path,)) from error
            if field.symbol in changed and changed[field.symbol] != value:
                raise BindingError("Aliases must have the same input value.", (path,))
            changed[field.symbol] = value
            # Definitions and inferred component observations retain their meaning.
            constraints = [c for c in constraints if c.left != field.symbol or c.kind is not ConstraintKind.INPUT]
            if value is not None:
                bad = [
                    key
                    for key, expected in field.symbol.assumptions0.items()
                    if getattr(value, f"is_{key}", None) not in (None, expected)
                ]
                if value.is_finite is False or value.has(S.NaN, S.ComplexInfinity) or bad:
                    requirements = ", ".join(bad) or "finite scalar"
                    raise BindingError(f"{path}: value violates {requirements}.", (path,))
                constraints.append(ModelConstraint(0, field.path, ConstraintKind.INPUT, field.symbol, value, path))
        return replace(
            problem, constraints=tuple(replace(c, index=i) for i, c in enumerate(constraints)), candidates=()
        )

    def snapshot(
        self, bindings: Mapping[str, ScalarValue | None] | None = None, *, raw: bool = False
    ) -> dict[str, Any]:
        """Return a JSON-ready view; invalid edits raise without changing state.

        Bindings replace initial inputs, and ``None`` releases an input. Raw
        views never invoke a solver. Bound views retain unresolved conditions;
        an undecided domain is clearly labelled as a conditional preview.
        """
        problem = self._bind(self._initial, bindings or {})
        values, verdict = ({}, "raw") if raw else _preview(problem)
        definitions = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.DEFINITION}
        supplied = {c.left: c.right for c in problem.constraints if c.kind is ConstraintKind.INPUT}
        inferred = {c.left for c in problem.constraints if c.kind is ConstraintKind.INFERRED}
        rows = []
        for field in problem.fields:
            path = ".".join(field.path)
            if self._hidden(path):
                continue
            info = self.metadata[path]
            expression = definitions.get(field.symbol, field.symbol) if raw else values[field.symbol]
            kind = (
                "input"
                if field.symbol in supplied
                else "fixed"
                if field.symbol in inferred or not expression.free_symbols
                else "free"
                if expression == field.symbol
                else "derived"
            )
            assumptions = _assumptions(field.symbol)
            rows.append(
                {
                    "path": path,
                    "name": field.path[-1],
                    "depth": len(field.path) - 1,
                    "title": info.title or field.path[-1].replace("_", " "),
                    "description": info.description,
                    "info": info.info,
                    "symbol": info.symbol or path,
                    "kind": kind,
                    "value": problem.format(expression),
                    "input": problem.format(supplied[field.symbol]) if field.symbol in supplied else "",
                    "editable": field.symbol not in inferred,
                    "assumptions": assumptions,
                    "constraints": [
                        {
                            "kind": str(c.kind),
                            "path": ".".join(c.path),
                            "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
                        }
                        for c in problem.uses(field)
                        if not self._hidden(".".join(c.path))
                    ],
                    "aliases": [".".join(f.path) for f in problem.aliases(field) if not self._hidden(".".join(f.path))],
                }
            )
        components = []
        for path, info in self.metadata.items():
            if path and path not in {".".join(f.path) for f in problem.fields} and not self._hidden(path):
                components.append(
                    {"path": path, "title": info.title or path.split(".")[-1], "description": info.description}
                )
        equations = [
            {
                "path": ".".join(c.path),
                "mathml": '<math xmlns="http://www.w3.org/1998/Math/MathML" display="block">'
                + re.sub(
                    r"&([A-Za-z][A-Za-z0-9]+);",
                    lambda match: (
                        match[0]
                        if match[1] in {"amp", "lt", "gt", "quot", "apos"}
                        else "".join(f"&#{ord(char)};" for char in html5[match[1] + ";"])
                    ),
                    self._printer.doprint(Eq(c.left, c.right, evaluate=False)),
                )
                + "</math>",
                "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
            }
            for c in self.problem.constraints
            if c.kind is ConstraintKind.DEFINITION and not self._hidden(".".join(c.path))
        ]
        root = self.metadata[""]
        return {
            "title": root.title or problem.model.__name__,
            "name": problem.name,
            "description": root.description,
            "info": root.info,
            "raw": raw,
            "fields": rows,
            "components": components,
            "equations": equations,
            "symbols": [
                {
                    "path": ".".join(f.path),
                    "title": self.metadata[".".join(f.path)].title or f.path[-1],
                    "description": self.metadata[".".join(f.path)].description,
                    "info": self.metadata[".".join(f.path)].info,
                    "value": "Hidden input" if f.symbol in supplied else "Hidden quantity",
                    "kind": "input" if f.symbol in supplied else "derived",
                    "assumptions": _assumptions(f.symbol),
                    "constraints": [],
                    "aliases": [],
                }
                for f in problem.fields
                if self._hidden(".".join(f.path))
            ],
            "status": "raw" if raw else "checked" if verdict == "satisfied" else "conditional",
            "condition": ""
            if raw
            else "All equations and domains checked"
            if verdict == "satisfied"
            else "Symbolic or undecided: not a validated numeric state",
            "hidden_inputs": sum(
                1 for c in problem.constraints if c.kind is ConstraintKind.INPUT and self._hidden(".".join(c.path))
            ),
        }


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
    app = Flask(__name__, template_folder="_gui", static_folder="_gui", static_url_path="/assets")
    app.config.update(MAX_CONTENT_LENGTH=16_384, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])

    @app.get("/")
    def index() -> str:
        return render_template("index.html")

    @app.post("/api/explore")
    def explore() -> Any:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or set(payload) - {"bindings", "raw"}:
            return jsonify(error="Expected bindings and an optional raw flag.", paths=[]), 400
        bindings = payload.get("bindings", {})
        raw = payload.get("raw", False)
        if not isinstance(bindings, dict) or not isinstance(raw, bool) or len(bindings) > len(explorer.problem.fields):
            return jsonify(error="Invalid bindings or raw flag.", paths=[]), 400
        parsed: dict[str, Expr | None] = {}
        try:
            for path, value in bindings.items():
                if path not in explorer.metadata or explorer._hidden(path):
                    raise BindingError("This field cannot be edited.", (path,))
                field = (
                    explorer.problem.field(*path.split("."))
                    if any(".".join(f.path) == path for f in explorer.problem.fields)
                    else None
                )
                if field is not None and any(
                    c.kind is ConstraintKind.INFERRED and c.left == field.symbol for c in explorer.problem.constraints
                ):
                    raise BindingError("A copied inferred observation is fixed in this snapshot.", (path,))
                if value is not None and not isinstance(value, str):
                    raise BindingError("Enter input values as text.", (path,))
                try:
                    parsed[path] = None if value is None or not value.strip() else _parse_scalar(value)
                except ModelInputError as error:
                    raise BindingError(str(error), (path,)) from error
            return jsonify(explorer.snapshot(parsed, raw=raw))
        except BindingError as error:
            return jsonify(error=str(error), paths=list(error.paths or parsed)), 422
        except (ModelInputError, ModelValidationError) as error:
            return jsonify(error=str(error), paths=list(parsed)), 422

    return app

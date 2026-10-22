"""Compile model presentation metadata independently of edits and solving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Mapping

from sympy import Symbol

from modello._gui.mathml import EquationPrinter
from modello.diagnostics import ConstraintKind, ModelField, ModelProblem
from modello.errors import ModelInputError
from modello.gui import FieldInfo, ModelInfo, Scenario

if TYPE_CHECKING:
    from modello import Modello


class Layout:
    """One index for names, aliases, visibility, edit policy, and raw equations."""

    def __init__(self, problem: ModelProblem, overrides: Mapping[str, FieldInfo]) -> None:
        self.fields = {".".join(field.path): field for field in problem.fields}
        self.metadata = dict.fromkeys(self.fields, FieldInfo())
        self.components: list[str] = []
        self._collect(problem.model)
        for path, info in overrides.items():
            self._set_info(path, info)

        hidden_roots = [path for path, info in self.metadata.items() if info.hidden]
        hidden = {
            path
            for path in self.metadata
            if any(not root or path == root or path.startswith(root + ".") for root in hidden_roots)
        }
        hidden_symbols = {field.symbol for path, field in self.fields.items() if path in hidden}
        hidden.update(path for path, field in self.fields.items() if field.symbol in hidden_symbols)
        self.hidden = frozenset(hidden)
        self.copied = frozenset(c.left for c in problem.constraints if c.kind is ConstraintKind.INFERRED)
        self.editable = frozenset(
            path for path, field in self.fields.items() if path not in hidden and field.symbol not in self.copied
        )
        self.aliases: dict[Symbol, list[str]] = {}
        for path, field in self.fields.items():
            self.aliases.setdefault(field.symbol, []).append(path)
        self.assumptions = {symbol: _assumptions(symbol) for symbol in self.aliases}
        self._labels = {
            symbol: (paths[0], self.metadata[paths[0]].symbol or paths[0]) for symbol, paths in self.aliases.items()
        }
        self._validate_navigation()
        printer = self.printer()
        self.equations = tuple(
            {
                "path": ".".join(c.path),
                "mathml": printer.equation(c.left, c.right),
                "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
            }
            for c in problem.constraints
            if c.kind is ConstraintKind.DEFINITION and ".".join(c.path) not in hidden
        )

    def _collect(self, model: type[Modello]) -> None:
        """Layer defaults, registry metadata, then overrides from children to parents."""
        annotated_models = ModelInfo.for_model(model)
        declarations: dict[str, ModelInfo] = {}
        for path, child in _model_tree(model):
            model_info = annotated_models.get(path, getattr(child, "__gui__", ModelInfo()))
            if not isinstance(model_info, ModelInfo):
                raise ModelInputError(f"{child.__name__}.__gui__ must be a ModelInfo declaration")
            declarations[path] = model_info
            self.metadata[path] = FieldInfo(
                title=model_info.title, description=model_info.description or child.__doc__ or "", info=model_info.info
            )
            if path:
                self.components.append(path)
        self.model_info = declarations[""]
        for path, info in FieldInfo.for_model(model).items():
            self._set_info(path, info)
        for prefix, declaration in reversed(declarations.items()):
            for relative_path, info in declaration.fields.items():
                if not relative_path:
                    raise ModelInputError("Model-level text belongs in ModelInfo, not an empty field path")
                self._set_info(_join(prefix, relative_path), info)

    def _set_info(self, path: str, info: FieldInfo) -> None:
        if path not in self.metadata or not isinstance(info, FieldInfo):
            raise ModelInputError(f"Invalid GUI metadata at {path!r}")
        if info.number_format not in {"number", "percent"}:
            raise ModelInputError(f"Invalid numeric format at {path!r}")
        self.metadata[path] = info

    def printer(self) -> EquationPrinter:
        """Give each render its own printer context, including concurrent requests."""
        return EquationPrinter(self._labels)

    def _validate_navigation(self) -> None:
        for path in self.model_info.highlights:
            self.field(path)
            if path in self.hidden:
                raise ModelInputError("A hidden quantity cannot be highlighted")
        if self.model_info.initial_field:
            self.field(self.model_info.initial_field)
        for scenario in self.model_info.scenarios:
            if not isinstance(scenario, Scenario):
                raise ModelInputError("Model scenarios must be Scenario declarations")
            if scenario.select:
                self.field(scenario.select)
            for path in scenario.bindings:
                if path not in self.editable:
                    raise ModelInputError(f"Scenario cannot edit {path!r}")

    def field(self, path: str) -> ModelField:
        """Resolve a leaf; component paths cannot be used as scalar inputs."""
        try:
            return self.fields[path]
        except KeyError as error:
            raise ModelInputError(f"Unknown field: {path}") from error


def _assumptions(symbol: Symbol) -> list[str]:
    """Omit predicates already implied by more informative requirements."""
    requirements = {k: v for k, v in symbol.assumptions0.items() if k != "commutative"}
    for key in sorted(requirements):
        others = {k: v for k, v in requirements.items() if k != key}
        if Symbol("requirements", **others).assumptions0.get(key) == requirements[key]:
            del requirements[key]
    return [key if expected else f"not {key}" for key, expected in sorted(requirements.items())]


def _join(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _model_tree(model: type[Modello], prefix: str = "") -> Iterator[tuple[str, type[Modello]]]:
    """Walk component types once, including components with no scalar leaves."""
    yield prefix, model
    for name, child in model._modello_namespace.nested_models.items():
        yield from _model_tree(child._modello_model, _join(prefix, name))

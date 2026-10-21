"""Compile model presentation metadata independently of edits and solving."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from sympy import Symbol

from modello._gui.mathml import EquationPrinter
from modello.diagnostics import ConstraintKind, ModelField, ModelProblem
from modello.errors import ModelInputError
from modello.gui import FieldInfo, ModelInfo

if TYPE_CHECKING:
    from modello import Modello


class Layout:
    """One index for names, aliases, visibility, edit policy, and raw equations."""

    def __init__(self, problem: ModelProblem, overrides: Mapping[str, FieldInfo]) -> None:
        self.fields = {".".join(field.path): field for field in problem.fields}
        self.metadata: dict[str, FieldInfo] = {}
        self.components: list[str] = []
        self._collect(problem.model, ())
        for path, info in overrides.items():
            self._set_info(path, info)

        hidden = {
            path
            for path in self.metadata
            if any(self.metadata[".".join(path.split(".")[:i])].hidden for i in range(len(path.split(".")) + 1))
        }
        hidden_symbols = {field.symbol for path, field in self.fields.items() if path in hidden}
        hidden.update(path for path, field in self.fields.items() if field.symbol in hidden_symbols)
        self.hidden = frozenset(hidden)
        self.copied = frozenset(c.left for c in problem.constraints if c.kind is ConstraintKind.INFERRED)
        self.editable = frozenset(
            path for path, field in self.fields.items() if path not in hidden and field.symbol not in self.copied
        )
        self.aliases: dict[Symbol, list[str]] = {}
        labels: dict[Symbol, tuple[str, str]] = {}
        for path, field in self.fields.items():
            self.aliases.setdefault(field.symbol, []).append(path)
            labels.setdefault(field.symbol, (path, self.metadata[path].symbol or path))
        self.assumptions = {symbol: _assumptions(symbol) for symbol in self.aliases}
        printer = EquationPrinter(labels)
        self.equations = tuple(
            {
                "path": ".".join(c.path),
                "mathml": printer.equation(c.left, c.right),
                "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
            }
            for c in problem.constraints
            if c.kind is ConstraintKind.DEFINITION and ".".join(c.path) not in hidden
        )

    def _collect(self, model: type[Modello], prefix: tuple[str, ...]) -> None:
        path = ".".join(prefix)
        declaration = getattr(model, "__gui__", ModelInfo())
        if not isinstance(declaration, ModelInfo):
            raise ModelInputError(f"{model.__name__}.__gui__ must be a ModelInfo declaration")
        self.metadata[path] = FieldInfo(
            title=declaration.title, description=declaration.description or model.__doc__ or "", info=declaration.info
        )
        for scalar in model._modello_namespace.dummies:
            self.metadata[".".join((*prefix, scalar))] = FieldInfo()
        for name, child in model._modello_namespace.nested_models.items():
            self.components.append(".".join((*prefix, name)))
            self._collect(child._modello_model, (*prefix, name))
        for relative_path, info in declaration.fields.items():
            if not relative_path:
                raise ModelInputError("Model-level text belongs in ModelInfo, not an empty field path")
            self._set_info(".".join((*prefix, *relative_path.split("."))), info)

    def _set_info(self, path: str, info: FieldInfo) -> None:
        if path not in self.metadata or not isinstance(info, FieldInfo):
            raise ModelInputError(f"Invalid GUI metadata at {path!r}")
        self.metadata[path] = info

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

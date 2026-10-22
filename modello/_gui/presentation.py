"""Compile model presentation metadata independently of edits and solving."""

from __future__ import annotations

from difflib import get_close_matches
from typing import TYPE_CHECKING, Iterator, Mapping

from sympy import Symbol

from modello._gui.mathml import EquationPrinter
from modello.diagnostics import ConstraintKind, ModelField, ModelProblem
from modello.errors import ModelInputError
from modello.gui import FieldInfo, ModelInfo, PlotInfo, Scenario

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
        # An unbound quantity copied from a component contributes only x = x;
        # it is still editable. Nontrivial recorded observations remain protected.
        self.copied = frozenset(
            c.left for c in problem.constraints if c.kind is ConstraintKind.INFERRED and c.left != c.right
        )
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
            if hasattr(child, "__gui__"):
                raise ModelInputError(
                    f"{child.__name__}.__gui__ is unsupported; use @ModelInfo(...) and gui(field, ...)"
                )
            model_info = annotated_models.get(path, ModelInfo())
            declarations[path] = model_info
            self.metadata[path] = FieldInfo(
                title=model_info.title, description=model_info.description or child.__doc__ or "", info=model_info.info
            )
            if path:
                self.components.append(path)
        self.model_info = declarations[""]
        for path, info in FieldInfo.for_model(model).items():
            self._set_info(path, info, context=f"gui({path!r})")
        for prefix, declaration in reversed(declarations.items()):
            for relative_path, info in declaration.fields.items():
                context = f"ModelInfo.fields[{relative_path!r}] on {prefix or model.__name__!r}"
                if not relative_path:
                    raise ModelInputError(
                        f"{context}: use ModelInfo for model-level text instead of an empty field path"
                    )
                self._set_info(_join(prefix, relative_path), info, context=context)

    def _set_info(self, path: str, info: FieldInfo, *, context: str = "metadata") -> None:
        _require_path(path, self.metadata, context)
        if not isinstance(info, FieldInfo):
            raise ModelInputError(f"{context}: expected FieldInfo at {path!r}")
        if info.number_format not in {"number", "percent"}:
            raise ModelInputError(f"{context}: invalid number_format at {path!r}; choose 'number' or 'percent'")
        self.metadata[path] = info

    def printer(self) -> EquationPrinter:
        """Give each render its own printer context, including concurrent requests."""
        return EquationPrinter(self._labels)

    def _validate_navigation(self) -> None:
        for path in self.model_info.highlights:
            self.field(path, context="ModelInfo.highlights")
            if path in self.hidden:
                raise ModelInputError(f"ModelInfo.highlights: hidden field {path!r} cannot be highlighted")
        if self.model_info.initial_field:
            self.field(self.model_info.initial_field, context="ModelInfo.initial_field")
        self._validate_plot(self.model_info.plot, "ModelInfo.plot")
        for index, scenario in enumerate(self.model_info.scenarios):
            if not isinstance(scenario, Scenario):
                raise ModelInputError(f"ModelInfo.scenarios[{index}]: expected a Scenario declaration")
            context = f"Scenario {scenario.title!r}"
            self._validate_plot(scenario.plot, f"{context}.plot")
            if scenario.select:
                self.field(scenario.select, context=f"{context}.select")
            for option, mapping in (("bindings", scenario.bindings), ("bounds", scenario.bounds)):
                for path in mapping:
                    self.field(path, context=f"{context}.{option}")
                    if path in self.hidden or (option == "bindings" and path not in self.editable):
                        source = "hidden" if path in self.hidden else "Recorded"
                        raise ModelInputError(f"{context}.{option}: cannot edit {source} field {path!r}")

    def _validate_plot(self, plot: PlotInfo | None, context: str) -> None:
        if plot is None:
            return
        if not isinstance(plot, PlotInfo):
            raise ModelInputError(f"{context}: expected PlotInfo")
        if not 1 <= len(plot.controls) <= 2 or len(set(plot.controls)) != len(plot.controls):
            raise ModelInputError(f"{context}: choose one or two distinct controls")
        fields = [self.field(path, context=context) for path in (*plot.controls, plot.output)]
        if any(path in self.hidden for path in (*plot.controls, plot.output)):
            raise ModelInputError(f"{context}: plotted quantities must be visible")
        if len({field.symbol for field in fields[:-1]}) != len(plot.controls):
            raise ModelInputError(f"{context}: choose distinct controls rather than aliases")
        if any(field.symbol.is_real is not True for field in fields[:-1]):
            raise ModelInputError(f"{context}: controls must be real quantities")
        if set(plot.windows) != set(plot.controls) or any(len(bounds) != 2 for bounds in plot.windows.values()):
            raise ModelInputError(f"{context}: supply two window endpoints for every control")
        if plot.view not in {"map", "surface"}:
            raise ModelInputError(f"{context}: view must be 'map' or 'surface'")

    def field(self, path: str, *, context: str = "bindings") -> ModelField:
        """Resolve a leaf; component paths cannot be used as scalar inputs."""
        _require_path(path, self.fields, context)
        return self.fields[path]


def _require_path(path: str, choices: Mapping[str, object], context: str) -> None:
    if path not in choices:
        suggestions = get_close_matches(path, choices, n=1)
        hint = f" Did you mean {suggestions[0]!r}?" if suggestions else ""
        raise ModelInputError(f"{context}: Unknown field path {path!r}.{hint}")


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

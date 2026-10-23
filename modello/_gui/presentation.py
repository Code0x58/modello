"""Compile model presentation metadata independently of edits and solving."""

from __future__ import annotations

from dataclasses import replace
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, Iterator, Mapping

from sympy import Symbol

from modello._gui.mathml import EquationPrinter
from modello._model import _known_integers
from modello._paths import _path_label, _within_path
from modello._symbols import _minimal_assumptions
from modello.diagnostics import ConstraintKind, ModelField, ModelProblem
from modello.errors import ModelInputError
from modello.gui import FieldInfo, ModelInfo, Scenario
from modello.vectors import _vector_input

if TYPE_CHECKING:
    from modello import Modello


class Layout:
    """One index for names, aliases, visibility, edit policy, and raw equations."""

    def __init__(
        self, problem: ModelProblem, overrides: Mapping[str, FieldInfo], *, resizable: frozenset[str] = frozenset()
    ) -> None:
        self.resizable = resizable
        internal = {
            _join(path, name)
            for path, child in _model_tree(problem.model)
            for name in getattr(child, "_modello_internal_vectors", ())
        }
        self.fields = {
            _path_label(field.path): field for field in problem.fields if _path_label(field.path[:-1]) not in internal
        }
        self.metadata = dict.fromkeys(self.fields, FieldInfo())
        self._explicit_metadata: set[str] = set()
        self.components: list[str] = []
        self.parents: dict[str, str] = {}
        self.collections: dict[str, Any] = {}
        self.recurrences: dict[str, tuple[str, Any, Any]] = {}
        self.folded: set[str] = set()
        self.vectors = {
            _path_label(path): vector for path, vector in problem.vectors.items() if _path_label(path) not in internal
        }
        self.vector_definitions = {_path_label(path): value for path, value in problem.vector_definitions.items()}
        self.templates = {_path_label(path): value for path, value in problem.templates.items()}
        self._collect(problem.model)
        for path, info in overrides.items():
            self._set_info(path, info)
        for path, vector in self.vectors.items():
            info = self.metadata[path]
            for i in range(len(vector)):
                coordinate = f"{path}[{i}]"
                if coordinate not in self._explicit_metadata:
                    self.metadata[coordinate] = replace(info, title=f"{info.title or path}[{i}]", symbol="")

        hidden_roots = [path for path, info in self.metadata.items() if info.hidden]
        hidden = {path for path in self.metadata if any(_within_path(path, root) for root in hidden_roots)}
        hidden_symbols = {field.symbol for path, field in self.fields.items() if path in hidden}
        hidden.update(path for path, field in self.fields.items() if field.symbol in hidden_symbols)
        self.hidden = frozenset(hidden)
        self.copied = frozenset(c.left for c in problem.constraints if c.kind is ConstraintKind.INFERRED)
        self.editable = frozenset(
            path for path, field in self.fields.items() if path not in hidden and field.symbol not in self.copied
        )
        self.editable_vectors = frozenset(
            path
            for path, vector in self.vectors.items()
            if (len(vector) or path in self.resizable)
            and all(f"{path}[{i}]" in self.editable for i in range(len(vector)))
        )
        self.aliases: dict[Symbol, list[str]] = {}
        for path, field in self.fields.items():
            self.aliases.setdefault(field.symbol, []).append(path)
        self.assumptions = {symbol: _assumptions(symbol) for symbol in self.aliases}
        self._labels = {
            symbol: (paths[0], self.metadata[paths[0]].symbol or paths[0]) for symbol, paths in self.aliases.items()
        }
        self._vector_labels = {
            vector: (path, self.metadata[path].symbol or path)
            for path, vector in reversed(sorted(self.vectors.items(), key=lambda item: item[0] in hidden))
        }
        self._coordinate_labels = {}
        self._template_labels = {
            template: (path, self.metadata[path].symbol or path) for path, template in self.templates.items()
        }
        for path, vector in reversed(self.vectors.items()):
            for i, symbol in enumerate(vector):
                coordinate = f"{path}[{i}]"
                if not self.metadata[coordinate].symbol and self.aliases[symbol][0] == coordinate:
                    self._coordinate_labels[symbol] = (coordinate, self.metadata[path].symbol or path, i)
        self._validate_navigation()
        printer = self.printer()
        self.equations = tuple(
            {
                "path": _path_label(c.path),
                "mathml": printer.equation(c.left, c.right),
                "text": f"{problem.format(c.left)} = {problem.format(c.right)}",
            }
            for c in problem.constraints
            if c.kind is ConstraintKind.DEFINITION and _path_label(c.path) not in hidden and not c.path[-1].isdecimal()
        )
        self.vector_equations = {
            path: {
                "path": path,
                "mathml": printer.vector_equation(
                    vector, self.vector_definitions[path], path=path, label=self.metadata[path].symbol or path
                ),
                "text": f"{path} = {self.vector_definitions[path]}",
            }
            for path, vector in self.vectors.items()
            if path in self.vector_definitions and path not in hidden
        }
        self.equations += tuple(self.vector_equations.values())
        self.template_equations = {
            path: {
                "path": path,
                "title": self.title(path),
                "mathml": printer.expression(template, display=True),
                "text": problem.format(template),
            }
            for path, template in self.templates.items()
            if template.equations and path not in hidden
        }
        self.equations += tuple(self.template_equations.values())
        relative = []
        for path, (prefix, namespace, spec) in self.recurrences.items():
            if path in hidden:
                continue
            model = self.collections[path]
            label = self.metadata[path].symbol or path
            known = _known_integers(namespace)
            initial = _join(prefix, _path_label(spec.initial.resolve(namespace, known)._modello_path))
            sources = {
                name: _join(prefix, _path_label(location.resolve(namespace, known)._modello_path))
                for name, location in spec.inputs.items()
            }
            inputs = tuple((self._component_label(source), name in spec.indexed) for name, source in sources.items())
            arguments = [
                f"{path}[n - 1]",
                *(f"{source}[n - 1]" if name in spec.indexed else source for name, source in sources.items()),
            ]
            transition = model._modello_transition.__name__ + "." + model._modello_result
            relative.append(
                {
                    "path": path,
                    "mathml": printer.model_recurrence(
                        label, self._component_label(initial), transition, inputs, int(spec.size - 1)
                    ),
                    "text": f"{path}[0] = {initial}; {path}[n] = {transition}({', '.join(arguments)})",
                }
            )
        self.equations = (*relative, *self.equations)

    def _component_label(self, path: str) -> str:
        if self.metadata[path].symbol:
            return self.metadata[path].symbol
        parent, _, index = path.rpartition("[")
        if path.endswith("]") and parent in self.collections:
            return f"{self.metadata[parent].symbol or parent}[{index[:-1]}]"
        return path

    def _collect(self, model: type[Modello]) -> None:
        """Layer defaults, registry metadata, then overrides from children to parents."""
        annotated_models = ModelInfo.for_model(model)
        declarations: dict[str, ModelInfo] = {}
        for path, child in _model_tree(model):
            for name, spec in child._modello_namespace.collections.items():
                if spec.wire is not None:
                    self.recurrences[_join(path, name)] = (path, child._modello_namespace, spec)
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
                self.parents[path] = _parent(path)
                if hasattr(child, "_modello_collection_size"):
                    self.collections[path] = child
                    if getattr(child, "_modello_collection_kind") == "sequence":
                        self.folded.update((_join(path, "initial"), _join(path, "steps")))
            for name in child._modello_namespace.vectors:
                vector_path = _join(path, name)
                if vector_path in self.vectors:
                    self.metadata[vector_path] = FieldInfo()
            for name in child._modello_namespace.templates:
                self.metadata[_join(path, name)] = FieldInfo()
        self.model_info = declarations[""]
        for path, info in FieldInfo.for_model(model).items():
            self._set_info(path, info, context=f"gui({path!r})")
        for component_path, declaration in reversed(declarations.items()):
            for relative_path, info in declaration.fields.items():
                context = f"ModelInfo.fields[{relative_path!r}] on {component_path or model.__name__!r}"
                if not relative_path:
                    raise ModelInputError(
                        f"{context}: use ModelInfo for model-level text instead of an empty field path"
                    )
                self._set_info(_join(component_path, relative_path), info, context=context)

    def _set_info(self, path: str, info: FieldInfo, *, context: str = "metadata") -> None:
        _require_path(path, self.metadata, context)
        if not isinstance(info, FieldInfo):
            raise ModelInputError(f"{context}: expected FieldInfo at {path!r}")
        if info.number_format not in {"number", "percent"}:
            raise ModelInputError(f"{context}: invalid number_format at {path!r}; choose 'number' or 'percent'")
        self.metadata[path] = info
        self._explicit_metadata.add(path)

    def printer(self) -> EquationPrinter:
        """Give each render its own printer context, including concurrent requests."""
        return EquationPrinter(self._labels, self._vector_labels, self._coordinate_labels, self._template_labels)

    def equation_path(self, path: tuple[str, ...], expression: object) -> str:
        """Prefer a complete law for a direct scalar evaluation of that law."""
        arguments = getattr(expression, "args", ())
        if arguments:
            for name, template in self.templates.items():
                if name in self.template_equations and template == arguments[0]:
                    return name
        return _path_label(path[:-1] if _path_label(path[:-1]) in self.vectors else path)

    def title(self, path: str) -> str:
        """Give coordinates a useful default name even with explicit blank metadata."""
        if self.metadata[path].title:
            return self.metadata[path].title
        field = self.fields.get(path)
        if field is not None and field.path[-1].isdecimal():
            return _path_label(field.path[-2:])
        return path.rpartition(".")[2].replace("_", " ")

    def _validate_navigation(self) -> None:
        quantities = self.fields | self.vectors
        for path in self.model_info.highlights:
            _require_path(path, quantities, "ModelInfo.highlights")
            if path in self.hidden:
                raise ModelInputError(f"ModelInfo.highlights: hidden field {path!r} cannot be highlighted")
        if self.model_info.initial_field:
            _require_path(self.model_info.initial_field, quantities, "ModelInfo.initial_field")
        for index, scenario in enumerate(self.model_info.scenarios):
            if not isinstance(scenario, Scenario):
                raise ModelInputError(f"ModelInfo.scenarios[{index}]: expected a Scenario declaration")
            context = f"Scenario {scenario.title!r}"

            def variable_coordinate(name: str) -> bool:
                return any(
                    name.startswith(path + "[") and name.endswith("]") and name[len(path) + 1 : -1].isdigit()
                    for path in self.resizable
                )

            variable_selection = variable_coordinate(scenario.select)
            if scenario.select and not variable_selection:
                _require_path(scenario.select, quantities, f"{context}.select")
            fixed_bindings = {}
            for path, raw in scenario.bindings.items():
                if path in self.resizable:
                    if path not in self.editable_vectors:
                        raise ModelInputError(f"{context}.bindings: cannot edit vector {path!r}")
                elif path not in self.fields and variable_coordinate(path):
                    # A scenario restores its original observation window before
                    # applying edits, which may include coordinates absent here.
                    continue
                else:
                    fixed_bindings[path] = raw
            for path, _ in self.expand_bindings(fixed_bindings):
                self.field(path, context=f"{context}.bindings")
                if path not in self.editable:
                    source = "hidden" if path in self.hidden else "Recorded"
                    raise ModelInputError(f"{context}.bindings: cannot edit {source} field {path!r}")

    def expand_bindings(self, bindings: Mapping[str, object]) -> Iterator[tuple[str, object]]:
        """Expand whole-vector observations at the same boundary as scalar paths."""
        for path, raw in bindings.items():
            if path in self.vectors:
                size = len(self.vectors[path])
                entries = (None,) * size if raw is None else _vector_input(raw, size, path)
                yield from ((f"{path}[{i}]", entry) for i, entry in enumerate(entries))
            else:
                yield path, raw

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
    requirements = _minimal_assumptions(symbol)
    return [key if expected else f"not {key}" for key, expected in sorted(requirements.items())]


def _join(prefix: str, name: str) -> str:
    if name.isdecimal():
        return f"{prefix}[{name}]"
    return prefix + ("" if name.startswith("[") or not prefix else ".") + name


def _parent(path: str) -> str:
    return path.rpartition("[")[0] if path.endswith("]") else path.rpartition(".")[0]


def _model_tree(model: type[Modello], prefix: str = "") -> Iterator[tuple[str, type[Modello]]]:
    """Walk component types once, including components with no scalar leaves."""
    yield prefix, model
    for name, child in model._modello_namespace.nested_models.items():
        yield from _model_tree(child._modello_model, _join(prefix, name))

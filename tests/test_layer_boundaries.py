"""Core imports point downwards; research layers stay independent of presentation."""

import ast
import subprocess
import sys

import pytest


def test_core_imports_follow_the_dependency_layers(pytestconfig: pytest.Config) -> None:
    layers = {
        name: index
        for index, name in enumerate(
            (
                "modello.errors",
                "modello._paths",
                "modello._symbols",
                "modello._values",
                "modello._operators",
                "modello.vectors",
                "modello._evaluation",
                "modello._collections",
                "modello._model_laws",
                "modello._shapes",
                "modello.series",
                "modello._expressions",
                "modello._problem",
                "modello._model",
                "modello.model_vectors",
                "modello._inputs",
                "modello.solutions",
                "modello.diagnostics",
                "modello",
            )
        )
    }
    for module, level in layers.items():
        filename = "__init__.py" if module == "modello" else module.rsplit(".", 1)[1] + ".py"
        tree = ast.parse((pytestconfig.rootpath / "modello" / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                node.body = []
        for node in ast.walk(tree):
            dependencies: tuple[str, ...]
            if isinstance(node, ast.ImportFrom):
                dependency = node.module or ""
                if node.level:
                    dependency = "modello" + ("." + dependency if dependency else "")
                dependencies = (dependency,)
            elif isinstance(node, ast.Import):
                dependencies = tuple(alias.name for alias in node.names)
            else:
                continue
            for dependency in dependencies:
                if dependency in layers:
                    assert layers[dependency] < level, f"{module} imports the higher layer {dependency}"


def test_layer_imports(pytestconfig: pytest.Config) -> None:
    # A fresh process makes accidental imports visible regardless of test order.
    program = """
import sys
from typing import get_type_hints

import examples.data
assert not {'modello', 'sympy', 'examples.portfolio'} & sys.modules.keys()

import modello
assert "pandas" not in sys.modules
assert not any(name == 'modello.contrib' or name.startswith('modello.contrib.') for name in sys.modules)
assert not hasattr(modello, 'ItoProcess')
assert get_type_hints(modello.inspect_model)["value_map"] == (
    modello.ScalarValue | modello.VectorValue | modello.Modello | modello.ModelVectorValue[modello.Modello]
)
assert get_type_hints(modello.component)["return"].__bound__ is modello.Modello

import examples.symbolic
import examples.named_members
import examples.reconcile
assert not {'examples.portfolio', 'examples.reconciliation'} & sys.modules.keys()

import examples.portfolio.account
import examples.portfolio.reporting
assert not {
    'examples.portfolio.demo', 'examples.portfolio.report_html',
} & sys.modules.keys()

from modello.contrib.stochastic import ItoProcess
x = modello.Coordinate('x', real=True)
t = modello.Coordinate('t', real=True)
assert ItoProcess(x, 0, 1, t).drift(x**2).doit() == 1
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=pytestconfig.rootpath, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr

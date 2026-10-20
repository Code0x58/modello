"""Research layers stay independent of finance, tutorials, and presentation."""

import subprocess
import sys

import pytest


def test_layer_imports(pytestconfig: pytest.Config) -> None:
    # A fresh process makes accidental imports visible regardless of test order.
    program = """
import sys
import examples.compute.named_map
import examples.compute.series
import examples.compute.providers
assert not {'modello', 'sympy', 'examples.portfolio'} & sys.modules.keys()

import examples.compute.scalar_plan
assert not {'modello', 'examples.portfolio'} & sys.modules.keys()

import examples.symbolic
import examples.model_plan
import examples.named_members
assert 'examples.portfolio' not in sys.modules

import examples.portfolio.account
import examples.portfolio.reporting
assert not {
    'examples.portfolio.explicit', 'examples.portfolio.named',
    'examples.portfolio.demo', 'examples.portfolio.report_html',
} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=pytestconfig.rootpath, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr

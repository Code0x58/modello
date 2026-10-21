"""Research layers stay independent of finance, tutorials, and presentation."""

import subprocess
import sys

import pytest


def test_layer_imports(pytestconfig: pytest.Config) -> None:
    # A fresh process makes accidental imports visible regardless of test order.
    program = """
import sys
import examples.data
assert not {'modello', 'sympy', 'examples.portfolio'} & sys.modules.keys()

import examples.symbolic
import examples.named_members
import examples.reconcile
assert not {'examples.portfolio', 'examples.reconciliation'} & sys.modules.keys()

import examples.portfolio.account
import examples.portfolio.reporting
assert not {
    'examples.portfolio.demo', 'examples.portfolio.report_html',
} & sys.modules.keys()
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=pytestconfig.rootpath, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stdout + result.stderr

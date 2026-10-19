"""Import boundaries keep the research layers independent of the installed core."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "program",
    [
        """
import sys
import modello
assert not any(name == 'examples' or name.startswith('examples.') for name in sys.modules)
""",
        """
import sys
from datetime import date
from examples.compute.named_map import NamedMap
from examples.compute.series import Series
series = Series((date(2026, 1, 1), date(2026, 1, 2)), (100, 110))
assert series.lag().values == (100,)
assert NamedMap({'price': series}).map(lambda item: item.values[-1]) == {'price': 110}
assert not any(name.split('.')[0] in {'modello', 'sympy'} for name in sys.modules)
assert 'examples.portfolio' not in sys.modules
""",
        """
import sys
from sympy import Symbol
from examples.compute.named_map import NamedMap
from examples.compute.scalar_plan import ScalarPlan
price = Symbol('price')
plan = ScalarPlan(NamedMap({'price': price}), NamedMap({'value': 2 * price}))
assert plan.evaluate(NamedMap({'price': 3})) == {'value': 6}
assert not any(name == 'modello' or name.startswith('modello.') for name in sys.modules)
assert 'examples.portfolio' not in sys.modules
""",
        """
import sys
from examples.named_members import component_members
assert callable(component_members)
assert not any(name == 'examples.portfolio' or name.startswith('examples.portfolio.') for name in sys.modules)
""",
        """
import sys
import examples.portfolio
assert not any(name.split('.')[0] in {'modello', 'sympy'} for name in sys.modules)
assert not any(name.startswith('examples.portfolio.') for name in sys.modules)
""",
        """
import sys
import examples.portfolio.account
import examples.portfolio.reporting
assert 'examples.portfolio.explicit' not in sys.modules
assert 'examples.portfolio.demo' not in sys.modules
assert 'examples.portfolio.report_html' not in sys.modules
""",
    ],
    ids=("core", "numeric-data", "symbolic-execution", "named-models", "portfolio-namespace", "finance"),
)
def test_layer_imports(program: str, pytestconfig: pytest.Config) -> None:
    result = subprocess.run(
        [sys.executable, "-c", program],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr

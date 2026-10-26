"""Rebuild the sdist and exercise both wheels outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path

from packaging.version import Version

SMOKE_TEST = r"""
from importlib import metadata
from pathlib import Path
import re
import sys

import modello
from sympy import FiniteSet, Tuple, sqrt

assert Path(modello.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
distribution = metadata.distribution("modello")
assert distribution.metadata["License-Expression"] == "MIT AND OFL-1.1"
assert distribution.metadata.get_all("License-File")
assert distribution.metadata["Requires-Python"] == ">=3.10"
assert Path(modello.__file__).with_name("py.typed").is_file()
assert "examples" in distribution.metadata.get_all("Provides-Extra", ())
assert "gui" in distribution.metadata.get_all("Provides-Extra", ())
assert "pandas" not in distribution.metadata.get_all("Provides-Extra", ())
assert not any(requirement.lower().startswith(("pandas", "polars", "numpy"))
               for requirement in distribution.requires or ())
assert not Path(modello.__file__).with_name("contrib").joinpath("pandas.py").exists()
assert "pandas" not in sys.modules
assert "flask" not in sys.modules
from modello.gui import Explorer, FieldInfo, ModelInfo, gui
for name in ("templates/index.html", "static/explorer.css", "static/explorer.js",
             "static/fonts/STIXTwoMath-Regular.woff2", "static/fonts/OFL.txt"):
    assert Path(modello.__file__).with_name("_gui").joinpath(name).is_file()
assert "flask" not in sys.modules
assert not any(str(path).startswith("tests/") for path in distribution.files or ())
assert not any(name == "examples" or name.startswith("examples.") for name in sys.modules)
namespace = {}
readme = distribution.metadata["Description"]
blocks = re.findall(r"^```python\n(.*?)^```", readme, re.MULTILINE | re.DOTALL)
assert blocks
for source in blocks:
    exec(compile(source, "README.md", "exec"), namespace)
assert not any(name == "examples" or name.startswith("examples.") for name in sys.modules)
if sys.argv[1]:
    assert distribution.version == sys.argv[1], distribution.version

@ModelInfo(title="A triangle", highlights=("c",))
class RightAngleTriangle(modello.Modello):
    a = gui(modello.InstanceDummy("a", real=True, positive=True), title="Side A", unit="m")
    b = modello.InstanceDummy("b", real=True, positive=True)
    c = sqrt(a**2 + b**2)

assert Explorer(RightAngleTriangle).snapshot(raw=True)["raw"]
assert RightAngleTriangle("T", a=3, b=4).c == 5
assert RightAngleTriangle("T", b=4, c=5).a == 3
assert FieldInfo.for_model(RightAngleTriangle)["a"].unit == "m"
assert RightAngleTriangle.__modello_annotations__[ModelInfo][""].title == "A triangle"
assert "flask" not in sys.modules

problem = modello.inspect_model(RightAngleTriangle, "family", b=4)
family = problem.solutions()
observed = family.observe(c=5)
assert observed.values(family.symbols.a, family.symbols.c) == FiniteSet(Tuple(3, 5))
point = observed.one()
assert isinstance(point, RightAngleTriangle)
assert (point.a, point.b, point.c) == (3, 4, 5)

class Box(modello.Modello):
    base = modello.component(RightAngleTriangle)
    height = modello.InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c**2 + height**2)

box = Box("B", base=RightAngleTriangle("base", b=4), height=12, diagonal=13)
assert (box.base.a, box.base.b, box.base.c) == (3, 4, 5)
assert FieldInfo.for_model(Box)["base.a"].title == "Side A"
try:
    Box("B", base={"b": 4})
except TypeError:
    pass
else:
    raise AssertionError("Component mappings must be rejected")

first = RightAngleTriangle("T")
second = RightAngleTriangle("T")
assert first.a != second.a

class Window(modello.Modello):
    samples = modello.Vector("x", size=3, real=True)
    i = modello.Index("i")
    total = samples.sum(i)
    changes = modello.vector(samples[i] - samples[i - 1], (i, 1, 2))

window = Window("observed", samples=[1, None, 3], total=6)
assert window.samples == modello.Vector(1, 2, 3)
assert window.changes == modello.Vector(1, 1)
problem = modello.inspect_model(Window, "family", samples=[1, None, 3])
assert ("changes",) in problem.vector_definitions
assert problem.latex(problem.symbols.samples) == r"\vec{\mathrm{samples}}"
assert r"\sum" in problem.latex(problem.constraints[0].right)
family = problem.solutions().observe(total=6)
assert family.values(family.symbols.samples) == FiniteSet(Tuple(1, 2, 3))
assert family.latex(family.symbols.samples) == r"\vec{\mathrm{samples}}"
class Relative(modello.Modello):
    n = modello.Index("n", nonnegative=True)
    T = modello.Series("T", n, real=True)
    opening = modello.InstanceDummy("opening", real=True)
    law = modello.Recurrence(T[n], 2*T[n-1], initial={0: opening})
    closing = law.at(4)
assert Relative("inverse", closing=16).opening == 1
assert ("law",) in modello.inspect_model(Relative, "law").templates
t = modello.Coordinate("t", real=True)
x = modello.Coordinate("x", real=True)
from modello.contrib.stochastic import ItoProcess
assert ItoProcess(x,0,1,t).drift(x**2).doit() == 1
class Sized(modello.Modello):
    N = modello.InstanceDummy("N", integer=True, nonnegative=True)
    samples = modello.Vector("x", size=N, real=True)
    total = samples.sum()
assert Sized("sized", samples=[1,2,3]).N == 3
assert Sized("sized", samples=[1,2,3]).total == 6
class AccountState(modello.Modello):
    cash = modello.InstanceDummy("cash", real=True)
class Debit(modello.Modello):
    amount = modello.InstanceDummy("amount", real=True)
class Accounts(modello.Modello):
    N = modello.InstanceDummy("N", integer=True, nonnegative=True)
    fills = modello.model_vector(Debit, size=N)
    states = modello.model_vector(AccountState, size=N+1)
    n = modello.Index("n")
    law = modello.model_recurrence(states[n], cash=states[n-1].cash-fills[n-1].amount)
    opening = states[0]
    terminal = states[-1]
    balances = states.project(lambda state: state.cash)
accounts = Accounts("accounts", opening=AccountState("opening", cash=100),
                    fills=[Debit("first", amount=10), Debit("next", amount=20)])
assert accounts.N == 2
assert accounts.balances == modello.Vector(100, 90, 70)
assert accounts.terminal is accounts.states[-1]
assert Accounts("empty", opening=AccountState("opening", cash=100), fills=[]).terminal.cash == 100

from modello.contrib.validation import WindowValidator
class ObservedCash(modello.Modello):
    cash = modello.InstanceDummy("cash", real=True)
    debit = modello.InstanceDummy("debit", real=True)
class ObservedLedger(modello.Modello):
    N = modello.InstanceDummy("N", integer=True, nonnegative=True)
    rows = modello.model_vector(ObservedCash, size=N)
    n = modello.Index("n")
    law = modello.model_recurrence(rows[n], cash=rows[n-1].cash-rows[n].debit)
validator = WindowValidator(ObservedLedger, "rows")
assert (validator.window_size, validator.first_test_position) == (2, 1)
assert validator(((100, 0), (90, 10)), position=1).valid is True
audit = validator.audit(((100, 0), (90, 10)))
assert (audit.row_count, audit.valid_rows, audit.assumed_rows) == (2, 1, 1)
assert audit.valid is None and audit.first_invalid is None
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
vector_snapshot = Explorer(window).snapshot()
assert vector_snapshot["vectors"][0]["shape"] == [3]
assert "mover" in vector_snapshot["vectors"][0]["symbol_mathml"]
assert "flask" not in sys.modules
print(f"Validated {distribution.metadata['Name']} {distribution.version}")
"""

EXAMPLES_SMOKE_TEST = """
from importlib import resources
from pathlib import Path
import sys

import examples
from examples.geometry import RightAngleTriangle
from examples.data import NamedMap
from examples.gui_quickstart import ReturnModel
from examples.gallery.models import AccountSequence, LinearTrend
from examples.accounting_breaks import EVENTS, Ledger, accounting_rows
from modello.contrib.validation import WindowValidator
import modello
from examples.portfolio.models import Execution, HoldingState
from modello.gui import Explorer

assert Path(examples.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(examples.__file__).with_name("py.typed").is_file()
assert RightAngleTriangle("T", a=3, b=4).c == 5
assert NamedMap({"a": 2}).map(lambda value: value * 3) == {"a": 6}
assert ReturnModel("example", capital=100_000, expected_return="2/25").gain == 8_000
assert LinearTrend("line", times=[0, 1, 2, 3], observations=[1, 2, 2, 4], slope=1, intercept=1).squared_error == 1
sequence = AccountSequence("installed",
    states=modello.ModelObservations({0: HoldingState("opening", quantity=3, cash=1000)}),
    fills=[Execution("fill", quantity=2, price=100, fee=1)])
assert sequence.terminal.cash == 799
assert sequence.terminal is sequence.states[-1]
assert Explorer(ReturnModel, inputs={"capital": 100_000}).snapshot(raw=True)["raw"]
assert "flask" not in sys.modules
gallery = resources.files("examples.gallery")
assert gallery.joinpath("templates/gallery.html").is_file()
assert gallery.joinpath("static/gallery.css").is_file()
data = resources.files("examples.portfolio").joinpath("data")
for name in ("market.csv", "transactions.csv", "benchmark.csv"):
    assert data.joinpath(name).is_file(), name
ledger_audit = WindowValidator(Ledger, "rows").audit(accounting_rows(), labels=EVENTS)
assert ledger_audit.first_invalid.position == 2
assert ledger_audit.assumed_rows == 1 and ledger_audit.valid_rows == 2
breaks = [check for check in ledger_audit.first_invalid.checks if check.invalid]
assert len(breaks) == 1
assert (ledger_audit.first_invalid.index, breaks[0].field, breaks[0].residual) == ("sale", "cash", -1)
assert not {"pandas", "polars", "numpy"} & sys.modules.keys()
print("Validated installed example modules, type marker, and portfolio fixtures")
"""


GUI_SMOKE_TEST = r"""
from examples.gallery.app import create_gallery

client = create_gallery().test_client()
assert client.get("/").status_code == 200
assert b'data-explore-url="/sequence/api/explore"' in client.get("/sequence/").data
assert client.get("/assets/gallery.css").status_code == 200
assert client.get("/sequence/assets/fonts/STIXTwoMath-Regular.woff2").data.startswith(b"wOF2")
response = client.post("/sequence/api/explore", json={"bindings": {"terminal.cash": "908"}})
assert response.status_code == 200, response.json
assert next(field["value"] for field in response.json["fields"] if field["path"] == "fills[1].price") == "110"
print("Validated installed gallery, assets, and inverse account solve")
"""


def check_wheel(wheel: Path, directory: Path, requirements: Mapping[str, Path], expected_version: str) -> None:
    """Exercise the base library, examples, and GUI outside the checkout."""
    environment = directory / "environment"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)

    def install(extra: str) -> None:
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r", str(requirements[extra])],
            check=True,
        )
        artifact = f"{wheel}[{extra}]" if extra else str(wheel)
        subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", artifact], check=True)

    install("")
    subprocess.run(
        [str(python), "-I", "-c", SMOKE_TEST, expected_version],
        cwd=directory,
        check=True,
    )

    install("examples")
    subprocess.run([str(python), "-I", "-c", EXAMPLES_SMOKE_TEST], cwd=directory, check=True)
    for module in (
        "examples.accounting_breaks",
        "examples.portfolio",
        "examples.portfolio.optimal_execution",
        "examples.solution_families",
        "examples.solution_views",
        "examples.applications",
        "examples.reconciliation",
        "examples.gallery",
    ):
        arguments = [str(python), "-I", "-m", module]
        if module == "examples.portfolio":
            arguments.extend(["--json", str(directory / "report.json"), "--html", str(directory / "report.html")])
        elif module == "examples.gallery":
            arguments.append("--help")
        subprocess.run(arguments, cwd=directory, check=True, stdout=subprocess.DEVNULL)
    assert json.loads((directory / "report.json").read_text())["snapshots"][-1]["nav"] == 3066.9
    assert (directory / "report.html").read_text().startswith("<!doctype html>")
    print(f"Validated all installed example runners from {wheel.name}", flush=True)
    install("gui")
    subprocess.run([str(python), "-I", "-c", GUI_SMOKE_TEST], cwd=directory, check=True)
    typing_checks = []
    for path in (
        "typing_checks.py",
        "tests/examples/test_named_members.py",
        "tests/examples/test_symbolic.py",
        "tests/examples/test_data.py",
        "tests/examples/reconcile_typing.py",
        "tests/examples/portfolio/typing_contracts.py",
    ):
        source = Path(__file__).resolve().parents[1] / path
        target = directory / source.name
        target.write_text(source.read_text())
        typing_checks.append(str(target))
    mypy_config = directory / "mypy.ini"
    mypy_config.write_text("[mypy]\nignore_missing_imports = true\n")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(mypy_config),
            "--python-executable",
            str(python),
            *typing_checks,
        ],
        cwd=directory,
        check=True,
    )


def main() -> None:
    """Validate the wheel and sdist, optionally matching a release tag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-version")
    arguments = parser.parse_args()
    expected_version = str(Version(arguments.expected_version)) if arguments.expected_version else ""
    if expected_version:
        version = Version(expected_version)
        if version.is_devrelease or version.local:
            parser.error("release tags must not contain development or local versions")

    directory = arguments.directory.resolve()
    wheels = list(directory.glob("*.whl"))
    sdists = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        parser.error("expected exactly one wheel and one source distribution")

    with tempfile.TemporaryDirectory(prefix="modello-distributions-") as temporary:
        scratch = Path(temporary)
        requirements = {extra: scratch / f"{extra or 'runtime'}-requirements.txt" for extra in ("", "examples", "gui")}
        for extra, target in requirements.items():
            options = ("--extra", extra) if extra else ()
            subprocess.run(
                [
                    "uv",
                    "export",
                    "--project",
                    str(Path(__file__).resolve().parents[1]),
                    "--locked",
                    "--no-dev",
                    "--no-emit-project",
                    *options,
                    "--output-file",
                    str(target),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
            )
        check_wheel(wheels[0], scratch / "wheel", requirements, expected_version)
        rebuilt = scratch / "rebuilt"
        subprocess.run(
            [
                "uv",
                "build",
                str(sdists[0]),
                "--wheel",
                "--no-build-isolation",
                "--python",
                sys.executable,
                "--out-dir",
                str(rebuilt),
            ],
            check=True,
        )
        rebuilt_wheels = list(rebuilt.glob("*.whl"))
        if len(rebuilt_wheels) != 1:
            parser.error("the source distribution must build exactly one wheel")
        check_wheel(rebuilt_wheels[0], scratch / "sdist", requirements, expected_version)


if __name__ == "__main__":
    main()

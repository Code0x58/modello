"""Rebuild the sdist and exercise both wheels outside the checkout."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.version import Version

SMOKE_TEST = r"""
from importlib import metadata
from pathlib import Path
import re
import sys

import modello
from sympy import sqrt

assert Path(modello.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
distribution = metadata.distribution("modello")
assert distribution.metadata["License-Expression"] == "MIT"
assert distribution.metadata.get_all("License-File")
assert distribution.metadata["Requires-Python"] == ">=3.10"
assert Path(modello.__file__).with_name("py.typed").is_file()
assert "examples" in distribution.metadata.get_all("Provides-Extra", ())
assert "gui" in distribution.metadata.get_all("Provides-Extra", ())
assert "flask" not in sys.modules
from modello.gui import Explorer
for name in ("index.html", "explorer.css", "explorer.js"):
    assert Path(modello.__file__).with_name("_gui").joinpath(name).is_file()
assert "flask" not in sys.modules
assert not any(str(path).startswith("tests/") for path in distribution.files or ())
assert not any(name == "examples" or name.startswith("examples.") for name in sys.modules)
namespace = {}
readme = distribution.metadata.get_payload()
blocks = re.findall(r"^```python\n(.*?)^```", readme, re.MULTILINE | re.DOTALL)
assert blocks
for source in blocks:
    exec(compile(source, "README.md", "exec"), namespace)
assert not any(name == "examples" or name.startswith("examples.") for name in sys.modules)
if sys.argv[1]:
    assert distribution.version == sys.argv[1], distribution.version

class RightAngleTriangle(modello.Modello):
    a = modello.InstanceDummy("a", real=True, positive=True)
    b = modello.InstanceDummy("b", real=True, positive=True)
    c = sqrt(a**2 + b**2)

assert Explorer(RightAngleTriangle).snapshot(raw=True)["raw"]
assert RightAngleTriangle("T", a=3, b=4).c == 5
assert RightAngleTriangle("T", b=4, c=5).a == 3

class Box(modello.Modello):
    base = modello.component(RightAngleTriangle)
    height = modello.InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c**2 + height**2)

box = Box("B", base=RightAngleTriangle("base", b=4), height=12, diagonal=13)
assert (box.base.a, box.base.b, box.base.c) == (3, 4, 5)
try:
    Box("B", base={"b": 4})
except TypeError:
    pass
else:
    raise AssertionError("Component mappings must be rejected")

first = RightAngleTriangle("T")
second = RightAngleTriangle("T")
assert first.a != second.a
print(f"Validated {distribution.metadata['Name']} {distribution.version}")
"""

EXAMPLES_SMOKE_TEST = """
from importlib import resources
from pathlib import Path
import sys

import examples
from examples.geometry import RightAngleTriangle
from examples.data import NamedMap

assert Path(examples.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(examples.__file__).with_name("py.typed").is_file()
assert RightAngleTriangle("T", a=3, b=4).c == 5
assert NamedMap({"a": 2}).map(lambda value: value * 3) == {"a": 6}
data = resources.files("examples.portfolio").joinpath("data")
for name in ("market.csv", "transactions.csv", "benchmark.csv"):
    assert data.joinpath(name).is_file(), name
print("Validated installed example modules, type marker, and portfolio fixtures")
"""


def check_wheel(
    wheel: Path, directory: Path, requirements: Path, examples_requirements: Path, expected_version: str
) -> None:
    """Exercise the base library and the examples extra outside the checkout."""
    environment = directory / "environment"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True)
    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r", str(requirements)],
        check=True,
    )
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", str(wheel)], check=True)
    subprocess.run(
        [str(python), "-I", "-c", SMOKE_TEST, expected_version],
        cwd=directory,
        check=True,
    )

    subprocess.run(
        ["uv", "pip", "install", "--python", str(python), "--require-hashes", "-r", str(examples_requirements)],
        check=True,
    )
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", f"{wheel}[examples]"], check=True)
    subprocess.run([str(python), "-I", "-c", EXAMPLES_SMOKE_TEST], cwd=directory, check=True)
    for module in (
        "examples.portfolio",
        "examples.portfolio.optimal_execution",
    ):
        arguments = [str(python), "-I", "-m", module]
        if module == "examples.portfolio":
            arguments.extend(["--json", str(directory / "report.json"), "--html", str(directory / "report.html")])
        subprocess.run(arguments, cwd=directory, check=True, stdout=subprocess.DEVNULL)
    assert json.loads((directory / "report.json").read_text())["snapshots"][-1]["nav"] == 3066.9
    assert (directory / "report.html").read_text().startswith("<!doctype html>")
    print(f"Validated all installed portfolio runners from {wheel.name}", flush=True)
    typing_checks = []
    for path in (
        "typing_checks.py",
        "tests/examples/test_named_members.py",
        "tests/examples/test_symbolic.py",
        "tests/examples/test_data.py",
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
        requirements = scratch / "runtime-requirements.txt"
        examples_requirements = scratch / "examples-requirements.txt"
        for target, options in ((requirements, ()), (examples_requirements, ("--extra", "examples"))):
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
        check_wheel(wheels[0], scratch / "wheel", requirements, examples_requirements, expected_version)
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
        check_wheel(rebuilt_wheels[0], scratch / "sdist", requirements, examples_requirements, expected_version)


if __name__ == "__main__":
    main()

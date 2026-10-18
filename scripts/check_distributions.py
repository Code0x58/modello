"""Rebuild the sdist and exercise both wheels outside the checkout."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.version import Version

SMOKE_TEST = """
from importlib import metadata
from pathlib import Path
import sys

import modello
from sympy import sqrt

assert Path(modello.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
distribution = metadata.distribution("modello")
assert distribution.metadata["License-Expression"] == "MIT"
assert distribution.metadata.get_all("License-File")
assert distribution.metadata["Requires-Python"] == ">=3.10"
if sys.argv[1]:
    assert distribution.version == sys.argv[1], distribution.version

class RightAngleTriangle(modello.Modello):
    a = modello.InstanceDummy("a", real=True, positive=True)
    b = modello.InstanceDummy("b", real=True, positive=True)
    c = sqrt(a**2 + b**2)

assert RightAngleTriangle("T", a=3, b=4).c == 5
assert RightAngleTriangle("T", b=4, c=5).a == 3

class Box(modello.Modello):
    base = RightAngleTriangle
    height = modello.InstanceDummy("height", positive=True)
    diagonal = sqrt(base.c**2 + height**2)

box = Box("B", base={"b": 4}, height=12, diagonal=13)
assert (box.base.a, box.base.b, box.base.c) == (3, 4, 5)

first = RightAngleTriangle("T")
second = RightAngleTriangle("T")
assert first.a != second.a
print(f"Validated {distribution.metadata['Name']} {distribution.version}")
"""


def check_wheel(wheel: Path, directory: Path, requirements: Path, expected_version: str) -> None:
    """Install locked dependencies and one wheel, then exercise the public API."""
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
        subprocess.run(
            [
                "uv",
                "export",
                "--project",
                str(Path(__file__).resolve().parents[1]),
                "--locked",
                "--no-dev",
                "--no-emit-project",
                "--output-file",
                str(requirements),
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

"""Sphinx configuration shared by local, CI, and Read the Docs builds."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx

project = "modello"
author = "Oliver Bristow"
copyright = "2018, Oliver Bristow"
release = metadata.version("modello")
version = release

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_title = "modello"
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}

autodoc_member_order = "bysource"
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


def write_research_report(app: Sphinx, exception: Exception | None) -> None:
    """Publish the runnable example with each successful HTML documentation build."""
    if exception is None and app.builder is not None and app.builder.format == "html":
        output = Path(app.outdir) / "_static" / "account-research.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "examples.account_demo", "--html", str(output)],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )


def setup(app: Sphinx) -> None:
    app.connect("build-finished", write_research_report)

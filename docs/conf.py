"""Sphinx configuration shared by local, CI, and Read the Docs builds."""

import os
from importlib import metadata

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

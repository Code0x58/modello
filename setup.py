"""Script for testing and packaging this library."""
import codecs
import os
from textwrap import dedent

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

TEST_REQUIRES = [
    "flake8-black",
    "flake8-docstrings",
    "flake8-isort",
    "pytest-cov>=2.6.1",
    "pytest-ruff",
    "pytest-mypy",
    "pytest>=4.1",
]
setup(
    author="Oliver Bristow",
    author_email="github+pypi@oliverbristow.co.uk",
    name="modello",
    use_scm_version=True,
    install_requires=["sympy"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    description="sympy expressions in models",
    setup_requires=["setuptools_scm", "wheel", "pytest-runner"],
    tests_require=TEST_REQUIRES,
    extras_require={"test": TEST_REQUIRES},
    py_modules=["modello"],
    python_requires=">=3.8",
    license="MIT",
    classifiers=dedent(
        """
        Development Status :: 4 - Beta
        Intended Audience :: Developers
        Intended Audience :: Science/Research
        License :: OSI Approved :: MIT License
        Operating System :: OS Independent
        Programming Language :: Python :: 3
        Programming Language :: Python :: 3 :: Only
        Programming Language :: Python :: 3.8
        Programming Language :: Python :: 3.9
        Programming Language :: Python :: 3.10
        Programming Language :: Python :: 3.11
        Programming Language :: Python :: 3.12
    """
    )
    .strip()
    .split("\n"),
    keywords="symbolic modeling",
    url="https://github.com/Code0x58/modello/",
)

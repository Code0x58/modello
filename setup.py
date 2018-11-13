import codecs
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

TEST_REQUIRES = ["pytest>=3.9", "pytest-cov", "pytest-mypy"]
setup(
    author="Oliver Bristow",
    author_email="github+pypi@oliverbristow.co.uk",
    name="modello",
    use_scm_version=True,
    install_requires=["sympy"],
    long_description=long_description,
    description="sympy expressions in models",
    setup_requires=["setuptools_scm", "wheel", "pytest-runner"],
    tests_require=TEST_REQUIRES,
    extras_require={"test": TEST_REQUIRES},
    py_modules=["modello"],
    python_requires=">=3.3",
)

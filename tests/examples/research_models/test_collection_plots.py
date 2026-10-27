"""Offline figures preserve application plotting state and bound source data."""

import subprocess
import sys

import pytest

from examples.research_models._plots import write_figure
from examples.research_models._study import Computation, Study, Variant


def test_import_keeps_numerical_and_plotting_dependencies_optional():
    program = (
        "import sys; import examples.research_models._plots; "
        "assert all(name not in sys.modules for name in ('numpy', 'scipy', 'matplotlib'))"
    )
    subprocess.run([sys.executable, "-c", program], check=True, capture_output=True, text=True)


def test_unsupported_key_is_rejected_before_writing(tmp_path):
    study = Study("not-a-model", "Unsupported", "", (), {})
    target = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="Unsupported study key"):
        write_figure(study, target)
    assert not target.exists()


def test_markowitz_artifacts_use_bound_results_and_preserve_plotting_state(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    pyplot = pytest.importorskip("matplotlib.pyplot")
    from examples.research_models.markowitz import allocate, portfolio_case

    # Changing the first variance changes the actual optimum to 10%. The
    # figure must read this result, rather than the collection's default data.
    case = portfolio_case(first_variance=9, second_variance=1)
    result = allocate(case)
    study = Study(
        "markowitz",
        "Bound covariance experiment",
        "An actual computed allocation, with changed covariance data.",
        (),
        {"changed_covariance": Variant(case, {"allocation": Computation(result)})},
    )
    with matplotlib.rc_context(
        {
            "backend": "Agg",
            "axes.facecolor": "#fff1e6",
            "axes.edgecolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "font.size": 13,
            "svg.hashsalt": "user-salt",
        }
    ):
        user_figure = pyplot.figure()
        user_axis = user_figure.subplots()
        user_axis.plot([0, 1], [1, 0])
        try:
            backend = matplotlib.get_backend()
            settings = matplotlib.rcParams.copy()
            figures = pyplot.get_fignums()
            files = write_figure(study, tmp_path)
            assert files == ("overview.svg", "overview.png")
            svg, png = (tmp_path / name for name in files)
            assert svg.stat().st_size > 1000
            assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            content = svg.read_text()
            assert study.title in content
            assert "v1=9, v2=1, cov=0" in content
            assert "10.0% / 90.0%" in content
            assert "Asset 1" in content and "Asset 2" in content
            assert matplotlib.get_backend() == backend
            assert matplotlib.rcParams == settings
            assert pyplot.get_fignums() == figures
            assert len(user_axis.lines) == 1
            # SVG identifiers and metadata are stable across the same replay.
            first_svg = svg.read_bytes()
            write_figure(study, tmp_path)
            assert svg.read_bytes() == first_svg
            assert matplotlib.rcParams == settings
            assert pyplot.get_fignums() == figures
        finally:
            pyplot.close(user_figure)


def test_blocked_only_study_does_not_invent_a_curve(tmp_path):
    pytest.importorskip("matplotlib")
    from examples.research_models.markowitz import portfolio_case

    case = portfolio_case()
    study = Study("markowitz", "No certified allocation", "", (), {"uncomputed": Variant(case)})
    write_figure(study, tmp_path)
    content = (tmp_path / "overview.svg").read_text()
    assert "No fully specified successful variant" in content
    assert "v1=" not in content


def test_gaussian_option_curves_are_sampled_as_arrays(tmp_path):
    pytest.importorskip("matplotlib")
    from examples.research_models.black_scholes import dividend_free_method, study_case

    case = study_case(dividends=False)
    result = case.run(dividend_free_method())
    study = Study(
        "black-scholes",
        "Bound Gaussian curves",
        "",
        (),
        {"without_dividends": Variant(case, {"parity": Computation(result)})},
    )
    write_figure(study, tmp_path)
    content = (tmp_path / "overview.svg").read_text()
    assert "No fully specified successful variant" not in content
    assert "Without dividends" in content
    assert "European call" in content and "European put" in content


def test_unbounded_weight_domain_retains_the_short_position(tmp_path):
    pytest.importorskip("matplotlib")
    from sympy import Rational

    from examples.research_models.markowitz import PortfolioRestrictions, allocate, portfolio_case

    case = portfolio_case(first_variance=1, second_variance=4, covariance=Rational(3, 2))
    result = allocate(case, PortfolioRestrictions(long_only=False))
    assert result.data.feasible.start.is_finite is False
    assert result.data.other_weight < 0
    study = Study(
        "markowitz",
        "Short positions allowed",
        "",
        (),
        {"shorts_allowed": Variant(case, {"allocation": Computation(result)})},
    )
    write_figure(study, tmp_path)
    content = (tmp_path / "overview.svg").read_text()
    assert f"{float(result.data.weight):.1%} / {float(result.data.other_weight):.1%}" in content
    assert "No fully specified successful variant" not in content

"""Exact allocation, active constraints, and the scope of the KKT certificate."""

from dataclasses import FrozenInstanceError

import pytest
from sympy import AtomicExpr, FiniteSet, Interval, Rational, S

from examples.research_models.markowitz import (
    Covariance,
    ExpectedReturns,
    PortfolioRestrictions,
    TwoAssetPortfolio,
    allocate,
    numerical_reference,
    portfolio_case,
)
from modello import component
from modello.contrib.research import ResearchCase, ResearchError


def test_interior_solution_has_an_exact_global_certificate():
    result = allocate(portfolio_case())
    assert result.data.weight == S.One / 5
    assert result.data.variance == S(4) / 5
    assert result.data.other_weight == S(4) / 5
    assert result.data.stationarity_residual == result.data.completion_residual == 0
    assert result.data.minimizers == FiniteSet(S.One / 5)
    assert result.report.explain("global_minimum").verdict == "strictly true"
    assert result.report.explain("global_minimum/quadratic_completion").verdict == "strictly true"


def test_adding_and_removing_a_return_target_preserves_the_original_branch():
    restrictions = PortfolioRestrictions(target=Rational(7, 100), cap=Rational(4, 5))
    original = allocate(portfolio_case(), restrictions).data
    relaxed = restrictions.without("target")
    changed = allocate(portfolio_case(), relaxed).data
    assert original.weight == S(3) / 4
    assert original.expected_return == S(7) / 100
    assert original.active == ("minimum_return",)
    assert dict(original.multipliers)["minimum_return"] == S(275) / 2
    assert changed.weight == S.One / 5
    assert restrictions.target == S(7) / 100 and relaxed.target is None
    assert relaxed.cap == restrictions.cap
    assert relaxed.changes == ("Removed target.",)
    with pytest.raises(FrozenInstanceError):
        restrictions.target = None


def test_an_upper_concentration_bound_has_the_correct_multiplier_sign():
    result = allocate(portfolio_case(), PortfolioRestrictions(cap=Rational(1, 10)))
    assert result.data.weight == S.One / 10
    assert result.data.variance == S(17) / 20
    assert result.data.active == ("concentration_cap",)
    assert dict(result.data.multipliers)["concentration_cap"] == 1


def test_removing_no_short_restrictions_changes_the_optimum():
    case = portfolio_case(first_variance=1, second_variance=4, covariance=Rational(3, 2))
    restricted = PortfolioRestrictions()
    long_only = allocate(case, restricted).data
    unrestricted = allocate(case, restricted.without("long_only")).data
    assert long_only.weight == 1 and long_only.active == ("second_nonnegative",)
    assert unrestricted.weight == S(5) / 4 and unrestricted.other_weight == -S.One / 4
    assert unrestricted.variance < long_only.variance
    assert unrestricted.feasible == S.Reals


def test_a_lower_return_on_the_first_asset_reverses_the_target_inequality():
    case = portfolio_case(first_return=Rational(4, 100), second_return=Rational(8, 100))
    data = allocate(case, PortfolioRestrictions(target=Rational(78, 1000))).data
    assert data.weight == S.One / 20
    assert data.active == ("minimum_return",)
    assert data.expected_return == S(78) / 1000
    assert dict(data.multipliers)["minimum_return"] >= 0


@pytest.mark.parametrize(
    "case,restrictions,message",
    [
        (portfolio_case(), PortfolioRestrictions(target=Rational(9, 100)), "infeasible"),
        (portfolio_case(), PortfolioRestrictions(cap=-1), "infeasible"),
        (
            portfolio_case(first_return=Rational(4, 100), second_return=Rational(4, 100)),
            PortfolioRestrictions(target=Rational(5, 100)),
            "same expected return",
        ),
    ],
)
def test_infeasible_restrictions_retain_the_case_and_mathematical_report(case, restrictions, message):
    with pytest.raises(ResearchError, match=message) as error:
        allocate(case, restrictions)
    assert error.value.case is case
    assert error.value.report.checks


def test_equal_means_with_a_redundant_target_do_not_need_division_by_zero():
    case = portfolio_case(first_return=Rational(4, 100), second_return=Rational(4, 100))
    result = allocate(case, PortfolioRestrictions(target=Rational(3, 100)))
    assert result.data.weight == S.One / 5
    assert result.report.explain("global_minimum").verdict == "strictly true"


def test_singular_covariance_can_have_a_unique_budget_line_minimum():
    case = portfolio_case(first_variance=1, second_variance=4, covariance=-2)
    result = allocate(case)
    assert result.data.weight == S(2) / 3
    assert result.data.variance == 0
    assert result.report.explain("global_minimum/valid_covariance").verdict == "strictly true"


def test_flat_curvature_loses_uniqueness_without_losing_all_minimizers():
    case = portfolio_case(first_variance=1, second_variance=1, covariance=1)
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain("unique_curvature").verdict == "strictly false"
    result = allocate(case, unique=False)
    assert result.data.variance == 1
    assert result.data.minimizers == Interval(0, 1)
    assert result.report.explain("global_minimum").verdict == "strictly true"


def test_a_singleton_feasible_set_is_supported_by_both_methods():
    case = portfolio_case()
    restrictions = PortfolioRestrictions(target=Rational(8, 100))
    result = allocate(case, restrictions)
    assert result.data.feasible == FiniteSet(1)
    assert result.data.minimizers == FiniteSet(1)
    assert numerical_reference(case, restrictions)["weight"] == 1.0


def test_invalid_covariance_blocks_the_solver_before_it_derives_a_candidate():
    case = portfolio_case(covariance=3)
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain("valid_covariance").verdict == "strictly false"


def test_a_changed_variance_law_cannot_reuse_the_quadratic_shortcut():
    class FourthPower(TwoAssetPortfolio):
        variance = (
            TwoAssetPortfolio.moments.first * TwoAssetPortfolio.weight**2
            + TwoAssetPortfolio.moments.second * (1 - TwoAssetPortfolio.weight) ** 2
            + 2 * TwoAssetPortfolio.moments.cross * TwoAssetPortfolio.weight * (1 - TwoAssetPortfolio.weight)
            + TwoAssetPortfolio.weight**4
        )

    case = ResearchCase.for_model(FourthPower).observe(
        moments=Covariance("moments", first=4, second=1, cross=0),
        means=ExpectedReturns("means", first=Rational(8, 100), second=Rational(4, 100)),
    )
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain("quadratic_variance_form").verdict == "strictly false"


@pytest.mark.parametrize(
    "restrictions",
    [
        PortfolioRestrictions(),
        PortfolioRestrictions(cap=Rational(1, 10)),
        PortfolioRestrictions(target=Rational(7, 100)),
    ],
)
def test_exact_solution_agrees_with_an_independent_numerical_minimizer(restrictions):
    case = portfolio_case()
    exact = allocate(case, restrictions).data
    numerical = numerical_reference(case, restrictions)
    assert numerical["weight"] == pytest.approx(float(exact.weight), abs=1e-7)
    assert numerical["variance"] == pytest.approx(float(exact.variance), abs=1e-10)
    assert numerical["kind"] == "numerical observation"


def test_restriction_validation_and_audits_distinguish_variants():
    with pytest.raises(ValueError, match="Unknown restriction"):
        PortfolioRestrictions().without("covariance_validity")
    with pytest.raises(ValueError, match="finite real"):
        PortfolioRestrictions(cap=S.Infinity)
    case = portfolio_case()
    plain = allocate(case).report.to_dict()
    capped = allocate(case, PortfolioRestrictions(cap=Rational(1, 10))).report.to_dict()
    assert plain["evidence_sha256"] != capped["evidence_sha256"]


def test_hidden_weight_assumptions_and_derived_observations_cannot_change_the_search_set():
    case = portfolio_case()
    weight = case.expression("weight", wrt="weight")
    for restricted in (case.assume(weight >= S(3) / 4), case.observe(variance=S(5) / 4)):
        with pytest.raises(ResearchError, match="PortfolioRestrictions"):
            allocate(restricted)


def test_a_false_determinant_cannot_certify_an_invalid_covariance():
    class FalsePSD(Covariance):
        determinant = S.One

    case = portfolio_case(covariance=-3).replace("moments", FalsePSD)
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain("determinant_form").verdict == "strictly false"


@pytest.mark.parametrize(
    "field,claim",
    [
        ("slope", "slope_form"),
        ("candidate_variance", "candidate_variance_form"),
        ("candidate_return", "candidate_return_form"),
        ("candidate_slope", "candidate_slope_form"),
    ],
)
def test_consumed_candidate_and_derivative_helpers_follow_the_actual_model(field, claim):
    from types import new_class

    def declare(namespace):
        namespace[field] = S(99)

    changed = new_class("ChangedHelper", (TwoAssetPortfolio,), exec_body=declare)
    case = ResearchCase.for_model(changed).observe(
        moments=Covariance("moments", first=4, second=1, cross=0),
        means=ExpectedReturns("means", first=Rational(8, 100), second=Rational(4, 100)),
    )
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain(claim).verdict != "strictly true"


def test_multiple_changed_helpers_cannot_fabricate_a_wrong_global_optimum():
    class WrongContrast(Covariance):
        contrast = S.One

    class WrongProof(TwoAssetPortfolio):
        moments = component(WrongContrast)
        candidate_slope = S.Zero
        completion_residual = S.Zero

    case = ResearchCase.for_model(WrongProof).observe(
        moments=WrongContrast("moments", first=4, second=1, cross=0),
        means=ExpectedReturns("means", first=Rational(8, 100), second=Rational(4, 100)),
    )
    with pytest.raises(ResearchError) as error:
        allocate(case)
    assert error.value.report.explain("contrast_form").verdict == "strictly false"


def test_an_undecidable_coefficient_sign_is_not_treated_as_zero():
    class UnknownReal(AtomicExpr):
        is_real = True
        is_finite = True

    case = portfolio_case(first_return=UnknownReal())
    with pytest.raises(ResearchError, match="sign.*coefficient.*unresolved"):
        allocate(case, PortfolioRestrictions(target=Rational(3, 100)))


@pytest.mark.parametrize("restriction", ["cap", "target"])
def test_undecidable_bound_ordering_retains_the_case_and_report(restriction):
    class UnknownReal(AtomicExpr):
        is_real = True
        is_finite = True

    case = portfolio_case()
    with pytest.raises(ResearchError, match="ordering.*restriction bound.*unresolved") as failure:
        allocate(case, PortfolioRestrictions(**{restriction: UnknownReal()}))
    assert failure.value.case is case
    assert failure.value.report.checks

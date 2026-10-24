"""Presentation and starting observations, separate from the model equations."""

from dataclasses import dataclass
from typing import Mapping, TypeAlias

from sympy import Rational

from examples.gallery.extensions import GeometricDiffusion
from examples.gallery.models import LinearTrend
from examples.geometry import RightAngleTriangle
from examples.nested_geometry import Box
from examples.portfolio.models import AccountSequence
from modello import Modello
from modello.gui import ExplorerValue, FieldInfo, ModelInfo, Scenario


@dataclass(frozen=True)
class Example:
    model: type[Modello]
    inputs: Mapping[str, ExplorerValue]
    info: ModelInfo


_Question: TypeAlias = tuple[str, Mapping[str, ExplorerValue | None], str]


def _questions(
    *,
    calculate: _Question,
    infer: _Question,
    change: _Question,
    unknown: _Question,
    check: _Question,
) -> tuple[Scenario, ...]:
    """Use the same progression to compare modelling behaviours across examples."""
    return tuple(
        Scenario(title=title, description=description, bindings=bindings, select=select)
        for title, (description, bindings, select) in (
            ("Calculate results", calculate),
            ("Infer missing inputs", infer),
            ("Try different inputs", change),
            ("Leave inputs unknown", unknown),
            ("Check the observations", check),
        )
    )


_FILLS = [{"quantity": 2, "price": 100, "fee": 1}, {"quantity": -1, "price": 110, "fee": 1}]
_UNPRICED_FILLS = [_FILLS[0], {"quantity": -1, "fee": 1}]


TUTORIAL = {
    "simple": Example(
        RightAngleTriangle,
        {"a": 3, "b": 4},
        ModelInfo(
            title="Scalar models",
            description="Name quantities, declare their domains, and solve the same equations in either direction.",
            info="Size a diagonal brace across a rectangular opening. All lengths must be positive.",
            initial_field="c",
            highlights=("a", "b", "c"),
            fields={
                "a": FieldInfo(title="Horizontal run", symbol="a", unit="cm"),
                "b": FieldInfo(title="Vertical rise", symbol="b", unit="cm"),
                "c": FieldInfo(title="Brace length", symbol="c", unit="cm"),
            },
            scenarios=_questions(
                calculate=(
                    "A 3 cm run and 4 cm rise need a 5 cm brace. Follow the equation to the calculated length.",
                    {},
                    "c",
                ),
                infer=(
                    "With an 8 cm rise and a 10 cm brace, how far can the support extend? Infer the 6 cm run.",
                    {"a": None, "b": 8, "c": 10},
                    "a",
                ),
                change=(
                    "Double the run and rise to 6 cm and 8 cm; the same relationship gives a 10 cm brace.",
                    {"a": 6, "b": 8},
                    "c",
                ),
                unknown=(
                    "Keep the rise at 4 cm and leave the run undecided. The brace length remains a formula.",
                    {"a": None},
                    "c",
                ),
                check=(
                    "A 4 cm brace conflicts with the measured 3 cm run and 4 cm rise. Release or "
                    "correct a measurement.",
                    {"c": 4},
                    "c",
                ),
            ),
        ),
    ),
    "components": Example(
        Box,
        {"base.a": 3, "base.b": 4, "height": 12},
        ModelInfo(
            title="Component models",
            description="Compose typed child models and solve their quantities together with the parent.",
            info=(
                "Check the diagonal clearance of a parcel with a 3 × 4 cm base and 12 cm "
                "height. The base reuses the triangle model."
            ),
            initial_field="diagonal",
            highlights=("base.c", "diagonal"),
            fields={
                "base": FieldInfo(title="Triangle component"),
                "base.a": FieldInfo(title="Base width", symbol="a", unit="cm"),
                "base.b": FieldInfo(title="Base depth", symbol="b", unit="cm"),
                "base.c": FieldInfo(title="Base diagonal", symbol="c", unit="cm"),
                "height": FieldInfo(title="Height", symbol="h", unit="cm"),
                "diagonal": FieldInfo(title="Box diagonal", symbol="d", unit="cm"),
            },
            scenarios=_questions(
                calculate=(
                    "What length fits along the parcel diagonal? The base diagonal is 5 cm "
                    "and the full clearance is 13 cm.",
                    {},
                    "diagonal",
                ),
                infer=(
                    "Specify exactly 13 cm diagonal clearance, with 4 cm depth and 12 cm height. "
                    "The shared equations infer a 3 cm width inside the base component.",
                    {"base.a": None, "diagonal": 13},
                    "base.a",
                ),
                change=(
                    "Double the parcel dimensions to 6 × 8 × 24 cm. Child and parent equations give 26 cm clearance.",
                    {"base.a": 6, "base.b": 8, "height": 24},
                    "diagonal",
                ),
                unknown=(
                    "Leave the parcel width undecided. Both the base and full diagonals retain that same unknown.",
                    {"base.a": None},
                    "diagonal",
                ),
                check=(
                    "A claimed 6 cm base diagonal conflicts with its 3 cm and 4 cm edges. Child "
                    "equations still constrain the parent.",
                    {"base.c": 6},
                    "base.c",
                ),
            ),
        ),
    ),
    "trend": Example(
        LinearTrend,
        {"times": [0, 1, 2, 3], "observations": [1, 2, 2, 4], "slope": 1, "intercept": 1},
        ModelInfo(
            title="Vector models",
            description="Let observations set vector sizes; combine indexed formulas, reductions, and derivatives.",
            info=(
                "Compare a proposed line with four sensor readings, then fit its "
                "coefficients. Least squares minimises error; it does not require noisy "
                "readings to lie on the line."
            ),
            initial_field="squared_error",
            highlights=("slope", "intercept", "squared_error"),
            fields={
                "N": FieldInfo(title="Measurement count", symbol="N"),
                "times": FieldInfo(title="Observation times", symbol="t"),
                "observations": FieldInfo(title="Observed levels", symbol="y"),
                "slope": FieldInfo(title="Slope", symbol="a"),
                "intercept": FieldInfo(title="Intercept", symbol="b"),
                "fitted": FieldInfo(title="Fitted levels", symbol="ŷ"),
                "residuals": FieldInfo(title="Residuals", symbol="e"),
                "squared_error": FieldInfo(title="Sum of squared errors", symbol="L"),
                "slope_gradient": FieldInfo(title="Slope gradient", symbol="Lₐ"),
                "intercept_gradient": FieldInfo(title="Intercept gradient", symbol="Lᵦ"),
            },
            scenarios=_questions(
                calculate=(
                    "Compare the proposed line y = 1 + t with four readings. Its residuals give a "
                    "sum of squared errors of 1.",
                    {},
                    "squared_error",
                ),
                infer=(
                    "Release both coefficients and set the objective gradients to zero. The "
                    "best-fit slope and intercept are both 0.9, with error 0.7.",
                    {"slope": None, "intercept": None, "slope_gradient": 0, "intercept_gradient": 0},
                    "slope",
                ),
                change=(
                    "Use six readings from a new calibration and refit. Their lengths set N = 6; "
                    "the fitted line is y = 1 + 2t.",
                    {
                        "times": [0, 1, 2, 3, 4, 5],
                        "observations": [1, 3, 5, 7, 9, 11],
                        "slope": None,
                        "intercept": None,
                        "slope_gradient": 0,
                        "intercept_gradient": 0,
                    },
                    "fitted",
                ),
                unknown=(
                    "Keep the intercept at 1 and leave the slope open. Predictions and error "
                    "remain formulas in that coefficient.",
                    {"slope": None},
                    "squared_error",
                ),
                check=(
                    "Claim zero error for the proposed line. Its residuals give error 1, so that "
                    "claim conflicts with the readings and coefficients.",
                    {"squared_error": 0},
                    "squared_error",
                ),
            ),
        ),
    ),
    "sequence": Example(
        AccountSequence,
        {"states[0].quantity": 3, "states[0].cash": 1000, "fills": _FILLS},
        ModelInfo(
            title="Recurrent models",
            description="Relate typed model states with relative indices and solve from boundary observations.",
            info=(
                "Start with 3 shares and USD 1,000. Buy 2 shares at USD 100, then sell 1 at "
                "USD 110, paying USD 1 per fill. Each fill connects two account boundaries."
            ),
            initial_field="terminal.cash",
            highlights=("terminal.quantity", "terminal.cash", "total_fees"),
            fields={
                "fills": FieldInfo(title="Executions", symbol="F"),
                "fills[]": FieldInfo(title="Execution", symbol="F"),
                "fills[].quantity": FieldInfo(title="Signed quantity", symbol="u", unit="shares"),
                "fills[].price": FieldInfo(title="Price", symbol="p", unit="USD/share"),
                "fills[].fee": FieldInfo(title="Fee", symbol="c", unit="USD"),
                "fills[].notional": FieldInfo(title="Trade value", symbol="v", unit="USD"),
                "states": FieldInfo(title="Account states", symbol="A"),
                "states[]": FieldInfo(title="Account state", symbol="A"),
                "states[].quantity": FieldInfo(title="Inventory", symbol="Q", unit="shares"),
                "states[].cash": FieldInfo(title="Cash", symbol="C", unit="USD"),
                "terminal": FieldInfo(title="Terminal account"),
                "terminal.quantity": FieldInfo(title="Terminal inventory", unit="shares"),
                "terminal.cash": FieldInfo(title="Terminal cash", unit="USD"),
                "total_fees": FieldInfo(title="Total fees", symbol="K", unit="USD"),
                "law": FieldInfo(title="Account recurrence"),
            },
            scenarios=_questions(
                calculate=(
                    "Post the two fills. The account closes with 4 shares, USD 908 cash, and USD 2 in total fees.",
                    {},
                    "terminal.cash",
                ),
                infer=(
                    "Leave the sale price unreported and observe USD 908 closing cash. The same "
                    "recurrence infers a USD 110 sale price.",
                    {"fills": _UNPRICED_FILLS, "terminal.cash": 908},
                    "fills[1].price",
                ),
                change=(
                    "Append a purchase of 1 share at USD 105 plus a USD 1 fee. N becomes 3; "
                    "closing cash is USD 802 and inventory is 5.",
                    {"fills": [*_FILLS, {"quantity": 1, "price": 105, "fee": 1}]},
                    "terminal.cash",
                ),
                unknown=(
                    "Leave the sale price unreported without a closing balance. Inventory is "
                    "known; cash remains a formula in the price.",
                    {"fills": _UNPRICED_FILLS},
                    "terminal.cash",
                ),
                check=(
                    "A statement showing USD 900 closing cash disagrees with the recorded fills, "
                    "which imply USD 908. Inspect the conflicting observations.",
                    {"terminal.cash": 900},
                    "terminal.cash",
                ),
            ),
        ),
    ),
}

EXTENSIONS = {
    "diffusion": Example(
        GeometricDiffusion,
        {"growth": Rational(1, 10), "volatility": Rational(1, 5), "opening": 100, "duration": 1, "wiener_increment": 0},
        ModelInfo(
            title="Itô diffusion",
            description="Use the stochastic contrib extension for local calculus and conditional moments.",
            info=(
                "One standard Wiener driver. A realised increment of zero gives a path value, "
                "distinct from the conditional mean."
            ),
            initial_field="log_drift",
            highlights=("growth", "volatility", "mean"),
            fields={
                "growth": FieldInfo(title="Growth coefficient", symbol="μ"),
                "volatility": FieldInfo(title="Diffusion coefficient", symbol="σ"),
                "opening": FieldInfo(title="Opening level", symbol="S₀"),
                "duration": FieldInfo(title="Elapsed time", symbol="τ"),
                "wiener_increment": FieldInfo(title="Realised Wiener increment", symbol="ΔW"),
                "log_drift": FieldInfo(title="Log-process drift", symbol="a"),
                "log_diffusion": FieldInfo(title="Log-process diffusion", symbol="b"),
                "closing": FieldInfo(title="Conditional closing level", symbol="S"),
                "mean": FieldInfo(title="Conditional mean", symbol="m"),
                "variance": FieldInfo(title="Conditional variance", symbol="v"),
                "process": FieldInfo(title="Itô process equation"),
            },
            scenarios=(
                Scenario(
                    title="Infer the process coefficients",
                    description="Infer growth and volatility from the transformed coefficients.",
                    bindings={
                        "growth": None,
                        "volatility": None,
                        "log_drift": Rational(2, 25),
                        "log_diffusion": Rational(1, 5),
                    },
                    select="growth",
                ),
                Scenario(
                    title="An empty time interval",
                    description="Zero duration and increment retain the opening state.",
                    bindings={"duration": 0, "wiener_increment": 0},
                    select="closing",
                ),
            ),
        ),
    ),
}

EXAMPLES = {**TUTORIAL, **EXTENSIONS}

# Display metadata is separate from the model equations.
for example in EXAMPLES.values():
    example.info(example.model)

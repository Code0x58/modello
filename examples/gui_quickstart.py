"""Run a minimal explorer with ``python -m examples.gui_quickstart``."""

from sympy import Rational

from modello import InstanceDummy, Modello
from modello.gui import ModelInfo, Scenario, create_app, gui


@ModelInfo(
    title="Return model",
    description="Explore return and expected gain.",
    initial_field="expected_return",
    highlights=("gain",),
    scenarios=(
        Scenario(
            title="8% return",
            description="Supply a return and calculate expected gain.",
            bindings={"expected_return": Rational(2, 25)},
            select="gain",
        ),
    ),
)
class ReturnModel(Modello):
    expected_return = gui(
        InstanceDummy("return", real=True),
        title="Expected return",
        symbol="μ",
        number_format="percent",
        description="The assumed mean annual arithmetic return.",
    )
    capital = gui(InstanceDummy("capital", positive=True), title="Starting capital", symbol="C", unit="USD")
    gain = gui(capital * expected_return, title="Expected gain", symbol="G", unit="USD")


if __name__ == "__main__":
    app = create_app(ReturnModel, inputs={"capital": 100_000})
    app.run(host="127.0.0.1", port=5058, debug=False)

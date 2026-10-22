"""Presentation for bounded calibration and coupled, three-coordinate reachability."""

from examples.projectile import Projectile
from examples.solution_families import ImpactStudy
from modello import SolutionFamily
from modello.gui import FieldInfo, ModelInfo


@ModelInfo(
    title="Execution cost lab",
    description="How much does uncertainty in a recorded trade tell us about the next trade?",
    info=(
        "An illustrative linear impact model. Recorded cost is between 200 and 202; "
        "the shared impact coefficient carries that uncertainty into the planned trade. "
        "These are possible values, with no probabilities attached."
    ),
    highlights=("impact", "recorded.cost", "planned.cost"),
    initial_field="planned.cost",
    fields={
        "impact": FieldInfo(title="Shared impact coefficient"),
        "recorded.cost": FieldInfo(title="Recorded execution cost", unit="USD"),
        "planned.cost": FieldInfo(title="Planned execution cost", unit="USD"),
    },
)
class CalibrationLab(ImpactStudy):
    """Calibrate one coefficient from a bounded observation, then bound a prediction."""


@ModelInfo(
    title="Reachability lab",
    description="Which targets can a ball launched at 20 m/s reach?",
    info=(
        "Uniform gravity of 10 m/s² and no air resistance. The envelope combines all launch directions. "
        "Its reachable region retains coupled constraints over three coordinates; "
        "coordinate count does not certify dimension. Preview x=0, y=0, target height=0, "
        "then vary one or two coordinates through that checked reference."
    ),
    highlights=("height", "clearance"),
    initial_field="height",
    fields={
        "x": FieldInfo(title="Forward distance", unit="m"),
        "y": FieldInfo(title="Sideways distance", unit="m"),
        "target_height": FieldInfo(title="Target height", unit="m"),
        "height": FieldInfo(title="Maximum reachable height", unit="m"),
        "clearance": FieldInfo(title="Height above target", unit="m"),
    },
)
class ReachabilityLab(Projectile):
    """A coupled set of targets above ground and beneath the trajectory envelope."""


def demo_family(name: str) -> SolutionFamily:
    """Build the demo from the example equations, retaining every family filter."""
    if name == "calibration":
        return CalibrationLab.calibrated()
    if name == "reachability":
        family = ReachabilityLab.envelope()
        flight = family.symbols
        return family.where(flight.target_height >= 0, flight.clearance >= 0)
    raise ValueError(f"Unknown family demo: {name!r}")

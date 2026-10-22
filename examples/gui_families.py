"""Guided nonlinear projectile views and bounded calibration."""

from examples.projectile import Projectile
from examples.solution_families import ImpactStudy
from modello import SolutionFamily
from modello.gui import FieldInfo, ModelInfo, PlotInfo, Scenario

PARABOLA = PlotInfo(controls=("x",), output="height", windows={"x": (-40, 40)})
REACHABLE_AREA = PlotInfo(
    controls=("x", "target_height"),
    output="clearance",
    windows={"x": (-40, 40), "target_height": (0, 20)},
)
PARABOLOID = PlotInfo(
    controls=("x", "y"),
    output="height",
    windows={"x": (-40, 40), "y": (-40, 40)},
    view="surface",
)


@ModelInfo(
    title="Projectile reachability lab",
    description="One throw speed. A curved boundary, a reachable area, and a surface of possibilities.",
    info=(
        "A ball leaves ground level at 20 m/s under gravity of 10 m/s², without air resistance. "
        "The same mechanics model used by the solution-view gallery combines every launch direction. "
        "Its highest reachable targets form a parabola in a vertical plane and a paraboloid in space. "
        "These are envelopes of many throws, rather than one ball’s trajectory. "
        "At this speed the envelope peaks at 20 m, and ground-level targets extend to 40 m from the launch point."
    ),
    highlights=("height", "clearance"),
    initial_field="height",
    plot=PARABOLA,
    scenarios=(
        Scenario(
            title="Safety parabola",
            description="At each horizontal distance, how high can any throw reach? Inspect the curved boundary.",
            select="height",
            plot=PARABOLA,
        ),
        Scenario(
            title="Reachable area",
            description="Targets above ground and beneath the parabola are reachable. Colour shows spare height.",
            bindings={"target_height": None},
            bounds={"target_height": (0, None), "clearance": (0, None)},
            select="clearance",
            plot=REACHABLE_AREA,
        ),
        Scenario(
            title="Safety paraboloid",
            description="Release sideways distance to see the curved ceiling over every reachable ground position.",
            bindings={"y": None},
            select="height",
            plot=PARABOLOID,
        ),
        Scenario(
            title="Targets 8–12 m high",
            description="Narrow the reachable area to an elevated band. Taller targets must be closer to the launch.",
            bindings={"target_height": None},
            bounds={"target_height": (8, 12), "clearance": (0, None)},
            select="target_height",
            plot=REACHABLE_AREA,
        ),
        Scenario(
            title="Boundary at 10 m",
            description="Observe an envelope height of 10 m. Two symmetric horizontal positions satisfy the equations.",
            bindings={"height": 10, "target_height": 10},
            select="x",
        ),
    ),
    fields={
        "x": FieldInfo(title="Horizontal distance", symbol="x", unit="m"),
        "y": FieldInfo(title="Sideways distance", symbol="y", unit="m"),
        "target_height": FieldInfo(title="Target height", symbol="z", unit="m"),
        "height": FieldInfo(
            title="Envelope height",
            description="The highest reachable target at this horizontal location, across all launch directions.",
            symbol="h",
            unit="m",
        ),
        "clearance": FieldInfo(title="Clearance above target", symbol="c", unit="m"),
        "gravity": FieldInfo(title="Gravity", unit="m/s²"),
        "free.speed": FieldInfo(title="Launch speed", unit="m/s"),
        "free.z": FieldInfo(title="Vertical aim coordinate", unit="m"),
        "free.time_squared": FieldInfo(title="Flight time squared", unit="s²"),
    },
)
class ProjectileLab(Projectile):
    """Use the same ballistic equations for a curve, constrained area, and curved surface."""


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
    if name == "projectile":
        family = ProjectileLab.envelope()
        return family.where(family.symbols.height >= 0).observe(y=0, target_height=0)
    if name == "reachability":
        family = ReachabilityLab.envelope()
        flight = family.symbols
        return family.where(flight.target_height >= 0, flight.clearance >= 0)
    raise ValueError(f"Unknown family demo: {name!r}")

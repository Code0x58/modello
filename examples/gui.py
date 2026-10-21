"""Run the optional browser explorer: ``uv run --extra gui python -m examples.gui``."""

from sympy import Rational

from modello import InstanceDummy, Modello, component
from modello.gui import FieldInfo, ModelInfo, create_app


class Room(Modello):
    """The dimensions and floor area of one room."""

    width = InstanceDummy("width", positive=True)
    length = InstanceDummy("length", positive=True)
    area = width * length
    __gui__ = ModelInfo(
        fields={
            "width": FieldInfo(description="The clear internal width, in metres.", symbol="w"),
            "length": FieldInfo(description="The clear internal length, in metres.", symbol="l"),
            "area": FieldInfo(description="The floor area, in square metres.", symbol="A"),
        }
    )


class Studio(Modello):
    """Explore how dimensions and material choices shape a small renovation."""

    room = component(Room)
    unit_cost = InstanceDummy("unit_cost", positive=True)
    waste = InstanceDummy("waste", nonnegative=True)
    material = room.area * (1 + waste)
    total = material * unit_cost
    __gui__ = ModelInfo(
        title="A room to explore",
        description=(
            "A small renovation, expressed as a connected model. Adjust a dimension or price and follow the effect."
        ),
        info=(
            "The waste allowance is a hidden input. It stays in the model as you explore; "
            "computed results stay derived."
        ),
        fields={
            "room": FieldInfo(title="Room geometry", description="Dimensions are in metres; area is in square metres."),
            "unit_cost": FieldInfo(title="Unit cost", description="Material price per square metre.", symbol="c"),
            "waste": FieldInfo(description="A 10% material allowance, held as an input.", symbol="r", hidden=True),
            "material": FieldInfo(
                title="Material required",
                description="Floor area plus the waste allowance, in square metres.",
                symbol="M",
            ),
            "total": FieldInfo(
                title="Estimated total", description="The material estimate before labour and tax.", symbol="C"
            ),
        },
    )


def main() -> None:
    """Open http://127.0.0.1:5058 after starting this local example."""
    app = create_app(
        Studio,
        "studio / renovation",
        inputs={"room.width": 4, "room.length": 5, "unit_cost": 32, "waste": Rational(1, 10)},
    )
    app.run(host="127.0.0.1", port=5058, debug=False)


if __name__ == "__main__":
    main()

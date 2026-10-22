"""The nested line example retains separate point instances."""


def test_line_length_preserves_distinct_point_instances():
    from examples.nested_line import Line, Point

    line = Line("L", start=Point("start", x=0, y=0), end=Point("end", x=3, y=4))
    assert line.length == 5
    assert isinstance(line.start, Point)
    assert line.start is not line.end

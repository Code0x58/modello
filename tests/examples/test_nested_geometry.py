"""Nested geometry supports inverse inference through a box base."""


def test_box_diagonal_can_infer_a_base_side():
    from examples.geometry import RightAngleTriangle
    from examples.nested_geometry import Box

    box = Box("B", base=RightAngleTriangle("base", b=4), height=12, diagonal=13)
    assert (box.base.a, box.base.b, box.base.c) == (3, 4, 5)

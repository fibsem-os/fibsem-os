from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import Point


def test_move_patterns_to_point_updates_all_stage_points_with_constant_offset() -> None:
    stage_1 = FibsemMillingStage(
        name="stage-1",
        pattern=RectanglePattern(point=Point(1e-6, 2e-6), width=1e-6, height=1e-6, depth=1e-6),
    )
    stage_2 = FibsemMillingStage(
        name="stage-2",
        pattern=RectanglePattern(point=Point(3e-6, 5e-6), width=1e-6, height=1e-6, depth=1e-6),
    )
    config = FibsemMillingTaskConfig(stages=[stage_1, stage_2])

    moved = config.move_patterns_to_point(Point(10e-6, 20e-6))

    assert moved is True
    assert config.stages[0].pattern.point == Point(10e-6, 20e-6)
    assert config.stages[1].pattern.point == Point(12e-6, 23e-6)


def test_move_patterns_to_point_returns_false_for_empty_stage_list() -> None:
    config = FibsemMillingTaskConfig(stages=[])

    moved = config.move_patterns_to_point(Point(10e-6, 20e-6))

    assert moved is False

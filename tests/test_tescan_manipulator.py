"""The Tescan manipulator methods return where the needle is afterwards.

Every backend's manipulator moves return the position after the move, as the
Manipulator device's commands do (the contract suite pins it on Demo and
LegacyDemo). These check Tescan's, which returned None, and its relative move,
which logged a failure and returned the exception instead of raising it.

No hardware or Tescan SDK required: the microscope is created without __init__
and its connection is a fake Nanomanipulator.
"""

import threading
from types import SimpleNamespace

import pytest

from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import BeamType, FibsemManipulatorPosition


class FakeNanomanipulator:
    Position = SimpleNamespace(Parking="parking", Standby="standby", Working="working")

    def __init__(self, fail_moves: bool = False):
        self.xyzr = [0.0, 0.0, 0.0, 0.0]  # mm, mm, mm, degrees
        self.fail_moves = fail_moves

    def IsCalibrated(self, index):
        return True

    def GetPosition(self, Index):
        return tuple(self.xyzr)

    def MoveTo(self, Index, X, Y, Z, Rot):
        if self.fail_moves:
            raise RuntimeError("the needle hit its limit")
        self.xyzr = [X, Y, Z, Rot]

    def MoveToPosition(self, Index, Position):
        self.xyzr = {"parking": [0, 0, 0, 0], "standby": [1, 2, 3, 0]}.get(
            Position, [4, 5, 6, 0]
        )


def _tescan(fail_moves: bool = False) -> TescanMicroscope:
    microscope = TescanMicroscope.__new__(TescanMicroscope)
    microscope._connection_lock = threading.RLock()
    microscope.connection = SimpleNamespace(
        Nanomanipulator=FakeNanomanipulator(fail_moves)
    )
    microscope.get_stage_position = lambda: SimpleNamespace(t=0.0)
    return microscope


def _xyz(position: FibsemManipulatorPosition):
    return pytest.approx([position.x, position.y, position.z])


@pytest.mark.parametrize(
    "move",
    [
        lambda m: m.insert_manipulator("Standby"),
        lambda m: m.retract_manipulator(),
        lambda m: m.move_manipulator_absolute(
            FibsemManipulatorPosition(x=1e-3, y=2e-3, z=3e-3, r=0)
        ),
        lambda m: m.move_manipulator_relative(
            FibsemManipulatorPosition(x=1e-3, y=0, z=0, r=0)
        ),
        lambda m: m.move_manipulator_corrected(1e-3, -1e-3, BeamType.ELECTRON),
    ],
    ids=["insert", "retract", "absolute", "relative", "corrected"],
)
def test_manipulator_moves_return_where_the_needle_is(move):
    microscope = _tescan()
    moved = move(microscope)
    assert isinstance(moved, FibsemManipulatorPosition)
    after = microscope.get_manipulator_position()
    assert _xyz(moved) == [after.x, after.y, after.z]


def test_a_failed_relative_move_raises():
    microscope = _tescan(fail_moves=True)
    with pytest.raises(RuntimeError, match="limit"):
        microscope.move_manipulator_relative(
            FibsemManipulatorPosition(x=1e-3, y=0, z=0, r=0)
        )


def test_home_reports_it_did_not_home():
    """Tescan's API cannot home; `home` returns False, not None, like the others' bool."""
    assert _tescan().home() is False

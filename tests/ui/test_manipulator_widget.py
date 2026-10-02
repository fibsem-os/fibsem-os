"""The manipulator widget builds from what the driver reports, not from its class.

The driver says which named positions the instrument has, which move types it
offers, and whether the arm rotates; the widget lists and shows exactly that.
ThermoFisher and Demo offer PARK/EUCENTRIC and corrected moves, Tescan its three
presets with relative moves and rotation, and a backend that says nothing gets
relative moves and the user's own saved positions.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""

import os
import threading
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem import utils
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import FibsemManipulatorPosition
from fibsem.ui.FibsemManipulatorWidget import FibsemManipulatorWidget


@pytest.fixture(scope="module", autouse=True)
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def demo():
    microscope, _ = utils.setup_session(manufacturer="Demo", setup_logging=False)
    microscope.insert_manipulator("PARK")
    return microscope


class StubMicroscope:
    """A backend the widget has never heard of: only the driver's answers."""

    def __init__(self, named=(), move_types=("relative",), rotation=False):
        self._named = list(named)
        self.manipulator_move_types = move_types
        self._rotation = rotation
        self.moves = []

    def manipulator_named_positions(self):
        return self._named

    def is_available(self, key):
        return key == "manipulator_rotation" and self._rotation

    def get_manipulator_state(self):
        return True

    def get_manipulator_position(self):
        return FibsemManipulatorPosition()

    def move_manipulator_to_named_position(self, name):
        self.moves.append(("named", name))
        return FibsemManipulatorPosition()

    def move_manipulator_absolute(self, position):
        self.moves.append(("absolute", position))
        return position


def _listed(widget):
    combo = widget.savedPosition_combobox
    return [combo.itemText(i) for i in range(combo.count())]


def test_demo_offers_park_eucentric_and_corrected_moves(demo):
    widget = FibsemManipulatorWidget(microscope=demo)

    assert _listed(widget) == ["PARK", "EUCENTRIC"]
    assert not widget.move_type_comboBox.isHidden()
    assert widget.dR_spinbox.isHidden()
    assert widget.beam_type_combobox.isHidden()  # relative move selected

    widget.move_type_comboBox.setCurrentText("Corrected Move")
    assert not widget.beam_type_combobox.isHidden()
    assert widget.dZ_spinbox.isHidden()


def test_device_demo_lists_its_manipulator_devices_named_positions():
    from tests.test_microscope_contract import _connect

    microscope = _connect("DeviceDemo")
    assert microscope.manipulator_named_positions() == ["PARK", "EUCENTRIC"]
    widget = FibsemManipulatorWidget(microscope=microscope)
    assert _listed(widget) == ["PARK", "EUCENTRIC"]


def test_demo_moves_to_a_named_position(demo):
    widget = FibsemManipulatorWidget(microscope=demo)
    widget.savedPosition_combobox.setCurrentText("EUCENTRIC")
    widget.move_to_saved_position()
    assert demo.get_manipulator_position() == demo._get_saved_manipulator_position(
        "EUCENTRIC"
    )


def test_tescan_like_offers_relative_moves_with_rotation():
    microscope = StubMicroscope(named=["Parking", "Standby", "Working"], rotation=True)
    widget = FibsemManipulatorWidget(microscope=microscope)

    assert _listed(widget) == ["Parking", "Standby", "Working"]
    assert widget.move_type_comboBox.isHidden()
    assert widget.beam_type_combobox.isHidden()
    assert not widget.dR_spinbox.isHidden()

    widget.savedPosition_combobox.setCurrentText("Working")
    widget.move_to_saved_position()
    assert microscope.moves == [("named", "Working")]


def test_a_backend_without_named_positions_lists_only_the_users():
    microscope = StubMicroscope()
    widget = FibsemManipulatorWidget(microscope=microscope)
    assert _listed(widget) == []

    widget.savedPositionName_lineEdit.setText("mine")
    widget.add_saved_position()
    widget.move_to_saved_position()
    assert _listed(widget) == ["mine"]
    assert microscope.moves[0][0] == "absolute"


def test_tescan_named_positions_are_insert_presets():
    microscope = TescanMicroscope.__new__(TescanMicroscope)
    microscope._connection_lock = threading.RLock()
    targets = []
    microscope.connection = SimpleNamespace(
        Nanomanipulator=SimpleNamespace(
            Position=SimpleNamespace(Parking="P", Standby="S", Working="W"),
            MoveToPosition=lambda Index, Position: targets.append(Position),
            GetPosition=lambda Index: (0.0, 0.0, 0.0, 0.0),
        )
    )

    assert microscope.manipulator_named_positions() == [
        "Parking",
        "Standby",
        "Working",
    ]
    assert microscope.manipulator_move_types == ("relative",)
    microscope.move_manipulator_to_named_position("Parking")
    assert targets == ["P"]

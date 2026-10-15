"""Any device drawn from what it describes about itself (`device_panel`), and the
configuration window's rows for devices beyond the fixed subsystems."""

import time

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.devices import Device, DeviceFault, Parameter, command  # noqa: E402
from fibsem.structures import RangeLimit  # noqa: E402
from fibsem.ui.widgets.device_panel import (  # noqa: E402
    DevicePanel,
    DevicesPanel,
    no_argument_commands,
)


class Unit(Device):
    gain = Parameter(float, limits=RangeLimit(0.0, 1.0))
    serial = Parameter(str)

    def __init__(self, name="unit"):
        super().__init__(name=name)
        self._gain, self.ran = 0.5, []

    def read_gain(self):
        return self._gain

    def write_gain(self, value):
        if value > 0.85:
            raise DeviceFault("the unit refused", device=self.name)
        self._gain = value

    def read_serial(self):
        return "U-1"

    def facts(self):
        return {"firmware": "1.2"}

    @command
    def zero(self) -> None:
        """Zero the unit."""
        self.ran.append("zero")

    @command
    def scan(self, lines: int) -> None:
        """Needs an argument: no button."""


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def test_a_command_needing_arguments_gets_no_button():
    assert set(no_argument_commands(Unit().connect())) == {"zero"}


def test_the_panel_shows_parameters_commands_and_facts(app):
    unit = Unit().connect()
    panel = DevicePanel(unit)

    assert set(panel.controls) == {"gain", "serial"}
    assert not panel.controls["serial"].widget.isEnabled()  # read only
    assert set(panel.buttons) == {"zero"}
    assert '"firmware": "1.2"' in panel.facts.toPlainText()


def test_an_edit_writes_and_a_refusal_says_why(app):
    unit = Unit().connect()
    panel = DevicePanel(unit)

    panel.controls["gain"].write(0.25)  # as the user's edit: the control signals
    assert unit.gain.get_value() == 0.25

    panel.controls["gain"].write(0.9)
    assert panel.status.text() == "gain: the unit refused"
    assert unit.gain.get_value() == 0.25
    assert panel.controls["gain"].read() == pytest.approx(0.25)  # shows what it is


def test_a_change_elsewhere_shows_in_the_panel(app):
    unit = Unit().connect()
    panel = DevicePanel(unit)

    unit.gain.set_value(0.75)
    app.processEvents()
    assert panel.controls["gain"].read() == pytest.approx(0.75)


def test_a_command_runs_off_the_gui_thread_and_reports(app):
    unit = Unit().connect()
    panel = DevicePanel(unit)

    panel.run("zero", confirm=False)
    deadline = time.monotonic() + 5
    while "done" not in panel.status.text():
        assert time.monotonic() < deadline
        app.processEvents()
        time.sleep(0.01)
    assert unit.ran == ["zero"]


def test_the_list_builds_each_panel_once(app):
    devices = DevicesPanel({"a": Unit("a").connect(), "b": Unit("b").connect()})
    first = devices.show_device("b")
    assert devices.show_device("b") is first
    assert devices.show_device("nothing") is None

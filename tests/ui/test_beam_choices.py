"""The widgets list a beam parameter's values from its device's choices."""

import pathlib
import re

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.structures import BeamType  # noqa: E402
from fibsem.ui.utils import beam_choices  # noqa: E402
from tests.test_microscope_contract import _connect, _plasma_configuration  # noqa: E402


def test_the_choices_follow_the_device_after_a_dependency_changes():
    """The ion currents change with the plasma gas, and the widgets see the new ones:
    the device keeps its choices up to date, so nothing here is cached to go stale."""
    microscope = _connect("Demo", _plasma_configuration())
    current = microscope.beams[BeamType.ION].current
    before = beam_choices(microscope, "current", BeamType.ION)
    assert before == list(current.choices)

    microscope.beams[BeamType.ION].plasma_gas.set_value("Argon")

    after = beam_choices(microscope, "current", BeamType.ION)
    assert after == list(current.choices) and after != before


def test_no_beam_or_no_parameter_has_no_choices():
    microscope = _connect("Demo")  # a Ga column: no plasma gas
    assert beam_choices(microscope, "plasma_gas", BeamType.ION) == []
    assert beam_choices(microscope, "not_a_parameter", BeamType.ION) == []
    microscope.beams = {BeamType.ELECTRON: microscope.beams[BeamType.ELECTRON]}
    assert beam_choices(microscope, "current", BeamType.ION) == []


def test_no_widget_asks_the_microscope_for_available_values():
    ui = pathlib.Path(__file__).parents[2] / "fibsem" / "ui"
    callers = [
        f"{path.relative_to(ui)}:{n}"
        for path in ui.rglob("*.py")
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"\bget_available_values(_cached)?\(", line)
    ]
    assert callers == []

"""The detector widget's mode list follows the detector type."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.devices.core import ParameterMetadata  # noqa: E402
from fibsem.structures import BeamType  # noqa: E402
from fibsem.ui.widgets.detector_settings_widget import (  # noqa: E402
    FibsemDetectorSettingsWidget,
)

_app = QApplication.instance() or QApplication([])

# The modes each type offers. The Demo offers the same modes for every type, so the
# test gives each its own, as the detector type's dependants are read again on a
# real instrument (AutoScript, Odemis).
MODES = {"ETD": ["SecondaryElectrons"], "TLD": ["BackscatteredElectrons", "EDS"]}


@pytest.fixture()
def widget(monkeypatch):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    mode = microscope.beams[BeamType.ELECTRON].detector_mode
    set_type = microscope.set_detector_type

    def set_detector_type(detector_type, beam_type):
        set_type(detector_type, beam_type)
        mode.metadata = ParameterMetadata(choices=MODES[detector_type])

    monkeypatch.setattr(microscope, "set_detector_type", set_detector_type)
    widget = FibsemDetectorSettingsWidget(
        microscope=microscope, beam_type=BeamType.ELECTRON
    )
    widget.populate_detector_combos()
    yield widget
    widget.close()
    microscope.disconnect()


def _modes(widget):
    return [widget.mode_combo.itemText(i) for i in range(widget.mode_combo.count())]


def test_choosing_a_type_lists_its_modes(widget):
    widget.type_combo.setCurrentText("TLD")
    assert _modes(widget) == MODES["TLD"]
    widget.type_combo.setCurrentText("ETD")
    assert _modes(widget) == MODES["ETD"]

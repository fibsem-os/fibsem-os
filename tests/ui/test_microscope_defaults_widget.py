"""The Defaults panel: read from the instrument, edit, save to the configuration."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.structures import MicroscopeSettings  # noqa: E402
from fibsem.ui.widgets.microscope_defaults_widget import (  # noqa: E402
    MicroscopeDefaultsWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def microscope(tmp_path):
    import yaml

    path = tmp_path / "site.yaml"
    path.write_text(
        yaml.safe_dump(
            utils.load_yaml(
                os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
            )
        )
    )
    microscope, _ = utils.setup_session(config_path=str(path), manufacturer="Demo")
    yield microscope
    microscope.disconnect()


@pytest.fixture()
def widget(qapp, microscope):
    w = MicroscopeDefaultsWidget()
    w.set_microscope(microscope)
    yield w
    w.close()
    w.deleteLater()


def test_disabled_until_there_is_a_microscope(qapp):
    w = MicroscopeDefaultsWidget()
    assert not w.isEnabled()
    w.deleteLater()


def test_the_form_shows_the_configured_defaults(widget, microscope):
    assert widget.electron.voltage.value() == microscope.system.electron.beam.voltage
    assert widget.ion.voltage.value() == microscope.system.ion.beam.voltage
    assert widget.electron.hfw.value() == pytest.approx(
        microscope.system.electron.beam.hfw * 1e6
    )


def test_reading_from_the_microscope_takes_the_live_values(widget, microscope):
    live = microscope.get_microscope_state().electron_beam.voltage
    microscope.system.electron.beam.voltage = (live or 0) + 4321  # stale
    widget.show_system()
    assert widget.electron.voltage.value() != live

    widget.read_from_microscope()

    assert microscope.system.electron.beam.voltage == live
    assert widget.electron.voltage.value() == live


def test_saving_writes_the_defaults_section_and_nothing_else(widget, microscope):
    path = microscope.configuration_path
    before = utils.load_yaml(path)
    widget.ion.voltage.set_value(8000)
    widget.electron.hfw.setValue(80.0)

    widget.save_to_configuration()

    written = utils.load_yaml(path)
    assert written["defaults"]["ion"]["voltage"] == 8000
    assert written["defaults"]["electron"]["hfw"] == pytest.approx(80.0e-6)
    assert written["hardware"] == before["hardware"]
    assert written["calibration"] == before["calibration"]
    assert written["defaults"]["imaging"] == before["defaults"]["imaging"]
    assert written["defaults"]["apply_on_connect"] is False
    assert utils.unrecognised_configuration_keys(written) == []
    reloaded = MicroscopeSettings.from_dict(written)
    assert reloaded.system.ion.beam.voltage == 8000


def test_saving_does_not_touch_the_column(widget, microscope):
    """Save decides what the defaults are; Apply is what pushes them."""
    live_before = microscope.get_beam_settings(microscope.system.ion.beam_type).voltage
    widget.ion.voltage.set_value(8000)

    widget.save_to_configuration()

    assert (
        microscope.get_beam_settings(microscope.system.ion.beam_type).voltage
        == live_before
    )

"""The configuration window: what the connected instrument is, read-only."""

import os

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.ui.widgets.microscope_configuration_window import (  # noqa: E402
    MicroscopeConfigurationWindow,
    devices_rows,
    session_sections,
    slot_rows,
)


@pytest.fixture
def microscope():
    path = os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml")
    microscope, _ = utils.setup_session(
        config_path=path, manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


def _texts(widget) -> str:
    from PyQt5.QtWidgets import QLabel, QTableWidget

    parts = [label.text() for label in widget.findChildren(QLabel)]
    for table in widget.findChildren(QTableWidget):
        for r in range(table.rowCount()):
            for c in range(table.columnCount()):
                item = table.item(r, c)
                if item is not None:
                    parts.append(item.text())
    return "\n".join(parts)


def test_the_window_has_its_tabs(qapp, microscope):
    window = MicroscopeConfigurationWindow(microscope)

    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        "Instrument",
        "Geometry",
        "Calibration",
        "Session",
    ]


def test_the_instrument_tab_shows_the_live_session(qapp, microscope):
    window = MicroscopeConfigurationWindow(microscope)
    text = _texts(window.tabs.widget(0))

    assert microscope.system.info.name in text
    assert "sim-iflm-configuration.yaml" in text
    assert "config/session/sim-iflm-configuration.yaml" in text


def test_the_geometry_tab_shows_the_configured_numbers(qapp, microscope):
    window = MicroscopeConfigurationWindow(microscope)
    text = _texts(window.tabs.widget(1))

    assert f"{microscope.system.ion.column_tilt:g}°" in text
    assert f"{microscope.system.ion.eucentric_height * 1e3:.2f} mm" in text
    fm_origin = microscope.system.stage.devices["FM"].origin.x
    assert f"{fm_origin * 1e3:.2f} mm" in text  # the plain position, not an offset


def test_devices_say_who_reported_them(microscope):
    """sim-iflm states `has_gis_multichem: false` -- the simulator's stand-in for the
    instrument answering -- and says nothing about the manipulator, so the backend's
    default answers for that one."""
    rows = {
        name: (fitted, source) for name, fitted, source, _ in devices_rows(microscope)
    }

    assert rows["Multichem"] == (False, "Instrument")
    assert rows["Manipulator"] == (True, "Backend default")
    assert rows["Fluorescence"] == (True, "Configuration")


def _fake_calibrated_slot(microscope):
    from fibsem.structures import FibsemStagePosition, SlotCalibration

    holder = microscope._stage.holder
    slot = holder.slots["Slot-01"]
    slot.position = FibsemStagePosition(
        name="Slot-01", x=-3.1e-3, y=0.4e-3, z=4e-3, r=0.0, t=0.61
    )
    slot.calibration = SlotCalibration(
        orientation="SEM",
        pre_tilt=holder.pre_tilt,
        rotation_reference=microscope.system.stage.rotation_reference,
        captured_at="2026-09-24T17:51:10",
        fibsem_version="0.5.2",
    )


def test_the_calibration_tab_shows_each_slot(qapp, microscope):
    _fake_calibrated_slot(microscope)

    rows = slot_rows(microscope._stage.holder)

    assert rows[0] == (
        "Slot-01",
        "Calibrated",
        "—",
        "x -3.10  y 0.40 mm",
        "2026-09-24 17:51:10",
    )
    assert rows[1][1] == "Not calibrated"
    window = MicroscopeConfigurationWindow(microscope)
    assert "1 of 2 calibrated" in _texts(window.tabs.widget(2))


def test_the_objective_values_are_shown_read_only(qapp, microscope):
    microscope.system.fm.focus_position = 8.0e-3
    microscope.system.fm.limit_position = None

    window = MicroscopeConfigurationWindow(microscope)
    text = _texts(window.tabs.widget(2))

    assert "8.00 mm" in text
    assert "Not calibrated" in text
    assert "Calibrate Objective" not in text


def test_saving_a_slot_calibration_refreshes_the_tab(qapp, microscope):
    window = MicroscopeConfigurationWindow(microscope)
    assert "0 of 2 calibrated" in _texts(window.tabs.widget(2))

    _fake_calibrated_slot(microscope)
    window.refresh_calibration(microscope._stage.holder)

    assert window.tabs.tabText(2) == "Calibration"
    assert "1 of 2 calibrated" in _texts(window.tabs.widget(2))


def test_a_loader_holder_has_no_slots_to_calibrate(qapp):
    from PyQt5.QtWidgets import QPushButton

    path = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    microscope, _ = utils.setup_session(
        config_path=path, manufacturer="Demo", setup_logging=False
    )
    try:
        window = MicroscopeConfigurationWindow(microscope)
        buttons = window.tabs.widget(2).findChildren(QPushButton, "calibrate_slots")
        assert microscope._stage.loader is not None
        assert buttons == []
        assert [row[1] for row in slot_rows(microscope._stage.holder)] == ["Built in"]
    finally:
        microscope.disconnect()


def test_each_session_section_shows_what_it_holds():
    sections = {
        title: rows
        for title, _, rows in session_sections(
            {
                "saved_positions": [
                    {"name": "cryo", "x": 1e-3, "y": -2e-3, "z": 0.0, "r": 0.0}
                ],
                "holder_occupancy": {"Slot-01": {"name": "grid-ash"}},
                "fm": {
                    "working": {
                        "channel_settings": [
                            {
                                "name": "GFP",
                                "excitation_wavelength": 488,
                                "emission_wavelength": 520,
                                "power": 0.05,
                                "exposure_time": 0.2,
                            }
                        ]
                    },
                    "recent_channels": [{"name": "Reflection"}],
                },
                "from_the_future": {"x": 1},
            }
        )
    }

    assert sections["Saved positions"] == [
        ("cryo", "1.00", "-2.00", "0.00", "0.0°", "—")
    ]
    assert sections["Grids in the holder"] == [("Slot-01", "grid-ash")]
    assert sections["Fluorescence channels"] == [
        ("GFP", "488 nm", "520 nm", "5.0 %", "200 ms")
    ]
    assert sections["Recent channels"] == [("Reflection", "—", "Reflection", "—", "—")]
    assert sections["Not used by this version"] == [("from_the_future", "dict")]


def test_an_empty_session_has_nothing_in_each_section():
    sections = session_sections({})

    assert [title for title, _, _ in sections] == [
        "Saved positions",
        "Grids in the holder",
        "Fluorescence channels",
        "Recent channels",
    ]
    assert all(rows == [] for _, _, rows in sections)


def test_the_calibration_tab_shows_the_grid_in_each_slot(qapp, microscope):
    from fibsem.structures import SampleGrid

    microscope._stage.assign_grid("Slot-01", SampleGrid(name="grid-ash"))

    assert slot_rows(microscope._stage.holder)[0][2] == "grid-ash"
    window = MicroscopeConfigurationWindow(microscope)
    assert "grid-ash" in _texts(window.tabs.widget(2))


def test_the_session_tab_shows_this_configuration_s_file(qapp, microscope):
    from fibsem.saved_positions import save_saved_positions
    from fibsem.session_state import session_state_for
    from fibsem.structures import FibsemStagePosition

    save_saved_positions(
        [FibsemStagePosition(name="cryo")], session_state_for(microscope, writable=True)
    )
    window = MicroscopeConfigurationWindow(microscope)
    text = _texts(window.tabs.widget(3))

    assert session_state_for(microscope).path.name in text
    assert "Saved positions  ·  1" in text
    assert "cryo" in text

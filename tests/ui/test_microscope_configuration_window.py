"""The configuration window: what the connected instrument is, and its defaults."""

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
        "Defaults",
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


def test_the_geometry_tab_shows_the_declared_fib_pose_rotation(qapp, microscope):
    """Read from the pose the stage declares (FIB-1101), not `stage.rotation_180`."""
    import math

    microscope.get_orientation("FIB").r = math.radians(123.0)
    window = MicroscopeConfigurationWindow(microscope)

    assert "123°  (derived)" in _texts(window.tabs.widget(1))


def test_devices_say_who_reported_them(microscope):
    """sim-iflm says nothing about the manipulator, so the backend's default answers
    for it."""
    rows = {
        name: (fitted, source) for name, fitted, source, _ in devices_rows(microscope)
    }

    assert rows["Manipulator"] == (True, "Backend default")
    assert rows["Fluorescence"] == (True, "Configuration")


@pytest.mark.parametrize(
    "configuration, detail",
    [
        ("sim-iflm-configuration.yaml", ""),
        ("sim-arctis-configuration.yaml", "No rotation"),
    ],
    ids=["rotating-stage", "compustage"],
)
def test_the_stage_row_says_when_the_stage_does_not_rotate(configuration, detail):
    """Read from the stage's axes at connect, not from the stage type."""
    path = os.path.join(cfg.CONFIG_PATH, configuration)
    microscope, _ = utils.setup_session(
        config_path=path, manufacturer="Demo", setup_logging=False
    )
    try:
        rows = {row[0]: row[3] for row in devices_rows(microscope)}
        assert rows["Stage"] == detail
    finally:
        microscope.disconnect()


def test_a_simulated_probe_is_reported_as_the_instrument(microscope):
    """`sim.has_manipulator` is the simulator's stand-in for the instrument
    answering."""
    microscope.system.sim["has_manipulator"] = False
    microscope._read_hardware_capabilities()

    rows = {
        name: (fitted, source) for name, fitted, source, _ in devices_rows(microscope)
    }

    assert rows["Manipulator"] == (False, "Instrument")


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
    text = _texts(window.tabs.widget(4))

    assert session_state_for(microscope).path.name in text
    assert "Saved positions  ·  1" in text
    assert "cryo" in text


# ---------------------------------------------------------------------------
# The Defaults tab: one Save for the window
# ---------------------------------------------------------------------------


@pytest.fixture
def toasts(monkeypatch):
    from fibsem.ui import notification_service

    shown = []
    monkeypatch.setattr(
        notification_service,
        "show_toast",
        lambda message, notification_type="info": shown.append(
            (notification_type, message)
        ),
    )
    return shown


@pytest.fixture
def site(tmp_path):
    """A copy of a shipped configuration, so Save writes somewhere disposable."""
    import shutil

    path = tmp_path / "site.yaml"
    shutil.copy(os.path.join(cfg.CONFIG_PATH, "sim-iflm-configuration.yaml"), path)
    microscope, _ = utils.setup_session(
        config_path=str(path), manufacturer="Demo", setup_logging=False
    )
    yield microscope
    microscope.disconnect()


def _edit(window, micrometres: float = 80.0) -> None:
    window.defaults.electron.hfw.setValue(micrometres)


def test_a_new_window_has_nothing_to_save(qapp, site, toasts):
    window = MicroscopeConfigurationWindow(site)

    assert not window.has_unsaved_changes()
    assert window.label_unsaved.text() == ""
    assert not window.pushButton_save.isEnabled()
    assert window.tabs.tabText(3) == "Defaults"


def test_an_edit_says_where_the_unsaved_change_is(qapp, site, toasts):
    window = MicroscopeConfigurationWindow(site)

    _edit(window)

    assert window.label_unsaved.text() == "Unsaved changes  ·  Defaults  ·  site.yaml"
    assert window.tabs.tabText(3) == "Defaults •"
    assert window.pushButton_save.isEnabled()


def test_save_writes_the_defaults_and_clears_the_change(qapp, site, toasts):
    window = MicroscopeConfigurationWindow(site)
    _edit(window)

    window.pushButton_save.click()

    written = utils.load_yaml(site.configuration_path)
    assert written["defaults"]["electron"]["hfw"] == pytest.approx(80.0e-6)
    assert not window.has_unsaved_changes()
    assert window.tabs.tabText(3) == "Defaults"


@pytest.mark.parametrize(
    "answer, closed, saved",
    [("Cancel", False, False), ("Discard", True, False), ("Save", True, True)],
)
def test_closing_with_unsaved_changes_asks_first(
    qapp, site, toasts, monkeypatch, answer, closed, saved
):
    from PyQt5.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: asked.append(a) or getattr(QMessageBox, answer),
    )
    before = utils.load_yaml(site.configuration_path)["defaults"]["electron"]["hfw"]
    window = MicroscopeConfigurationWindow(site)
    window.show()
    _edit(window)

    window.pushButton_close.click()

    assert len(asked) == 1
    assert window.isVisible() is not closed
    after = utils.load_yaml(site.configuration_path)["defaults"]["electron"]["hfw"]
    assert (after == pytest.approx(80.0e-6)) is saved
    assert saved or after == before
    window.close_without_asking()


def test_closing_with_nothing_to_save_does_not_ask(qapp, site, toasts, monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: pytest.fail("asked"))
    window = MicroscopeConfigurationWindow(site)
    window.show()

    window.pushButton_close.click()

    assert not window.isVisible()

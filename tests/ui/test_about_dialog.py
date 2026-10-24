"""Tests for the About dialog."""

from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from fibsem.ui.widgets.about_dialog import AboutDialog, mask_serial_number


def _fake_microscope(serial="8372-1194"):
    info = SimpleNamespace(
        name="METEOR",
        manufacturer="Thermo Fisher",
        model="Helios 5 UX",
        serial_number=serial,
        hardware_version="1.7.2",
        software_version="4.13.0",
    )
    return SimpleNamespace(system=SimpleNamespace(info=info))


@pytest.mark.parametrize(
    "serial, expected",
    [
        ("8372-1194", "8372-****"),
        ("ABCD1234567", "ABCD*******"),
        ("SN/2024/0091", "SN/2***/****"),  # separators survive
        ("12", "**"),  # shorter than the visible prefix
        ("", "Unknown"),
        (None, "Unknown"),
        ("   ", "Unknown"),
    ],
)
def test_mask_serial_number(serial, expected):
    assert mask_serial_number(serial) == expected


def _rows(dialog):
    """Every displayed key/value, flattened across sections."""
    return {key: value for _, section in dialog._sections for key, value in section}


def test_disconnected_dialog_has_no_microscope_section(qapp):
    dialog = AboutDialog(application="fibsemOS")

    titles = [title for title, _ in dialog._sections]
    assert "Microscope" not in titles
    assert titles[0] == "Software"
    assert _rows(dialog)["Application"] == "fibsemOS"


def test_connected_dialog_shows_masked_serial(qapp):
    dialog = AboutDialog(microscope=_fake_microscope(), application="AutoLamella")

    rows = _rows(dialog)
    assert rows["Name"] == "METEOR"
    assert rows["Serial number"] == "8372-****"
    assert rows["Application"] == "AutoLamella"


def test_copied_text_never_contains_the_full_serial(qapp):
    dialog = AboutDialog(microscope=_fake_microscope(), application="AutoLamella")

    text = dialog.as_text()

    assert "8372-1194" not in text
    assert "8372-****" in text
    # what you see is what you copy
    assert "METEOR" in text and "AutoLamella" in text


def test_microscope_section_skipped_when_info_unavailable(qapp):
    """A microscope in a bad state must not stop the dialog from opening."""

    class _Boom:
        @property
        def system(self):
            raise RuntimeError("disconnected mid-call")

    dialog = AboutDialog(microscope=_Boom(), application="AutoLamella")

    assert "Microscope" not in [title for title, _ in dialog._sections]


def _slot_text(dialog):
    from PyQt5.QtWidgets import QLabel

    return " ".join(
        w.text() for w in dialog._update_slot.findChildren(QLabel) if w.text()
    )


def _slot_buttons(dialog):
    from PyQt5.QtWidgets import QToolButton

    return dialog._update_slot.findChildren(QToolButton)


@pytest.fixture
def wheel_install(monkeypatch):
    monkeypatch.setattr("fibsem.ui.widgets.about_dialog.get_revision", lambda: None)
    monkeypatch.setattr("fibsem.update_check.get_revision", lambda: None)


def test_source_install_gets_no_update_slot_content(qapp, monkeypatch):
    """A PyPI comparison is meaningless from a checkout, so offer nothing."""
    monkeypatch.setattr("fibsem.update_check.get_revision", lambda: "v0.5.1-48-gabc")

    dialog = AboutDialog(application="fibsemOS")

    assert dialog._update_slot.layout().count() == 0


def test_check_button_offered_when_no_result_yet(qapp, wheel_install, monkeypatch):
    monkeypatch.setattr("fibsem.update_check.get_available_update", lambda: None)

    dialog = AboutDialog(application="fibsemOS")

    assert _slot_text(dialog) == ""
    assert len(_slot_buttons(dialog)) == 1


def test_background_result_shown_instead_of_the_button(
    qapp, wheel_install, monkeypatch
):
    monkeypatch.setattr("fibsem.update_check.get_available_update", lambda: "0.5.3")

    dialog = AboutDialog(application="fibsemOS")

    assert "0.5.3 available" in _slot_text(dialog)
    assert _slot_buttons(dialog) == []


@pytest.mark.parametrize(
    "render, expected_text, keeps_button",
    [
        ("_render_up_to_date", "up to date", False),
        ("_render_check_failed", "could not check", True),
    ],
)
def test_manual_check_reports_every_outcome(
    qapp, wheel_install, render, expected_text, keeps_button
):
    """Unlike the passive indicator, a check the user asked for always answers."""
    dialog = AboutDialog(application="fibsemOS")

    getattr(dialog, render)()

    assert expected_text in _slot_text(dialog)
    # A failed check keeps the button so a transient blip can be retried.
    assert bool(_slot_buttons(dialog)) is keeps_button


def test_result_after_close_is_ignored(qapp, wheel_install):
    """PyQt5 qFatals on an exception escaping a slot, so this must not throw."""
    from fibsem.update_check import UpdateStatus

    dialog = AboutDialog(application="fibsemOS")
    dialog.done(0)

    dialog._on_update_check_returned(
        UpdateStatus(current="0.5.1", latest="0.5.3", update_available=True)
    )

    assert dialog._closed is True


def test_update_slot_never_stops_the_dialog_opening(qapp, monkeypatch):
    def _boom():
        raise RuntimeError("kaboom")

    monkeypatch.setattr("fibsem.update_check.is_supported", _boom)

    AboutDialog(application="fibsemOS")  # must still construct


def test_environment_section_is_present(qapp):
    dialog = AboutDialog(application="fibsemOS")

    titles = [title for title, _ in dialog._sections]
    assert "Environment" in titles
    env = dict(dict(dialog._sections)["Environment"])
    assert "Python" in env

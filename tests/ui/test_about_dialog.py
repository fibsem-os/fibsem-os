"""Tests for the About dialog."""

from types import SimpleNamespace

import pytest

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


def test_environment_section_is_present(qapp):
    dialog = AboutDialog(application="fibsemOS")

    titles = [title for title, _ in dialog._sections]
    assert "Environment" in titles
    env = dict(dict(dialog._sections)["Environment"])
    assert "Python" in env

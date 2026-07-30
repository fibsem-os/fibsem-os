"""No microscope is a normal state for the text overlay, not an error.

`FibsemUI` builds `FibsemMinimapWidget` during setup, before any connection. That
widget's `microscope` property reads through to its parent, so it is None at that
point -- which the widget already handles everywhere else with
`if self.microscope is None: return`. `update_text_overlay` was the one path that
passed it through to be dereferenced, so every launch logged:

    WARNING:root:Error updating text overlay: 'NoneType' object has no attribute
    '_stage_position'
"""

import logging
from types import SimpleNamespace

import pytest

napari = pytest.importorskip("napari", reason="text overlay is a napari widget")

from fibsem.ui.napari.utilities import update_text_overlay  # noqa: E402

UNAVAILABLE = "<STAGE POSITION UNAVAILABLE>"


@pytest.fixture()
def viewer():
    """Just enough of a napari viewer for the overlay path."""
    return SimpleNamespace(
        text_overlay=SimpleNamespace(text="", visible=False, position=None)
    )


def test_no_microscope_does_not_warn(viewer, caplog):
    with caplog.at_level(logging.WARNING):
        update_text_overlay(viewer, None)

    assert caplog.records == [], (
        "a disconnected microscope is an expected startup state; warning about it "
        f"on every launch is noise. Got: {[r.getMessage() for r in caplog.records]}"
    )


def test_no_microscope_shows_the_unavailable_text(viewer):
    """Same text the failure path already produced, so the display is unchanged."""
    update_text_overlay(viewer, None)

    assert viewer.text_overlay.text == UNAVAILABLE


def test_a_real_failure_still_warns(viewer, caplog):
    """The guard must not become a blanket silencer.

    A microscope that raises when queried is a genuine problem and should still be
    reported, so narrowing the None case must not swallow the rest.
    """

    class Broken:
        @property
        def _stage_position(self):
            raise RuntimeError("comms lost")

    with caplog.at_level(logging.WARNING):
        update_text_overlay(viewer, Broken())

    assert any("Error updating text overlay" in r.getMessage() for r in caplog.records)
    assert viewer.text_overlay.text == UNAVAILABLE

"""No microscope is a normal state for the text overlay, not an error.

**`update_text_overlay` has no live caller left.** It wrote onto a napari viewer, and
the last one outside labelling/segmentation went with `FibsemMinimapWidget`. Its one
remaining import, `objective_control_widget`, calls it behind
`if getattr(self.parent_widget, "viewer", None) is not None` -- and no host sets a
viewer any more, so that branch cannot be reached. The function, this test and the rest
of `fibsem/ui/napari/utilities.py` go together in FIB-407 §1; kept green until then
rather than deleted piecemeal.

What it pins, from when it was live: the minimap was built during window setup, before
any connection, and its `microscope` property read through to its parent -- so it was
None at that point. The widget handled that everywhere else with
`if self.microscope is None: return`; this was the one path that passed it through to be
dereferenced, so every launch logged:

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

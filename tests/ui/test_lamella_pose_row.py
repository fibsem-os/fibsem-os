"""The pose row, and the details affordance on its position.

A pose is a whole `MicroscopeState`; the row has space for one line of it. These cover
the three places the rest of it now goes -- the tooltip, the popup, and the formatters
the line itself is built from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QPushButton

from fibsem.applications.autolamella.structures import Lamella
from fibsem.applications.autolamella.ui.lamella_pose_list_widget import (
    LamellaPoseListWidget,
    LamellaPoseRowWidget,
)
from fibsem.applications.autolamella.workflows.tasks.reference_image import (
    AcquireReferenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.trench import MillTrenchTaskConfig
from fibsem.applications.autolamella.workflows.tasks.undercut import (
    MillUndercutTaskConfig,
)
from fibsem.structures import (
    BeamSettings,
    BeamType,
    FibsemStagePosition,
    MicroscopeState,
)


def _state(x=0.0, y=0.0, ion_current=2e-11) -> MicroscopeState:
    return MicroscopeState(
        timestamp=1788000000.0,
        stage_position=FibsemStagePosition(
            x=x, y=y, z=0.0, r=0.0, t=np.radians(12), coordinate_system="RAW"
        ),
        electron_beam=BeamSettings(
            beam_type=BeamType.ELECTRON, voltage=2000.0, beam_current=1e-10, hfw=150e-6
        ),
        ion_beam=BeamSettings(
            beam_type=BeamType.ION,
            voltage=30000.0,
            beam_current=ion_current,
            hfw=150e-6,
        ),
    )


def _lamella() -> Lamella:
    lamella = Lamella(path=Path("/tmp/pose-row-test"), number=1, petname="01-test")
    lamella.milling_pose = _state()
    lamella.fluorescence_pose = _state(x=42e-6, y=-13e-6)
    return lamella


# ---------------------------------------------------------------------------
# The position is a control
# ---------------------------------------------------------------------------


def test_the_position_is_a_button_not_a_label(qapp):
    """A `QLabel` with a `mousePressEvent` would not be focusable or in the tab order,
    which would make Details the one action in the row unreachable from the keyboard
    while Move To and Update stayed reachable."""
    row = LamellaPoseRowWidget("MILLING", _state())

    assert isinstance(row.position_button, QPushButton)
    assert row.position_button.focusPolicy() != Qt.NoFocus
    assert row.position_button.cursor().shape() == Qt.PointingHandCursor


def test_the_row_looks_unchanged_until_hovered(qapp):
    """The affordance is the hover state; at rest the row must read as it always did.

    Asserted on the stylesheet because that is where the decision lives -- resting
    colour muted and no underline, both only appearing under `:hover`.
    """
    row = LamellaPoseRowWidget("MILLING", _state())
    style = row.position_button.styleSheet()

    resting, hover = style.split("QPushButton:hover")
    assert "text-decoration: underline" not in resting
    assert "text-decoration: underline" in hover
    assert "border: none" in resting


def test_the_position_uses_the_scaled_formatters(qapp):
    """Not `stage_position.pretty`, which is fixed millimetres and would round this
    13 micrometre offset to "-0.01mm"."""
    row = LamellaPoseRowWidget("FLUORESCENCE", _state(x=42e-6, y=-13e-6))

    text = row.position_button.text()
    assert "42.00 µm" in text
    assert "-13.00 µm" in text


# ---------------------------------------------------------------------------
# The tooltip -- a glance, no click
# ---------------------------------------------------------------------------


def test_the_tooltip_carries_the_beams(qapp):
    """The point of the affordance: these are recorded at every pose and the row has
    no space for them."""
    row = LamellaPoseRowWidget("MILLING", _state())
    tooltip = row.position_button.toolTip()

    assert "MILLING" in tooltip
    # SEM and FIB, matching the canvases and the popup this previews.
    assert "SEM" in tooltip and "2.00 kV" in tooltip and "100 pA" in tooltip
    assert "FIB" in tooltip and "30.00 kV" in tooltip and "20 pA" in tooltip
    assert "Electron" not in tooltip and "Ion" not in tooltip


def test_the_tooltip_survives_a_beam_that_was_never_configured(qapp):
    state = _state()
    state.ion_beam = None
    row = LamellaPoseRowWidget("MILLING", state)

    assert "FIB" in row.position_button.toolTip()


# ---------------------------------------------------------------------------
# The popup
# ---------------------------------------------------------------------------


def test_clicking_the_position_opens_the_full_state(qapp):
    row = LamellaPoseRowWidget("MILLING", _state())
    row.position_button.click()

    assert row._popup is not None
    assert row._popup.state_widget.label_title.text() == "MILLING"


def test_the_popup_dismisses_itself(qapp):
    """`Qt.Popup`, not a dialog. It closes on a click outside and does not block, so a
    pose can be read without losing the canvas and two poses are two clicks."""
    row = LamellaPoseRowWidget("MILLING", _state())
    row.position_button.click()

    assert bool(row._popup.windowFlags() & Qt.Popup)
    assert row._popup.isModal() is False


def test_the_popup_frame_style_is_scoped_to_itself(qapp):
    """An unscoped `QFrame { border }` cascades onto every descendant, drawing a box
    around every value in the table inside. Found by rendering it."""
    row = LamellaPoseRowWidget("MILLING", _state())
    row.position_button.click()

    assert "QFrame#PoseDetailPopup" in row._popup.styleSheet()


def test_an_open_popup_follows_the_row(qapp):
    """A pose updated while its detail is open should not go on showing the old one."""
    row = LamellaPoseRowWidget("MILLING", _state())
    row.position_button.click()
    row.set_state(_state(x=42e-6))

    values = row._popup.state_widget.grid_stage._layout
    texts = [
        values.itemAt(i).widget().text()
        for i in range(values.count())
        if values.itemAt(i).widget() is not None
    ]
    assert "42.00 µm" in texts


# ---------------------------------------------------------------------------
# A pose with nothing behind it
# ---------------------------------------------------------------------------


def test_a_row_with_no_record_offers_nothing_to_open(qapp):
    row = LamellaPoseRowWidget("LANDING", None)

    assert row.position_button.text() == "Unknown"
    assert row.position_button.isEnabled() is False


def test_the_name_stays_a_plain_label(qapp):
    """Only the position became a control; the name has nothing to expand."""
    row = LamellaPoseRowWidget("MILLING", _state())
    assert isinstance(row.name_label, QLabel)


# ---------------------------------------------------------------------------
# The list carries records, not rendered strings
# ---------------------------------------------------------------------------


def test_the_list_builds_rows_from_the_poses(qapp):
    widget = LamellaPoseListWidget()
    widget.set_lamella(_lamella())

    rows = [
        widget._list.itemWidget(widget._list.item(i))
        for i in range(widget._list.count())
    ]
    assert [row.pose_name for row in rows] == ["MILLING", "FLUORESCENCE"]
    assert "42.00 µm" in rows[1].position_button.text()


def test_refresh_pose_takes_the_record(qapp):
    """It used to take a pre-rendered string, and every caller reached into the state
    for `stage_position.pretty` to produce one. The row needs more of the record than
    that now, so the record is what it is given."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_lamella())

    widget.refresh_pose("MILLING", _state(x=99e-6))

    row = widget._list.itemWidget(widget._list.item(0))
    assert "99.00 µm" in row.position_button.text()
    assert "MILLING" in row.position_button.toolTip()


def test_refreshing_an_unknown_pose_is_a_no_op(qapp):
    widget = LamellaPoseListWidget()
    widget.set_lamella(_lamella())
    widget.refresh_pose("NOT_A_POSE", _state(x=99e-6))

    row = widget._list.itemWidget(widget._list.item(0))
    assert "99.00 µm" not in row.position_button.text()


# ---------------------------------------------------------------------------
# Overlay Pattern -- which steps' milling patterns to draw over the FIB image
# ---------------------------------------------------------------------------


def _milling_lamella() -> Lamella:
    """A lamella carrying task configs, as a protocol seeds it with.

    The real config classes rather than stand-ins: their ``__post_init__`` is what
    fills in the default milling stages, and having milling at all is the thing the
    entries are derived from. ``AcquireReferenceImageConfig`` is the control -- a
    task that mills nothing and must not be offered.
    """
    lamella = _lamella()
    lamella.task_config["Trench Milling"] = MillTrenchTaskConfig(
        task_name="Trench Milling"
    )
    lamella.task_config["Undercut Milling"] = MillUndercutTaskConfig(
        task_name="Undercut Milling"
    )
    lamella.task_config["Reference Image"] = AcquireReferenceImageConfig(
        task_name="Reference Image"
    )
    return lamella


def _rows(widget: LamellaPoseListWidget) -> dict:
    return {
        widget._list.itemWidget(
            widget._list.item(i)
        ).pose_name: widget._list.itemWidget(widget._list.item(i))
        for i in range(widget._list.count())
    }


def test_only_the_milling_row_carries_the_overlay_button(qapp):
    """The patterns belong to the lamella, not to a pose, but the milling pose is
    where they would be cut. On every row instead, a lamella with a fluorescence pose
    grows a second button that does exactly the same thing."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())

    rows = _rows(widget)
    assert rows["MILLING"].btn_overlay is not None
    assert rows["FLUORESCENCE"].btn_overlay is None


def test_the_rows_stay_aligned_without_the_button(qapp):
    """A hidden widget takes no space in a box layout, so the row without the button
    needs a spacer or its Move To and Update sit a button-width right of the others."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())

    rows = _rows(widget)
    assert rows["MILLING"].layout().count() == rows["FLUORESCENCE"].layout().count()


def test_the_popup_offers_the_tasks_that_mill_something(qapp):
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())

    controls = _rows(widget)["MILLING"]._overlay_controls
    assert list(controls.keys()) == ["Trench Milling", "Undercut Milling"]


def test_a_lamella_with_nothing_to_mill_has_no_button(qapp):
    widget = LamellaPoseListWidget()
    widget.set_lamella(_lamella())

    assert _rows(widget)["MILLING"].btn_overlay is None


def test_ticking_a_task_reports_the_whole_selection(qapp):
    """The canvas draws a set, so the set is what travels -- not the one box that
    changed, which would make every consumer reconstruct it."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())
    seen = []
    widget.pattern_overlays_changed.connect(seen.append)

    controls = _rows(widget)["MILLING"]._overlay_controls
    controls.set_visible("Trench Milling", True)
    controls.set_visible("Undercut Milling", True)

    assert seen == [["Trench Milling"], ["Trench Milling", "Undercut Milling"]]


def test_selecting_another_lamella_reports_that_lamella(qapp):
    """Rebuilding the rows leaves whatever was drawn describing the previous lamella,
    so the rebuild has to say what this one has ticked -- including when that is
    nothing, which is what clears the overlay."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())
    _rows(widget)["MILLING"]._overlay_controls.set_visible("Trench Milling", True)

    seen = []
    widget.pattern_overlays_changed.connect(seen.append)
    widget.set_lamella(_lamella())  # no task configs at all

    assert seen == [[]]


def test_the_selection_survives_a_lamella_that_does_not_have_it(qapp):
    """Remembered whole, emitted narrowed. Ticking Trench Milling, looking at a
    lamella without it and coming back should not have silently unticked it."""
    widget = LamellaPoseListWidget()
    widget.set_lamella(_milling_lamella())
    _rows(widget)["MILLING"]._overlay_controls.set_visible("Trench Milling", True)

    widget.set_lamella(_lamella())
    widget.set_lamella(_milling_lamella())

    controls = _rows(widget)["MILLING"]._overlay_controls
    assert controls.is_visible("Trench Milling") is True
    assert controls.is_visible("Undercut Milling") is False

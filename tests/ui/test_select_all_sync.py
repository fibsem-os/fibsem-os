"""A select-all has to leave the header agreeing with the rows it just changed.

The same shape as FIB-577, in a list widget the AutoLamella fix did not cover:
`_on_select_all` ticked every row and emitted, but never called `_sync_select_all()`.

`MillingStageListWidget` is the one that shows it. `_sync_select_all` turns the header
tristate the first time a row is unticked and never turns it back off, and a click on a
tristate box cycles through PartiallyChecked, which the header reports as
`bool(Qt.PartiallyChecked)` -- True. So the click that ticks every row can leave the
header showing a dash, with no external caller involved at all.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt

from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.patterning import get_pattern
from fibsem.structures.settings import FibsemMillingSettings
from fibsem.ui.widgets.milling_stage_list_widget import MillingStageListWidget


def _stage(name: str) -> FibsemMillingStage:
    return FibsemMillingStage(
        name=name,
        milling=FibsemMillingSettings(milling_current=2e-9),
        pattern=get_pattern(
            "Rectangle", {"depth": 1e-6, "width": 10e-6, "height": 5e-6}
        ),
    )


def _milling_list(enabled: bool = True) -> MillingStageListWidget:
    widget = MillingStageListWidget()
    widget.set_stages([_stage("A"), _stage("B"), _stage("C")])
    if not enabled:
        widget.set_all_selected(False)
    return widget


def _header_matches_rows(widget) -> bool:
    """The invariant the widget is supposed to hold after any select-all."""
    rows = [widget._row(i).checkbox.isChecked() for i in range(widget._list.count())]
    state = widget._header.checkbox_all.checkState()
    if all(rows):
        return state == Qt.Checked
    if not any(rows):
        return state == Qt.Unchecked
    return state == Qt.PartiallyChecked


# ── MillingStageListWidget ───────────────────────────────────────────────────


def test_tristate_header_clicks_never_disagree_with_the_rows(qapp):
    """The live bug: the third click ticked every row and left the header on a dash.

    Nothing here is programmatic -- one row toggle to make the header tristate, then
    plain clicks on it, which is all a user has to do.
    """
    widget = _milling_list()
    header = widget._header.checkbox_all

    widget._row(0).checkbox.setChecked(False)
    assert header.checkState() == Qt.PartiallyChecked
    assert header.isTristate(), "the premise: the header stays tristate from here on"

    for click in range(4):
        header.click()
        assert _header_matches_rows(widget), (
            f"header disagrees with the rows on click {click + 1}"
        )


def test_clearing_the_stage_selection_unticks_the_header(qapp):
    widget = _milling_list()
    assert widget._header.checkbox_all.checkState() == Qt.Checked

    widget.set_all_selected(False)

    assert widget.get_enabled_stages() == []
    assert widget._header.checkbox_all.checkState() == Qt.Unchecked


def test_selecting_all_stages_ticks_every_row(qapp):
    widget = _milling_list(enabled=False)

    widget.set_all_selected(True)

    assert len(widget.get_enabled_stages()) == 3
    assert widget._header.checkbox_all.checkState() == Qt.Checked


@pytest.mark.parametrize("checked", [True, False])
def test_select_all_writes_through_to_the_stages(qapp, checked):
    """`enabled` lives on the stage, not just the checkbox -- milling reads the model."""
    widget = _milling_list(enabled=not checked)
    seen = []
    widget.enabled_changed.connect(seen.append)

    widget.set_all_selected(checked)

    assert [s.enabled for s in widget.get_stages()] == [checked] * 3
    assert seen == [widget.get_enabled_stages()]


def test_reorder_after_clearing_the_stages_keeps_them_cleared(qapp):
    widget = _milling_list()

    widget.set_all_selected(False)
    widget._on_reordered(list(reversed(widget.get_stages())))

    assert widget.get_enabled_stages() == []
    assert widget._header.checkbox_all.checkState() == Qt.Unchecked


@pytest.mark.parametrize("checked", [True, False])
def test_the_stage_header_checkbox_still_drives_the_rows(qapp, checked):
    """Regression guard: the header signal now lands on the public method."""
    widget = _milling_list(enabled=not checked)
    seen = []
    widget.enabled_changed.connect(seen.append)

    widget._header.checkbox_all.setChecked(checked)

    assert len(widget.get_enabled_stages()) == (3 if checked else 0)
    assert seen, "the header path must still emit enabled_changed"

"""The replay window, on an experiment the real code recorded on Demo."""

import time

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.tools.replay import EventKind  # noqa: E402
from fibsem.applications.autolamella.ui.experiment_replay_widget import (  # noqa: E402
    ExperimentReplayWidget,
)
from tests.autolamella._replay_experiment import (  # noqa: E402
    FM_CHANNELS,
    FM_STACK,
    MILLING_STAGES,
    SPOTS,
    record_demo_experiment,
)


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    monkeypatch = pytest.MonkeyPatch()
    try:
        return record_demo_experiment(
            tmp_path_factory.mktemp("replay") / "exp", monkeypatch
        )
    finally:
        monkeypatch.undo()


@pytest.fixture
def widget(qapp, experiment):
    w = ExperimentReplayWidget.from_directory(experiment)
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.pause()
    w.close()
    w.deleteLater()
    qapp.processEvents()


def _first(widget, kind, where=lambda e: True):
    return next(
        i for i, e in enumerate(widget.replay.events) if e.kind == kind and where(e)
    )


def test_it_opens_on_the_first_action(widget):
    assert widget.index == 0
    assert widget.table.rowCount() == len(widget.replay.events)
    assert widget.table.currentRow() == 0


def test_a_saved_image_is_shown_on_its_beams_canvas(widget):
    index = _first(
        widget, EventKind.IMAGE, lambda e: e.image_on_disk and e.beam == "ION"
    )
    widget.seek(index)
    event = widget.replay.events[index]
    assert event.image_path.name in widget.fib_canvas._title_text
    assert widget.fib_canvas._hint_text is None


def test_milling_is_drawn_over_the_fib_image(widget):
    index = _first(widget, EventKind.MILLING, lambda e: "stage" in e.data)
    widget.seek(index)
    assert [s.name for s in widget.milling_overlay._stages] == list(MILLING_STAGES)
    assert widget.milling_overlay._selected_index == 0  # the stage being milled
    assert widget.milling_overlay._artists, "the patterns were drawn"


def test_spot_burns_are_drawn_as_they_happen(widget):
    last = max(
        i
        for i, e in enumerate(widget.replay.events)
        if e.kind == EventKind.MILLING and "spot" in e.data
    )
    widget.seek(last)
    assert len(widget.spot_overlay._points) == len(SPOTS)
    assert widget.milling_overlay._stages == []  # the mill before it has ended


def test_the_fm_z_stack_is_shown_in_the_fm_pane(widget):
    assert widget.fm_widget.layers == []  # nothing acquired yet at the start
    widget.seek(_first(widget, EventKind.FLUORESCENCE))
    assert len(widget.fm_widget.layers) == FM_CHANNELS
    assert FM_STACK in widget.fm_widget.canvas._title_text


def test_the_stage_is_marked_at_each_action(widget):
    widget.seek(_first(widget, EventKind.STAGE))
    assert [p.name for p in widget.stage_view._positions] == ["Stage"]
    assert widget.stage_view.canvas._info_text.startswith("x ")


def test_hiding_a_kind_hides_its_rows_and_steps_past_it(widget):
    widget.filter_boxes[EventKind.IMAGE].setChecked(False)
    images = [
        i for i, e in enumerate(widget.replay.events) if e.kind == EventKind.IMAGE
    ]
    assert all(widget.table.isRowHidden(i) for i in images)
    widget.seek(0)
    for _ in range(len(widget.replay.events)):
        widget.step(1)
        assert widget.replay.events[widget.index].kind != EventKind.IMAGE


def test_choosing_an_item_shows_and_plays_only_its_actions(widget):
    assert widget.item_combo.itemText(0) == "All items"
    index = widget.item_combo.findData("test")
    assert widget.item_combo.itemText(index) == "Lamella · test"
    widget.item_combo.setCurrentIndex(index)
    ours = [i for i, e in enumerate(widget.replay.events) if e.item == "test"]
    shown = [
        i for i in range(widget.table.rowCount()) if not widget.table.isRowHidden(i)
    ]
    assert shown == ours
    assert widget.replay.events[widget.index].item == "test"
    for _ in range(len(ours) + 1):
        widget.step(1)
        assert widget.replay.events[widget.index].item == "test"
    # The panes follow the item: no image from outside it.
    for pane in (widget.sem_canvas, widget.fib_canvas):
        assert pane._title_text is None or "/" not in pane._title_text
    # The counts follow the lamella: its task ran, but the mill came after it.
    assert widget.filter_boxes[EventKind.TASK].text() != "Task (0)"
    assert widget.filter_boxes[EventKind.MILLING].text() == "Milling (0)"


def test_the_actions_outside_any_item_can_be_chosen(widget):
    widget.item_combo.setCurrentIndex(widget.item_combo.findText("Outside a workflow"))
    assert widget.replay.events[widget.index].item is None
    widget.item_combo.setCurrentIndex(0)  # back to all
    assert not any(widget.table.isRowHidden(i) for i in range(widget.table.rowCount()))


def test_play_advances_and_pause_stops(widget, qapp):
    widget.speed_combo.setCurrentText("600×")
    widget.play()
    deadline = time.time() + 5
    while widget.index < 3 and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert widget.index >= 3
    widget.pause()
    stopped_at = widget.index
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.01)
    assert widget.index == stopped_at
    assert not widget.is_playing


def test_clicking_a_row_goes_to_that_action(widget):
    target = len(widget.replay.events) - 1
    widget.table.cellClicked.emit(target, 0)
    assert widget.index == target
    assert widget.event_label.text() == widget.replay.events[target].summary


def test_no_stored_overview_is_said_rather_than_left_blank(widget):
    # The Demo run saves no overview image.
    assert widget.stage_view.views == []
    assert widget.stage_view.canvas._hint_text


def test_playback_stops_at_the_end(widget, qapp):
    widget.seek(len(widget.replay.events) - 2)
    widget.speed_combo.setCurrentText("600×")
    widget.play()
    deadline = time.time() + 5
    while widget.is_playing and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    assert not widget.is_playing
    assert widget.index == len(widget.replay.events) - 1
    QApplication.processEvents()

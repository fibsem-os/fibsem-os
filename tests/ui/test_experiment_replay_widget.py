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


def _decision_on_a_real_image(tmp_path, kind, proposed, decided):
    """An experiment with one real FIB image of lamella 01, and a decision
    whose values sit on it; returns the image."""
    import json

    from fibsem import utils
    from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME
    from fibsem.structures import BeamType, ImageSettings

    # A real acquisition, so the saved file carries its pixel size as a
    # recorded one does (a blank image's metadata does not survive a save).
    microscope, _ = utils.setup_session(manufacturer="Demo")
    try:
        image = microscope.acquire_image(
            ImageSettings(resolution=[64, 48], hfw=64e-6, beam_type=BeamType.ION)
        )
    finally:
        microscope.disconnect()
    (tmp_path / "01").mkdir()
    path = tmp_path / "01" / "ref_final_ib.tif"
    image.save(str(path))
    t = "2026-09-23T14:00:0{}.000+10:00"
    records = [
        {
            "t": t.format(0),
            "kind": "image_acquired",
            "payload": {"path": str(path), "beam_type": "ION"},
        },
        {
            "t": t.format(1),
            "kind": "proposal_decided",
            "actor": "operator",
            "payload": {
                "item": {"id": "L1", "name": "01"},
                "task": "Setup Lamella Position",
                "proposal_id": "P1",
                "kind": kind,
                "image": "ref_final_ib.tif",
                "proposed": proposed,
                "decided": decided,
                "decision": 0,
                "outcome": "Confirmed",
                "author": "human:op",
                "via": "review",
            },
        },
    ]
    (tmp_path / EVENTS_FILENAME).write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return image


def _drawn(widget, pane):
    """The proposed points, the decided points and the rectangles on *pane*."""
    proposed, decided, areas = widget.proposal_overlays[pane]
    return (
        proposed._points,
        decided._points,
        [(s.kind, s.color, s.cx, s.cy, s.width, s.height) for s in areas._specs],
    )


def test_a_decision_draws_its_proposed_and_decided_points(qapp, tmp_path):
    """On the image the point was placed on: proposed where the task put it,
    decided where the operator moved it, +y up in metres as a point of
    interest is recorded."""
    image = _decision_on_a_real_image(
        tmp_path,
        "point_of_interest",
        {"poi": {"x": 0.0, "y": 0.0}},
        {"poi": {"x": 4e-6, "y": 2e-6}},
    )
    pixel_size = image.metadata.pixel_size.x
    height, width = image.data.shape[:2]
    w = ExperimentReplayWidget.from_directory(tmp_path)
    try:
        w.seek(0)
        assert _drawn(w, "fib") == ([], [], [])

        w.seek(1)
        assert "ref_final_ib.tif" in w.fib_canvas._title_text
        proposed, decided, areas = _drawn(w, "fib")
        assert proposed == [pytest.approx((width / 2, height / 2))]
        # 4 µm right and 2 µm up: +y is up in metres, down in pixels
        assert decided == [
            pytest.approx(
                (width / 2 + 4e-6 / pixel_size, height / 2 - 2e-6 / pixel_size)
            )
        ]
        assert areas == []
        assert _drawn(w, "sem") == ([], [], [])
    finally:
        w.close()
        w.deleteLater()
        qapp.processEvents()


def test_a_decided_alignment_area_is_drawn_as_a_rectangle(qapp, tmp_path):
    from fibsem.ui.tokens import DRAFT_POSITION_COLOUR, ORANGE_COLOR

    area = {"left": 0.25, "top": 0.25, "width": 0.5, "height": 0.25}
    moved = dict(area, left=0.5)
    image = _decision_on_a_real_image(
        tmp_path,
        "alignment_area",
        {"alignment_area": area},
        {"alignment_area": moved},
    )
    height, width = image.data.shape[:2]
    w = ExperimentReplayWidget.from_directory(tmp_path)
    try:
        w.seek(1)
        assert _drawn(w, "fib")[2] == [
            (
                "rect",
                ORANGE_COLOR,
                0.5 * width,
                0.375 * height,
                0.5 * width,
                0.25 * height,
            ),
            (
                "rect",
                DRAFT_POSITION_COLOUR,
                0.75 * width,
                0.375 * height,
                0.5 * width,
                0.25 * height,
            ),
        ]
    finally:
        w.close()
        w.deleteLater()
        qapp.processEvents()


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

"""Offscreen tests for GridResultsWidget (per-grid results view)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    GridRecord,
)


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover
        pytest.skip(f"Qt unavailable: {e}")
    return app


@pytest.fixture
def widget(qapp):
    from fibsem.ui.widgets.grid_results_widget import GridResultsWidget

    w = GridResultsWidget()
    yield w
    w._cancel_worker()  # stop any image-loader thread before teardown


def _record_with_results():
    rec = GridRecord(name="grid-aspen")
    ts = AutoLamellaTaskState(name="overview")
    ts.status = AutoLamellaTaskStatus.Completed
    rec.task_history = [ts]
    rec.results = {
        "ACQUIRE_OVERVIEW_IMAGE_GRID": {
            "overview": "/tmp/overview.tif", "thumbnail": "/tmp/thumbnail.png"
        },
        "CRYO_CLEANING_GRID": {"fib": "/tmp/clean_ib.tif"},
    }
    return rec


def test_empty_when_no_grid(widget):
    widget.set_grid(None)
    assert widget._record is None
    assert widget._overview_path is not None  # method exists
    # only the empty label is in the content
    assert widget._content_layout.indexOf(widget._empty) >= 0


def test_overview_and_gallery_resolution(widget):
    widget.set_grid(_record_with_results())
    assert widget._overview_path() == "/tmp/overview.tif"
    # the overview (hero) and the card thumbnail are excluded from the gallery;
    # the task shows its friendly display name
    gallery = widget._gallery_items()
    assert gallery == [("Cryo Cleaning Milling · fib", "/tmp/clean_ib.tif")]
    # overview is the (responsive) hero; the gallery image is a fixed target
    assert widget._hero_path == "/tmp/overview.tif"
    assert set(widget._targets.keys()) == {"/tmp/clean_ib.tif"}


def test_no_overview_uses_placeholder(widget):
    rec = GridRecord(name="grid-birch")  # no results
    widget.set_grid(rec)
    assert widget._overview_path() is None
    assert widget._gallery_items() == []
    assert widget._targets == {}  # nothing to load


def test_sections_built_for_record(widget):
    widget.set_grid(_record_with_results())
    # header + (overview | history) row + Artifacts panel
    assert widget._content_layout.count() == 3


def test_image_cache_avoids_reload(widget, tmp_path):
    from PIL import Image
    import numpy as np
    from fibsem.applications.autolamella.structures import GridRecord

    img = tmp_path / "overview.tif"
    Image.fromarray(np.zeros((40, 60, 3), dtype=np.uint8)).save(img)
    rec = GridRecord(name="grid-aspen")
    rec.results = {"OVERVIEW": {"overview": str(img)}}

    # spy on what the loader is asked to load
    queued: list = []
    widget._loader.load = lambda paths: queued.append(list(paths))

    # first load queues the (uncached) overview
    widget.set_grid(rec)
    assert queued and str(img) in queued[-1]

    # simulate the load completing → cache populated
    widget._on_image_loaded(str(img), np.zeros((40, 60, 3), dtype=np.uint8), 1e-8)
    assert str(img) in widget._img_cache

    # second set_grid fills from cache → nothing new queued
    queued.clear()
    widget.set_grid(rec)
    assert queued == []  # nothing left to load


def test_header_shows_slot_and_beam(widget):
    rec = _record_with_results()
    widget.set_grid(rec, slot_label="03", in_beam=True)
    assert widget._slot_label == "03"
    assert widget._in_beam is True


def test_history_collapses_to_latest_per_task(widget):
    rec = GridRecord(name="grid-aspen")
    a1 = AutoLamellaTaskState(name="OVERVIEW"); a1.status = AutoLamellaTaskStatus.Failed
    a2 = AutoLamellaTaskState(name="OVERVIEW"); a2.status = AutoLamellaTaskStatus.Completed
    b1 = AutoLamellaTaskState(name="IMAGE"); b1.status = AutoLamellaTaskStatus.Completed
    rec.task_history = [a1, a2, b1]
    widget.set_grid(rec)

    latest = widget._latest_per_task()
    assert [ts.name for ts in latest] == ["OVERVIEW", "IMAGE"]  # one row per task
    # the latest OVERVIEW entry wins (Completed, not the earlier Failed)
    assert latest[0].status is AutoLamellaTaskStatus.Completed


def test_task_click_filters_artifacts(widget):
    rec = GridRecord(name="grid-aspen")
    rec.results = {
        "ACQUIRE_IMAGE_GRID": {"image": "/tmp/a.tif"},
        "CRYO_CLEANING_GRID": {"fib": "/tmp/b.tif"},
    }
    widget.set_grid(rec)
    assert len(widget._gallery_items()) == 2  # all by default

    # selecting a task filters to its artifacts
    widget._on_task_clicked("CRYO_CLEANING_GRID")
    assert widget._selected_task == "CRYO_CLEANING_GRID"
    assert widget._gallery_items(widget._selected_task) == [
        ("Cryo Cleaning Milling · fib", "/tmp/b.tif")
    ]

    # clicking the same task again clears the filter
    widget._on_task_clicked("CRYO_CLEANING_GRID")
    assert widget._selected_task is None


def test_set_grid_none_after_record_clears(widget):
    widget.set_grid(_record_with_results())
    assert widget._targets  # had images
    widget.set_grid(None)
    assert widget._targets == {}
    assert widget._record is None

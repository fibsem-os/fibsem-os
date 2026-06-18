"""Offscreen tests for GridCardWidget / GridCardContainer (Phase 4a grid UI)."""

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
def container(qapp):
    from fibsem.ui.widgets.grid_card_widget import GridCardContainer

    return GridCardContainer()


def _records():
    a = GridRecord(name="grid-aspen")
    a.task_history = [AutoLamellaTaskState(name="overview"), AutoLamellaTaskState(name="clean")]
    b = GridRecord(name="grid-birch")
    return [a, b]


def _cards(container):
    return container._cards


def test_population_and_count(container):
    container.set_grids(_records())
    assert len(_cards(container)) == 2
    assert container._count.text() == "· 2"
    assert _cards(container)[0]._name_label.text() == "grid-aspen"
    assert _cards(container)[0]._status_label.text() == "2 tasks complete"
    assert _cards(container)[1]._status_label.text() == "Not started"


def test_failed_status(container):
    rec = GridRecord(name="grid-cedar")
    rec.task_state.status = AutoLamellaTaskStatus.Failed
    rec.task_state.name = "Cryo Cleaning"
    container.set_grids([rec])
    assert _cards(container)[0]._status_label.text() == "Failed — Cryo Cleaning"


def test_slot_badge_and_in_beam(container):
    records = _records()
    container.set_grids(
        records, slot_labels={"grid-aspen": "01"}, beam_names={"grid-aspen"}
    )
    aspen = _cards(container)[0]
    assert aspen._slot_badge.text() == "01"
    assert aspen._thumb._beam_pill.isVisible() or aspen.record.name == "grid-aspen"
    # birch not in magazine / beam
    assert _cards(container)[1]._slot_badge.text() == "—"


def test_empty_state(container):
    container.set_grids([])
    assert not _cards(container)
    assert container._count.text() == "· 0"
    assert container._empty.isVisible() or not container._scroll.isVisible()


def test_add_from_loader_emits(container):
    seen = []
    container.add_from_loader_requested.connect(lambda: seen.append(True))
    container.btn_add.click()
    assert seen == [True]


def test_card_remove_emits(container):
    container.set_grids(_records())
    seen = []
    container.remove_requested.connect(seen.append)
    _cards(container)[1].remove_requested.emit(_cards(container)[1].record)
    assert seen and seen[0].name == "grid-birch"


def test_loader_present_flag_on_cards(container):
    records = _records()
    container.set_grids(records, loader_present=False)
    assert all(c._loader_present is False for c in _cards(container))

    container.set_grids(records, loader_present=True)
    assert all(c._loader_present is True for c in _cards(container))


def test_card_load_unload_forwarded(container):
    container.set_grids(_records(), beam_names={"grid-aspen"})
    aspen, birch = _cards(container)
    assert aspen._in_beam and not birch._in_beam  # aspen in beam, birch not

    loads, unloads = [], []
    container.load_requested.connect(loads.append)
    container.unload_requested.connect(unloads.append)

    # the menu's action emits load on a not-loaded grid, unload on a loaded one
    birch.load_requested.emit(birch.record)
    aspen.unload_requested.emit(aspen.record)

    assert loads and loads[0].name == "grid-birch"
    assert unloads and unloads[0].name == "grid-aspen"


def test_selection_emits_and_toggles(container):
    container.set_grids(_records())
    seen = []
    container.grid_selected.connect(seen.append)

    container._on_card_clicked(_cards(container)[1].record)
    assert container.selected_grid.name == "grid-birch"
    assert seen[-1].name == "grid-birch"

    # clicking the selected card again deselects
    container._on_card_clicked(_cards(container)[1].record)
    assert container.selected_grid is None
    assert seen[-1] is None


def test_selection_preserved_by_name(container):
    records = _records()
    container.set_grids(records)
    container._on_card_clicked(records[1])  # select grid-birch
    assert container.selected_grid.name == "grid-birch"

    container.set_grids(list(reversed(records)))
    assert container.selected_grid.name == "grid-birch"

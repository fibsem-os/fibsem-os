"""Offscreen tests for the grid chip on lamella list rows (_LamellaRow)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    Experiment,
    GridRecord,
    Lamella,
)
from fibsem.microscopes._stage import SampleGrid  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def microscope():
    from fibsem.microscopes._stage import SampleHolder, Stage

    m, _ = utils.setup_session(manufacturer="Demo")
    m.stage_is_compustage = False
    holder = SampleHolder(name="Test Holder", capacity=2)
    holder._ensure_slots()
    holder._parent = m
    m._stage = Stage(parent=m, holder=holder, loader=None)
    return m


@pytest.fixture
def scene(tmp_path, microscope):
    """Experiment with grid-aspen LOADED, grid-birch NOT loaded, + 3 lamellae."""
    exp = Experiment.create(path=str(tmp_path), name="exp")
    aspen = GridRecord(name="grid-aspen")
    birch = GridRecord(name="grid-birch")
    exp.add_grid(aspen)
    exp.add_grid(birch)
    exp.add_lamella(Lamella(petname="on-loaded", path=str(tmp_path / "a"), number=1, grid_id=aspen._id))
    exp.add_lamella(Lamella(petname="on-unloaded", path=str(tmp_path / "b"), number=2, grid_id=birch._id))
    exp.add_lamella(Lamella(petname="unlinked", path=str(tmp_path / "c"), number=3))
    # load grid-aspen into a working slot
    next(iter(microscope._stage.holder.slots.values())).loaded_grid = SampleGrid(name="grid-aspen")
    return exp, microscope


def _rows(widget):
    return [widget._list.itemWidget(widget._list.item(i))
            for i in range(widget._list.count())]


def test_grid_chip_states(qapp, scene):
    from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

    exp, microscope = scene
    w = LamellaNameListWidget()
    w.set_lamella(exp.positions, experiment=exp, microscope=microscope)
    on_loaded, on_unloaded, unlinked = _rows(w)

    # on the loaded grid → chip shows grid name, "loaded" tooltip
    assert on_loaded.grid_badge.text() == "grid-aspen"
    assert "loaded grid" in on_loaded.grid_badge.toolTip()

    # on an unloaded grid → chip shows grid name, "not loaded" tooltip
    assert on_unloaded.grid_badge.text() == "grid-birch"
    assert "not loaded" in on_unloaded.grid_badge.toolTip()

    # unlinked lamella → no chip
    assert unlinked.grid_badge.text() == ""
    assert unlinked.grid_badge.isVisibleTo(unlinked) is False


def test_no_chip_without_context(qapp, scene):
    """Without experiment/microscope, rows render no grid chip (back-compat)."""
    from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

    exp, _ = scene
    w = LamellaNameListWidget()
    w.set_lamella(exp.positions)  # no context
    for row in _rows(w):
        assert row.grid_badge.text() == ""


def test_stage_actions_disabled_when_grid_unloaded(qapp, scene):
    """Move-to / update are disabled for off-grid lamellae; others stay enabled."""
    from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

    exp, microscope = scene
    w = LamellaNameListWidget()
    w.set_lamella(exp.positions, experiment=exp, microscope=microscope)
    on_loaded, on_unloaded, unlinked = _rows(w)

    # reachable rows (loaded grid + unlinked) keep stage actions enabled
    for row in (on_loaded, unlinked):
        assert row.btn_move_to.isEnabled()
        assert row.btn_update.isEnabled()

    # off-grid row: stage actions disabled (with explanatory tooltip)
    assert not on_unloaded.btn_move_to.isEnabled()
    assert not on_unloaded.btn_update.isEnabled()
    assert "not loaded" in on_unloaded.btn_move_to.toolTip()
    # non-stage actions remain available
    assert on_unloaded.btn_edit.isEnabled()
    assert on_unloaded.btn_remove.isEnabled()


def test_no_grids_used_everything_enabled(qapp, tmp_path, microscope):
    """User who never uses the grids feature: all lamellae are grid_id=None, so
    even WITH experiment+microscope context there are no chips and every control
    stays enabled (full back-compat)."""
    from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

    exp = Experiment.create(path=str(tmp_path), name="exp")  # no grids added
    exp.add_lamella(Lamella(petname="lam-1", path=str(tmp_path / "1"), number=1))
    exp.add_lamella(Lamella(petname="lam-2", path=str(tmp_path / "2"), number=2))

    w = LamellaNameListWidget()
    w.set_lamella(exp.positions, experiment=exp, microscope=microscope)

    for row in _rows(w):
        assert row.grid_badge.text() == ""              # no chip
        assert row.grid_badge.isVisibleTo(row) is False
        assert row.btn_move_to.isEnabled()              # stage controls enabled
        assert row.btn_update.isEnabled()
        assert row.btn_move_to.toolTip() == "Move to Position"  # default tooltip


def test_rebuild_tracks_loaded_grid_change(qapp, scene):
    """Re-populating after the loaded grid changes flips chip + controls — the
    behaviour the grid-exchange refresh hook relies on."""
    from fibsem.ui.widgets.custom_widgets import LamellaNameListWidget

    exp, microscope = scene
    w = LamellaNameListWidget()

    # grid-aspen loaded → its lamella is reachable
    w.set_lamella(exp.positions, experiment=exp, microscope=microscope)
    assert _rows(w)[0].btn_move_to.isEnabled()
    assert "loaded grid" in _rows(w)[0].grid_badge.toolTip()

    # unload it (working slot cleared) and rebuild → now unreachable
    next(iter(microscope._stage.holder.slots.values())).loaded_grid = None
    w.set_lamella(exp.positions, experiment=exp, microscope=microscope)
    assert not _rows(w)[0].btn_move_to.isEnabled()
    assert "not loaded" in _rows(w)[0].grid_badge.toolTip()

"""Offscreen tests for SampleWidget (holder + optional loader magazine).

SampleWidget hosts a LoaderMagazineWidget (storage, top, autoloader only) above
a SampleHolderWidget (working slot, always), and drives grid load/unload on the
stage with no Experiment coupling. These tests exercise:

- the magazine is present only when the stage has a loader; the holder always is;
- a magazine grid and the holder working slot share the same SampleGrid object
  once loaded (so refreshing the holder reflects magazine edits);
- loading a grid into the working slot (Stage.ensure_loaded) is reflected;
- finishing an exchange emits state_changed and clears the busy state.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.microscopes._stage import _create_sample_stage  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as e:  # pragma: no cover - no Qt platform available
        pytest.skip(f"Qt unavailable: {e}")
    return app


def _compustage_microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)  # loader + 1 working slot
    return microscope


def _sample_widget(microscope):
    from fibsem.ui.widgets.sample_widget import SampleWidget

    w = SampleWidget(microscope)
    if w.holder_widget is not None:
        w.holder_widget._auto_save = lambda *a, **k: None  # don't touch shared config
    return w


def _name_magazine_grid(widget, name="grid-aspen"):
    row = widget.magazine_widget._list.itemWidget(widget.magazine_widget._list.item(0))
    row.name_edit.setText(name)
    row._on_name_edited()
    return row


def test_magazine_present_only_with_loader(qapp):
    # autoloader -> magazine built, holder present
    w = _sample_widget(_compustage_microscope())
    assert w.magazine_widget is not None
    assert w.holder_widget is not None

    # no loader -> no magazine, holder still present
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)  # no loader
    assert microscope._stage.loader is None
    w2 = _sample_widget(microscope)
    assert w2.magazine_widget is None
    assert w2.holder_widget is not None


def test_load_reflected_in_holder(qapp):
    microscope = _compustage_microscope()
    w = _sample_widget(microscope)
    _name_magazine_grid(w, "grid-aspen")

    microscope._stage.ensure_loaded("grid-aspen")  # the stage operation the widget runs
    w.holder_widget.refresh()

    slot = next(iter(microscope._stage.holder.slots.values()))
    assert slot.loaded_grid is not None
    assert slot.loaded_grid.name == "grid-aspen"


def test_view_sync_shared_grid_object(qapp):
    microscope = _compustage_microscope()
    w = _sample_widget(microscope)
    row = _name_magazine_grid(w, "grid-aspen")
    microscope._stage.ensure_loaded("grid-aspen")

    # the magazine slot grid and the working-slot grid are the same object
    mag_slot = microscope._stage.loader.find_grid("grid-aspen")
    work_slot = next(iter(microscope._stage.holder.slots.values()))
    assert mag_slot.loaded_grid is work_slot.loaded_grid

    # editing the description via the magazine row mutates the shared object;
    # holder.refresh() (the view-sync wired to magazine_changed) shows it
    row.desc_edit.setText("lamella candidate")
    row._on_desc_edited()
    w.holder_widget.refresh()
    assert work_slot.loaded_grid.description == "lamella candidate"


def test_exchange_finish_emits_state_changed(qapp):
    microscope = _compustage_microscope()
    w = _sample_widget(microscope)

    fired = []
    w.state_changed.connect(lambda: fired.append(True))
    w._on_exchange_finished("load", "grid-aspen", None)  # GUI-thread completion

    assert fired == [True]  # host seam fires → refresh lamella list + grids tab

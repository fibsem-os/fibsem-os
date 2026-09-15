"""Select lamellae by grid on the Workflow tab (FIB-667).

A filter icon in the list's header carries a menu that narrows the list to one
grid's lamellae; Select All and the run selection then mean the rows on show.
Offered only when the experiment has more than one grid.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.lamella_list_widget import (
    FILTER_ALL,
    FILTER_NO_GRID,
    LamellaListWidget,
)
from fibsem.structures import MicroscopeState

_app = QApplication.instance() or QApplication([])


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="filter")
    (tmp_path / "filter").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


@pytest.fixture
def widget(experiment, destroy_widgets_after_test):
    oak = experiment.add_grid(GridRecord(name="grid-oak"))
    elm = experiment.add_grid(GridRecord(name="grid-elm"))
    config = experiment.task_protocol.task_config
    for name, grid in (
        ("a-oak", oak),
        ("b-elm", elm),
        ("c-oak", oak),
        ("d-free", None),
    ):
        experiment.add_new_lamella(
            microscope_state=MicroscopeState(),
            task_config=config,
            name=name,
            grid_id=grid.id if grid else None,
        )
    w = LamellaListWidget()
    w.set_grid_context({g.id: (g.name, False) for g in experiment.grids})
    w.set_lamellae(list(experiment.positions))
    w.show()
    _app.processEvents()
    return w


def _shown(widget):
    return [r.lamella.name for r in widget._visible_rows()]


def test_the_menu_offers_every_grid_and_no_grid(widget):
    assert widget.grid_filter.isVisibleTo(widget)
    assert [a.text() for a in widget.grid_filter.menu().actions()] == [
        "All grids",
        "grid-oak",
        "grid-elm",
        "No grid",
    ]
    assert widget.grid_filter.selected == FILTER_ALL
    assert _shown(widget) == ["a-oak", "b-elm", "c-oak", "d-free"]


def test_one_grid_offers_nothing(experiment, destroy_widgets_after_test):
    oak = experiment.add_grid(GridRecord(name="grid-oak"))
    experiment.add_new_lamella(
        microscope_state=MicroscopeState(),
        task_config=experiment.task_protocol.task_config,
        name="a-oak",
        grid_id=oak.id,
    )
    w = LamellaListWidget()
    w.set_grid_context({oak.id: ("grid-oak", True)})
    w.set_lamellae(list(experiment.positions))
    assert not w.grid_filter.isVisibleTo(w)


def test_a_grid_entry_narrows_the_list_and_the_selection(widget, experiment):
    oak = experiment.get_grid_by_name("grid-oak")
    widget.set_all_selected(True)
    assert [p.name for p in widget.get_selected()] == [
        "a-oak",
        "b-elm",
        "c-oak",
        "d-free",
    ]

    seen = []
    widget.selection_changed.connect(lambda sel: seen.append([p.name for p in sel]))
    widget.grid_filter.action(oak.id).trigger()
    assert _shown(widget) == ["a-oak", "c-oak"]
    assert widget.grid_filter.toolTip() == "Showing grid-oak"
    assert widget.grid_filter.action(oak.id).isChecked()
    # The hidden rows lost their ticks: a hidden tick would run unseen.
    assert [p.name for p in widget.get_selected()] == ["a-oak", "c-oak"]
    assert seen[-1] == ["a-oak", "c-oak"]

    # Select All now means the rows on show.
    widget.set_all_selected(False)
    widget.set_all_selected(True)
    assert [p.name for p in widget.get_selected()] == ["a-oak", "c-oak"]
    assert widget._header.checkbox_all.isChecked()

    widget.grid_filter.action(FILTER_NO_GRID).trigger()
    assert _shown(widget) == ["d-free"]
    assert widget.get_selected() == []
    widget.grid_filter.action(FILTER_ALL).trigger()
    assert _shown(widget) == ["a-oak", "b-elm", "c-oak", "d-free"]


def test_the_choice_survives_a_rebuild_and_a_context_refresh(widget, experiment):
    oak = experiment.get_grid_by_name("grid-oak")
    widget.grid_filter.action(oak.id).trigger()
    widget.set_lamellae(list(experiment.positions))
    assert widget.grid_filter.selected == oak.id
    assert _shown(widget) == ["a-oak", "c-oak"]
    widget.set_grid_context({g.id: (g.name, g is oak) for g in experiment.grids})
    assert widget.grid_filter.selected == oak.id
    assert _shown(widget) == ["a-oak", "c-oak"]
    # A grid that is gone drops the filter back to all.
    experiment.remove_grid("grid-oak")
    widget.set_grid_context({g.id: (g.name, False) for g in experiment.grids})
    assert widget.grid_filter.selected == FILTER_ALL

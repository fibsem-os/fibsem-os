"""Grids · Positions: lamellae marked on a grid's stored overviews (FIB-71).

The screening hand-off. The grid's overviews come off disk and are placed from
their own metadata; the lamellae marked here carry the grid's id whether or not
the grid is on the stage; nothing reads or moves the microscope.
"""

import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

import fibsem.config as fibsem_config
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.applications.autolamella.ui.grid_positions_widget import (
    GridPositionsWidget,
)
from fibsem.applications.autolamella.ui.grids_tab_widget import (
    VIEW_POSITIONS,
    VIEW_RESULTS,
    GridsTabWidget,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings
from fibsem.ui.widgets.stored_overview_canvas import VIEW_FM

_app = QApplication.instance() or QApplication([])

_ARCTIS = os.path.join(
    os.path.dirname(fibsem_config.__file__), "config", "sim-arctis-configuration.yaml"
)


@pytest.fixture
def ui(qapp, monkeypatch, tmp_path):
    widget = AutoLamellaUI(parent_ui=None)
    monkeypatch.setattr(
        widget.system_widget,
        "load_configuration",
        lambda configuration_name=None: _ARCTIS,
    )
    widget.system_widget.connect_to_microscope()
    experiment = Experiment(path=tmp_path, name="positions")
    os.makedirs(str(experiment.path), exist_ok=True)
    experiment.task_protocol = AutoLamellaTaskProtocol()
    widget.experiment = experiment
    yield widget
    widget.experiment = None
    widget.microscope.disconnect()
    widget.close()


def _entry(name, relpath):
    state = AutoLamellaTaskState(name=name, status=AutoLamellaTaskStatus.Completed)
    state.end_timestamp = state.start_timestamp + 60
    state.outputs = {name: [relpath]}
    return state


def _save_beam_overview(microscope, path, beam, position):
    image = FibsemImage.generate_blank_image(resolution=(96, 64), hfw=200e-6)
    image.data = (np.random.default_rng(1).random((64, 96)) * 255).astype(np.uint8)
    state = microscope.get_microscope_state(beam_type=beam)
    state.stage_position = position
    image.metadata.image_settings = ImageSettings(hfw=200e-6, beam_type=beam)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    image.save(str(path))


def _save_fm_overview(microscope, path, position):
    image = FluorescenceImage(
        data=(np.random.default_rng(2).random((1, 1, 128, 128)) * 4000).astype(
            np.uint16
        ),
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-14T10:00:00",
            pixel_size_x=1e-7,
            pixel_size_y=1e-7,
            stage_position=position,
            channels=[
                FluorescenceChannelMetadata(
                    name="GFP",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                    color="cyan",
                )
            ],
        ),
    )
    image.metadata.geometry = microscope.fm_image_geometry()
    image.save(str(path))


@pytest.fixture
def grid(ui):
    """A grid with an SEM, a FIB and an FM overview on disk, one lamella on it
    and one lamella that is not."""
    experiment, microscope = ui.experiment, ui.microscope
    grid = experiment.add_grid(GridRecord(name="grid-oak"))
    root = experiment.grid_path(grid)
    sem = microscope.get_stage_position()
    fib = microscope.get_target_position(deepcopy(sem), "FIB")
    fm = microscope.get_target_position(deepcopy(sem), "FM")
    for name, rel, beam, position in (
        ("overview_sem", "overview_sem/overview.tif", BeamType.ELECTRON, sem),
        ("overview_fib", "overview_fib/overview.tif", BeamType.ION, fib),
    ):
        (root / name).mkdir(parents=True)
        _save_beam_overview(microscope, root / rel, beam, position)
        grid.task_history.append(_entry(name, rel))
    (root / "overview_fm").mkdir()
    _save_fm_overview(microscope, root / "overview_fm/mosaic.ome.tiff", fm)
    grid.task_history.append(_entry("overview_fm", "overview_fm/mosaic.ome.tiff"))

    ui.add_new_lamella(stage_position=sem, name="on-grid", grid_id=grid.id)
    ui.add_new_lamella(stage_position=sem, name="elsewhere", grid_id=None)
    assert [p.grid_id for p in experiment.positions] == [grid.id, None]
    return grid


@pytest.fixture
def widget(ui, grid, destroy_widgets_after_test):
    w = GridPositionsWidget()
    w.set_autolamella_ui(ui)
    w.set_experiment(ui.experiment)
    w.set_grid(grid)
    w.resize(900, 600)
    w.show()
    _app.processEvents()
    yield w
    w.close()


def _centre(widget):
    record = next(r for r in widget.canvas.overviews if r.view == widget.canvas.view)
    (cx, cy), _ = record.extent
    return cx, cy


def _names(widget):
    return [p.name for p in widget.canvas._positions]


class TestShowingAGrid:
    def test_its_overviews_are_placed_and_only_its_lamellae_marked(self, widget, grid):
        views = widget.views
        assert len(views) == 3 and views[-1] == VIEW_FM
        assert views[0].startswith("SEM @") and views[1].startswith("FIB @")
        assert widget.canvas.view == views[0]
        assert [c.text() for c in widget._chips.values()] == [
            "overview_sem",
            "overview_fib",
            "overview_fm",
        ]
        assert _names(widget) == ["on-grid"]
        assert widget.count_label.text() == "Positions on grid-oak · 1"
        assert "not on the stage" in widget.state_label.text()

    def test_the_fm_view_marks_the_fluorescence_pose(self, widget, ui):
        lamella = ui.experiment.positions[0]
        widget.canvas.show_view(VIEW_FM)
        assert widget.canvas.view == VIEW_FM
        marked = widget.canvas._positions[0]
        fm = lamella.fluorescence_pose.stage_position
        assert (marked.r, marked.t) == (fm.r, fm.t)
        assert widget._chips[VIEW_FM].isChecked()

    def test_the_banner_says_when_the_grid_is_on_the_stage(self, widget):
        widget.set_stage_state(loaded=True, can_load=True)
        assert widget.state_label.text() == "Stored overview · on the stage"
        assert not widget.btn_load.isVisible()
        widget.set_stage_state(loaded=False, can_load=True)
        assert widget.btn_load.isVisible()
        assert widget.btn_load.text() == "Load grid-oak"


class TestMarking:
    def test_a_position_added_on_a_beam_overview_belongs_to_the_grid(
        self, widget, ui, grid
    ):
        cx, cy = _centre(widget)
        widget.canvas.request_add_at(cx + 5, cy + 5)
        lamellae = ui.experiment.positions
        assert len(lamellae) == 3
        new = lamellae[-1]
        assert new.grid_id == grid.id
        assert _names(widget) == ["on-grid", new.name]
        assert widget.lamella_list.selected_name == new.name
        # A beam-view mark is a milling pose, at the view's orientation.
        sem = ui.microscope.get_stage_position()
        assert (new.stage_position.r, new.stage_position.t) == (sem.r, sem.t)

    def test_a_position_added_on_the_fm_view_is_a_fluorescence_pose(
        self, widget, ui, grid
    ):
        widget.canvas.show_view(VIEW_FM)
        cx, cy = _centre(widget)
        widget.canvas.request_add_at(cx, cy)
        new = ui.experiment.positions[-1]
        assert new.grid_id == grid.id
        fm = ui.microscope.get_target_position(
            deepcopy(ui.microscope.get_stage_position()), "FM"
        )
        pose = new.fluorescence_pose.stage_position
        assert (pose.r, pose.t) == (fm.r, fm.t)
        assert new.stage_position.t != pose.t  # the milling pose was derived

    def test_move_and_remove(self, widget, ui):
        experiment = ui.experiment
        lamella = experiment.positions[0]
        before = lamella.stage_position.x
        target = widget.canvas.stage_position_at(*_centre(widget))
        target.x = before + 25e-6
        seen = []
        experiment.positions.events.changed.connect(lambda *a: seen.append(a))
        widget.canvas.position_move_requested.emit(lamella.name, target)
        assert lamella.stage_position.x == pytest.approx(before + 25e-6)
        assert seen, "a move re-marks the Overview tabs through the changed event"

        widget.lamella_list.remove_requested.emit(lamella)
        assert [p.name for p in experiment.positions] == ["elsewhere"]
        assert _names(widget) == []
        assert widget.count_label.text() == "Positions on grid-oak · 0"

    def test_without_a_microscope_the_add_is_refused(self, widget, ui, grid):
        widget.set_autolamella_ui(None)
        cx, cy = _centre(widget)
        widget.canvas.request_add_at(cx, cy)
        assert len(ui.experiment.positions) == 2


class TestFromTheGridsTab:
    @pytest.fixture
    def tab(self, ui, grid, destroy_widgets_after_test):
        tab = GridsTabWidget(synchronous=True)
        tab.set_autolamella_ui(ui)
        tab.set_microscope(ui.microscope)
        tab.set_experiment(ui.experiment)
        tab.resize(1200, 700)
        tab.show()
        _app.processEvents()
        yield tab
        tab.close()

    def test_the_positions_chip_follows_the_card_selection(self, tab, grid):
        """Reloading an experiment rebuilds the cards with nothing selected; the
        chip then has to come alive on the click, not on the next inventory
        refresh."""
        assert not tab.view_chips[VIEW_POSITIONS].isEnabled()
        tab.cards._on_card_clicked(grid)
        assert tab.view_chips[VIEW_POSITIONS].isEnabled()
        tab.show_view(VIEW_POSITIONS)
        assert tab.view == VIEW_POSITIONS
        tab.cards._on_card_clicked(grid)  # deselect
        assert not tab.view_chips[VIEW_POSITIONS].isEnabled()
        assert tab.view == VIEW_RESULTS
        # A reload: new cards, nothing selected, then a click.
        tab.set_experiment(tab._experiment)
        assert not tab.view_chips[VIEW_POSITIONS].isEnabled()
        tab.cards._on_card_clicked(tab._experiment.grids[0])
        assert tab.view_chips[VIEW_POSITIONS].isEnabled()

    def test_a_results_row_opens_the_positions_view_on_that_overview(self, tab, grid):
        assert tab.view == VIEW_RESULTS
        assert not tab.view_chips[VIEW_POSITIONS].isEnabled()
        tab.cards._on_card_clicked(grid)
        rows = {r.state.name: r for r in tab.results_widget.rows}
        rows["overview_fib"].btn_mark.click()
        assert tab.view == VIEW_POSITIONS
        assert tab.positions_widget.grid is grid
        assert tab.positions_widget.canvas.view.startswith("FIB @")
        assert tab.view_chips[VIEW_POSITIONS].isChecked()

        tab.positions_widget.btn_done.click()
        assert tab.view == VIEW_RESULTS
        assert tab.view_chips[VIEW_RESULTS].isChecked()

    def test_the_load_button_asks_the_tab_for_the_grid(self, tab, ui):
        """A grid the loader holds can be brought onto the stage from here; the
        banner follows the inventory."""
        experiment, stage = ui.experiment, ui.microscope._stage
        experiment.sync_grids_from_inventory(stage)
        record = experiment.get_grid_by_name("Grid-02")
        tab.refresh()
        tab.cards._on_card_clicked(record)
        tab.show_view(VIEW_POSITIONS)
        positions = tab.positions_widget
        assert positions.grid is record
        assert positions.btn_load.isVisible()
        positions.btn_load.click()
        assert stage.loaded_grids and stage.loaded_grids[0].name == "Grid-02"
        assert positions.state_label.text() == "No stored overview · on the stage"
        assert not positions.btn_load.isVisible()


class TestNewOverviewsArrive:
    def test_an_overview_saved_while_the_grid_is_showing_is_placed_on_refresh(
        self, widget, ui, grid
    ):
        """A grid task finishing while its Positions view is open: the next
        refresh places the new image, keeps what was placed, and keeps the view
        the user was on."""
        experiment, microscope = ui.experiment, ui.microscope
        widget.canvas.show_view(VIEW_FM)
        before = len(widget.canvas.overviews)

        root = experiment.grid_path(grid)
        (root / "overview_sem_2").mkdir()
        sem = microscope.get_stage_position()
        later = FibsemStagePosition(
            x=sem.x + 150e-6, y=sem.y, z=sem.z, r=sem.r, t=sem.t
        )
        _save_beam_overview(
            microscope, root / "overview_sem_2/overview.tif", BeamType.ELECTRON, later
        )
        grid.task_history.append(_entry("overview_sem", "overview_sem_2/overview.tif"))

        widget.refresh()
        assert len(widget.canvas.overviews) == before + 1
        assert widget.canvas.view == VIEW_FM  # the user's view is kept
        # And a second refresh reads nothing again.
        widget.refresh()
        assert len(widget.canvas.overviews) == before + 1

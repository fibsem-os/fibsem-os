"""The Grids tab: cards for the experiment's records, chips from the hardware."""

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Experiment,
    GridQuality,
    GridRecord,
)
from fibsem.applications.autolamella.ui.grid_card_widget import grid_headline
from fibsem.applications.autolamella.ui.grids_tab_widget import GridsTabWidget
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
)
from fibsem.microscopes._stage import SampleGrid, _create_sample_stage


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    return exp


@pytest.fixture
def tab(qapp, arctis, experiment):
    widget = GridsTabWidget(synchronous=True)
    widget.set_microscope(arctis)
    widget.set_experiment(experiment)
    return widget


def entry(status, name="overview_sem"):
    state = AutoLamellaTaskState(name=name, status=status)
    state.end_timestamp = state.start_timestamp + 1
    return state


class TestHeadline:
    def test_nothing_yet(self):
        assert grid_headline(GridRecord(name="g"))[0] == "Not run"

    def test_complete_after_a_load(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.Completed),
            entry(AutoLamellaTaskStatus.Completed, "overview_fib"),
        ]
        assert grid_headline(grid)[0].startswith("overview_fib (")

    def test_failed_tasks_are_counted(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.Failed),
            entry(AutoLamellaTaskStatus.Completed, "overview_fib"),
        ]
        assert grid_headline(grid)[0] == "1 task failed"

    def test_a_failed_load_with_nothing_after_it(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.Completed),
            entry(AutoLamellaTaskStatus.Failed, LOAD_ENTRY_NAME),
        ]
        assert grid_headline(grid)[0] == "Load failed"

    def test_a_run_in_progress(self):
        grid = GridRecord(name="g")
        grid.task_state.name = "overview_fm"
        grid.task_state.status = AutoLamellaTaskStatus.InProgress
        assert grid_headline(grid)[0] == "Running overview_fm"


class TestCards:
    def test_empty_experiment_invites_an_inventory(self, tab):
        assert tab.cards.cards == []
        assert "Run inventory" in tab.summary_label.text()
        assert tab.btn_inventory.isEnabled()

    def test_inventory_creates_a_card_per_present_grid(self, tab, experiment):
        tab.btn_inventory.click()
        assert [c.grid.name for c in tab.cards.cards] == [
            "Grid-01",
            "Grid-02",
            "Grid-03",
        ]
        assert tab.summary_label.text() == "3 in this experiment · 3 present"
        card = tab.cards.cards[0]
        assert [c.text() for c in card._chip_widgets] == []  # present, not in beam
        assert "slot 01" in card._status_label.toolTip()
        assert card.is_present and card.status_text == "Not run"
        assert card._action_load.isVisible() and card._action_load.isEnabled()
        assert not card._action_unload.isVisible()
        assert tab.status_label.text() == "Inventory complete."
        # saved as it went
        assert (
            len(Experiment.load(Path(experiment.path) / "experiment.yaml").grids) == 3
        )

    def test_a_record_whose_hardware_has_gone_is_kept_dimmed(self, tab, experiment):
        experiment.add_grid(GridRecord(name="grid-oak"))
        tab.set_experiment(experiment)
        (card,) = tab.cards.cards
        assert [c.text() for c in card._chip_widgets] == ["not present"]
        assert not card._action_load.isEnabled()
        assert tab.summary_label.text() == "1 in this experiment · 0 present"

    def test_load_and_unload_follow_the_card(self, tab, arctis):
        tab.btn_inventory.click()
        card = tab.cards.cards[1]
        card._action_load.trigger()
        assert arctis._stage.loaded_grids[0].name == "Grid-02"
        assert [c.text() for c in card._chip_widgets] == ["Loaded"]
        assert card._action_unload.isVisible() and not card._action_load.isVisible()
        assert tab.status_label.text() == "Grid-02 is in the beam."
        card._action_unload.trigger()
        assert arctis._stage.loaded_grids == []
        assert [c.text() for c in card._chip_widgets] == []

    def test_a_refused_exchange_is_reported(self, tab, arctis):
        tab.btn_inventory.click()
        arctis._stage.loader.fail_next_exchange = True
        tab.cards.cards[0]._action_load.trigger()
        assert "Simulated autoloader exchange failure" in tab.status_label.text()
        assert tab.cards.cards[0]._action_load.isEnabled()

    def test_quality_is_set_on_the_record_and_saved(self, tab, experiment):
        tab.btn_inventory.click()
        changed = []
        tab.experiment_changed.connect(lambda: changed.append(True))
        card = tab.cards.cards[0]
        card.set_quality(GridQuality.GOOD)
        assert card.grid.quality is GridQuality.GOOD
        assert "Good" in card._btn_quality.toolTip()
        assert changed == [True]
        loaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert loaded.get_grid_by_name("Grid-01").quality is GridQuality.GOOD
        # a task outcome does not touch it
        assert grid_headline(card.grid)[0] == "Not run"

    def test_rename_writes_through_to_the_slot(self, tab, arctis, experiment):
        tab.btn_inventory.click()
        card = tab.cards.cards[2]
        tab._on_rename(card.grid, "grid-cedar")
        assert card.grid.name == "grid-cedar"
        assert arctis._stage.loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
        assert card._name_label.text() == "grid-cedar"
        assert "slot 03" in card._status_label.toolTip()
        assert experiment.get_grid_by_name("grid-cedar") is card.grid

    def test_rename_refuses_a_duplicate(self, tab):
        tab.btn_inventory.click()
        tab._on_rename(tab.cards.cards[0].grid, "Grid-02")
        assert tab.cards.cards[0].grid.name == "Grid-01"
        assert "already a grid named" in tab.status_label.text()

    def test_selection_toggles_and_is_announced(self, tab):
        tab.btn_inventory.click()
        picked = []
        tab.grid_selected.connect(picked.append)
        tab.cards._on_card_clicked(tab.cards.cards[0].grid)
        assert tab.selected_grid.name == "Grid-01"
        tab.cards._on_card_clicked(tab.cards.cards[0].grid)
        assert tab.selected_grid is None
        assert [p.name if p else None for p in picked] == ["Grid-01", None]

    def test_the_host_can_lock_the_hardware_conveniences(self, tab):
        tab.btn_inventory.click()
        tab.set_controls_enabled(False)
        assert not tab.btn_inventory.isEnabled()
        assert not tab.cards.cards[0]._action_load.isEnabled()
        assert not tab.cards.cards[0]._action_rename.isEnabled()

    def test_remove_stops_tracking_the_grid(self, tab, experiment):
        tab.btn_inventory.click()
        tab.cards.remove_requested.emit(tab.cards.cards[0].grid)
        assert [g.name for g in experiment.grids] == ["Grid-02", "Grid-03"]
        assert [c.grid.name for c in tab.cards.cards] == ["Grid-02", "Grid-03"]
        assert tab.summary_label.text() == "2 in this experiment · 2 present"
        # still in the magazine: an inventory brings it back as a fresh record
        tab.btn_inventory.click()
        assert [g.name for g in experiment.grids] == ["Grid-02", "Grid-03", "Grid-01"]

    def test_a_card_shows_the_latest_overview_thumbnail(self, tab, experiment):
        import numpy as np

        from fibsem.applications.autolamella.structures import AutoLamellaTaskState
        from fibsem.imaging.thumbnail import write_thumbnail

        tab.btn_inventory.click()
        grid = experiment.get_grid_by_name("Grid-01")
        root = experiment.grid_path(grid)
        write_thumbnail(
            (np.random.rand(200, 300) * 255).astype(np.uint8),
            root / "overview_sem" / "overview-thumbnail.png",
        )
        state = AutoLamellaTaskState(
            name="overview_sem", status=AutoLamellaTaskStatus.Completed
        )
        state.outputs = {
            "overview_sem_thumbnail": ["overview_sem/overview-thumbnail.png"]
        }
        grid.task_history.append(state)
        card = tab.cards.card_for(grid)
        assert card._thumb_label.pixmap() is None or card._thumb_label.pixmap().isNull()
        card.refresh()
        assert not card._thumb_label.pixmap().isNull()
        assert card.status_text.startswith("overview_sem (")
        # cozy by default: the big thumbnail; the standard row keeps the small one
        assert card.mode == "cozy" and card._thumb_label.height() == 170
        tab.cards.set_mode("standard")
        assert (
            card._thumb_label.height() == 44 and not card._thumb_label.pixmap().isNull()
        )
        tab.cards.set_mode("cozy")


def test_on_a_fixed_holder_there_is_nothing_to_load(qapp, experiment):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    microscope._stage.holder.slots["Slot-01"].loaded_grid = SampleGrid(
        name="grid-aspen"
    )
    tab = GridsTabWidget(synchronous=True)
    tab.set_microscope(microscope)
    tab.set_experiment(experiment)
    tab.btn_inventory.click()  # a plain refresh: no scan, no confirmation
    (card,) = tab.cards.cards
    assert card.grid.name == "grid-aspen"
    assert [c.text() for c in card._chip_widgets] == ["Loaded"]
    assert not card._action_load.isVisible() and not card._action_unload.isVisible()


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    window = module.AutoLamellaSingleWindowUI()
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


class _NoMinimap:
    def set_experiment(self) -> None:
        pass


def test_the_tab_sits_behind_the_flag_between_lamella_and_workflow(main_ui):
    tabs = main_ui.tab_widget
    index = tabs.indexOf(main_ui.grids_tab)
    labels = [tabs.tabText(i) for i in range(tabs.count())]
    assert labels[index - 1] == "Lamella" and labels[index + 1] == "Workflow"
    # Both states set outright: the window loads (and on close saves) the
    # preference file, so "off by default" would read whatever the last run left.
    was = main_ui._preferences.features.grid_workflow
    try:
        main_ui._preferences.features.grid_workflow = False
        main_ui._apply_grid_workflow_visibility()
        assert not tabs.isTabVisible(index)
        main_ui._preferences.features.grid_workflow = True
        main_ui._apply_grid_workflow_visibility()
        assert tabs.isTabVisible(index)
    finally:
        main_ui._preferences.features.grid_workflow = was


def test_the_tab_follows_the_connection_and_the_experiment(main_ui, tmp_path):
    ui = main_ui.autolamella_ui
    assert main_ui.grids_tab.stage is None
    ui.system_widget.connect_to_microscope()
    assert main_ui.grids_tab.stage is ui.microscope._stage
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.add_grid(GridRecord(name="grid-oak"))
    ui.experiment = exp
    # The fixture leaves the napari Minimap tab unbuilt (it owns a viewer that
    # cannot live in a test), so the one call the update makes on it is stood in
    # for. Called rather than emitted: an exception inside a Qt slot aborts the
    # process under PyQt5, and a traceback is worth more than a core dump.
    main_ui.minimap_widget = _NoMinimap()
    main_ui._on_experiment_update()
    assert [c.grid.name for c in main_ui.grids_tab.cards.cards] == ["grid-oak"]
    assert main_ui.tab_widget.isTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab)
    )


def test_an_experiment_with_no_lamella_tasks_loads(main_ui, tmp_path):
    """A grid-only experiment, or one built from an empty protocol: the lamella
    task editor has nothing to show and says so by hiding its columns, where it
    used to raise on the empty selection and stop the whole experiment load."""
    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.minimap_widget = _NoMinimap()
    main_ui._on_experiment_update()
    editor = main_ui.task_widget
    assert not editor.task_parameters_config_widget.isVisibleTo(editor)
    assert not editor.milling_task_editor.isVisibleTo(editor)

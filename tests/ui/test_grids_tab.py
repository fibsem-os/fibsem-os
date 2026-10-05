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
        assert grid_headline(GridRecord(name="g"))[0] == ""

    def test_a_load_with_no_task_after_it_says_nothing_either(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed),
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
        ]
        assert grid_headline(grid)[0] == ""

    def test_complete_after_a_load(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.Completed),
            entry(AutoLamellaTaskStatus.Completed, "overview_fib"),
        ]
        assert grid_headline(grid)[0].startswith("overview_fib (")

    def test_a_task_awaiting_a_decision_is_the_headline(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.AwaitingDecision),
            entry(AutoLamellaTaskStatus.Failed, "overview_fib"),
        ]
        assert grid_headline(grid)[0] == "overview_sem awaits a decision"

    def test_it_is_found_across_a_later_load(self):
        """A run that moved on and came back loads the grid again; the task
        waiting from before that load is still the news."""
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.AwaitingDecision),
            entry(AutoLamellaTaskStatus.AwaitingDecision, "overview_fib"),
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.Completed, "overview_fm"),
        ]
        assert grid_headline(grid)[0] == "2 tasks await a decision"

    def test_a_decided_task_is_not_waiting(self):
        grid = GridRecord(name="g")
        grid.task_history += [
            entry(AutoLamellaTaskStatus.Completed, LOAD_ENTRY_NAME),
            entry(AutoLamellaTaskStatus.AwaitingDecision),
            entry(AutoLamellaTaskStatus.Completed),  # re-run, finished
        ]
        assert grid_headline(grid)[0].startswith("overview_sem (")

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
        assert [c.text() for c in card._chip_widgets] == []  # present, not loaded
        assert card._status_label.toolTip() == "slot 01"  # no "Not run" to join
        assert card.is_present and card.status_text == ""
        assert card._action_load.isVisible() and card._action_load.isEnabled()
        assert not card._action_unload.isVisible()
        assert tab.status_label.text() == "Inventory read."
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
        assert tab.status_label.text() == "Grid-02 is loaded."
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
        assert card.grid.quality.verdict is GridQuality.GOOD
        assert "Good" in card._btn_quality.toolTip()
        assert changed == [True]
        loaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert loaded.get_grid_by_name("Grid-01").quality.verdict is GridQuality.GOOD
        # a task outcome does not touch it
        assert grid_headline(card.grid)[0] == ""

    def test_rename_writes_through_to_the_slot(self, tab, arctis, experiment):
        tab.btn_inventory.click()
        card = tab.cards.cards[2]
        tab._on_rename(card.grid, "grid-cedar")
        assert card.grid.name == "grid-cedar"
        assert arctis._stage.loader.slots["Slot-03"].loaded_grid.name == "grid-cedar"
        assert card._name_label.text() == "grid-cedar"
        assert "slot 03" in card._status_label.toolTip()
        assert experiment.get_grid_by_name("grid-cedar") is card.grid

    def test_a_slot_that_keeps_its_name_keeps_the_record_too(
        self, tab, arctis, monkeypatch
    ):
        """The next inventory read would bring the slot's name back, so a write
        that did not take leaves the record as it was, and says why."""
        tab.btn_inventory.click()

        def refuse(slot_name, grid, persist=False):
            raise RuntimeError("the name did not stick")

        monkeypatch.setattr(arctis._stage, "assign_grid", refuse)
        card = tab.cards.cards[0]
        tab._on_rename(card.grid, "grid-aspen")
        assert card.grid.name == "Grid-01"
        assert "did not stick" in tab.status_label.text()

    def test_rename_refuses_a_duplicate(self, tab):
        tab.btn_inventory.click()
        tab._on_rename(tab.cards.cards[0].grid, "Grid-02")
        assert tab.cards.cards[0].grid.name == "Grid-01"
        assert "already a grid named" in tab.status_label.text()

    def test_a_grid_that_has_run_keeps_its_name(self, tab, arctis):
        """Its folder and images carry the name, so Rename is greyed out with the
        reason, and the rename itself is refused for any other way in."""
        tab.btn_inventory.click()
        card = tab.cards.cards[0]
        assert card._action_rename.isEnabled()
        card.grid.task_history.append(entry(AutoLamellaTaskStatus.Completed))
        card.refresh()
        assert not card._action_rename.isEnabled()
        assert "Named once it has run" in card._action_rename.toolTip()

        tab._on_rename(card.grid, "grid-aspen")
        assert card.grid.name == "Grid-01"
        assert arctis._stage.loader.slots["Slot-01"].loaded_grid.name == "Grid-01"
        assert "cannot be renamed" in tab.status_label.text()

    def test_a_grid_only_loaded_can_still_be_renamed(self, tab):
        """A load writes nothing under the grid's name, so it does not fix it."""
        tab.btn_inventory.click()
        card = tab.cards.cards[0]
        card.grid.task_history.append(
            entry(AutoLamellaTaskStatus.Completed, name=LOAD_ENTRY_NAME)
        )
        card.refresh()
        assert card._action_rename.isEnabled()
        tab._on_rename(card.grid, "grid-aspen")
        assert card.grid.name == "grid-aspen"

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


def test_the_tab_sits_between_lamella_and_workflow_for_everyone(main_ui):
    tabs = main_ui.tab_widget
    index = tabs.indexOf(main_ui.grids_tab)
    labels = [tabs.tabText(i) for i in range(tabs.count())]
    assert labels[index - 1] == "Lamella" and labels[index + 1] == "Workflow"
    assert tabs.isTabVisible(index)


def test_the_grid_workflow_flag_is_gone():
    """The Grids tab, the Workflow tab's Grids view, the Protocol tab's Grid page and
    the grid report were behind `features.grid_workflow`; they ship to everyone now.

    Removed rather than defaulted on: every preferences save writes every key, so a
    changed default would reach only machines that have never opened an experiment.
    A saved file still carrying the key, either value, must load: `_sub_from_dict`
    drops unknown keys, checked here rather than assumed, because the failure would
    be at every startup.
    """
    assert not hasattr(cfg.FeatureFlags(), "grid_workflow")
    for value in (False, True):
        stale = {"features": {"grid_workflow": value}}
        assert cfg.UserPreferences.from_dict(stale) is not None


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


def test_a_grid_task_added_on_the_protocol_tab_reaches_the_run_view(main_ui, tmp_path):
    """The Workflow tab's Grids view lists the protocol's grid tasks; a task
    added on the Protocol tab's Grid page used to appear there only after an
    inventory or an experiment reload."""
    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.minimap_widget = _NoMinimap()
    main_ui._on_experiment_update()
    run_view = main_ui.grid_workflow_widget
    assert list(run_view._task_rows) == []

    main_ui.task_widget.grid_protocol.add_task("BEAM_OVERVIEW_GRID", "SEM Overview")

    assert list(run_view._task_rows) == ["SEM Overview"]
    assert run_view.get_selected_task_names() == ["SEM Overview"]


def test_a_load_from_a_card_reaches_the_sample_view(main_ui, tmp_path):
    """Seen on the bench: unloading from the Grids tab left the Sample view
    showing the grid still on the stage. The Sample view draws from the stage
    and never polls it, so the Grids tab's exchanges have to tell it."""
    from fibsem.microscopes._stage import DemoSampleLoader
    from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget

    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)
    microscope._stage.loader = DemoSampleLoader(microscope, occupied=(1, 2))
    # The Sample view is built at connect, against the stage of that moment;
    # rebuild it for the swapped stage the way a connect would.
    ui.sample_widget = FibsemSampleWidget(microscope=microscope)
    main_ui._refresh_grids_tab_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.grids_tab.set_experiment(exp)
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )
    main_ui.grids_tab._synchronous = True
    main_ui.grids_tab.btn_inventory.click()

    def sample_states():
        return [r.state for r in ui.sample_widget.loader_widget._rows[:2]]

    assert sample_states() == ["occupied", "occupied"]
    card = main_ui.grids_tab.cards.cards[1]
    card._action_load.trigger()
    assert sample_states() == ["occupied", "loaded"]
    card._action_unload.trigger()
    assert sample_states() == ["occupied", "occupied"]


def _window_with_magazine(main_ui, tmp_path):
    """The window on a simulated autoloader with grids in slots 1-3, an
    experiment open, and the Grids tab's inventory read."""
    from fibsem.microscopes._stage import DemoSampleLoader
    from fibsem.ui.FibsemSampleWidget import FibsemSampleWidget

    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_is_compustage = True
    microscope._stage = _create_sample_stage(microscope)
    microscope._stage.loader = DemoSampleLoader(microscope, occupied=(1, 2, 3))
    ui.sample_widget = FibsemSampleWidget(microscope=microscope)
    main_ui._refresh_grids_tab_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.grids_tab.set_experiment(exp)
    main_ui.grid_workflow_widget.set_experiment(exp)
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )
    main_ui.grids_tab._synchronous = True
    main_ui.grids_tab.btn_inventory.click()
    assert [g.name for g in exp.grids] == ["Grid-01", "Grid-02", "Grid-03"]
    return ui, microscope, exp


def _name_on_the_sample_view(ui, index, name):
    row = ui.sample_widget.loader_widget._row_widget(index)
    row.name_edit.setText(name)
    row.name_edit.editingFinished.emit()


class TestNamingOnTheSampleView:
    """The Sample view names grids on the hardware; the experiment's record has
    to follow, or the next inventory sync adds a second record."""

    def test_naming_a_slot_renames_its_record(self, main_ui, tmp_path):
        ui, microscope, exp = _window_with_magazine(main_ui, tmp_path)
        record = exp.get_grid_by_name("Grid-02")
        _name_on_the_sample_view(ui, 1, "grid-birch")
        assert [g.name for g in exp.grids] == ["Grid-01", "grid-birch", "Grid-03"]
        assert exp.get_grid_by_name("grid-birch") is record
        assert (
            microscope._stage.loader.slots["Slot-02"].loaded_grid.name == "grid-birch"
        )
        assert [c.grid.name for c in main_ui.grids_tab.cards.cards][1] == "grid-birch"
        saved = Experiment.load(Path(exp.path) / "experiment.yaml")
        assert [g.name for g in saved.grids] == ["Grid-01", "grid-birch", "Grid-03"]

    def test_a_grid_that_has_run_is_refused(self, main_ui, tmp_path):
        ui, microscope, exp = _window_with_magazine(main_ui, tmp_path)
        exp.get_grid_by_name("Grid-01").task_history.append(
            entry(AutoLamellaTaskStatus.Completed)
        )
        _name_on_the_sample_view(ui, 0, "grid-aspen")
        assert [g.name for g in exp.grids] == ["Grid-01", "Grid-02", "Grid-03"]
        assert microscope._stage.loader.slots["Slot-01"].loaded_grid.name == "Grid-01"
        assert "cannot be renamed" in ui.sample_widget.loader_widget.status_label.text()

    def test_a_name_another_grid_has_is_refused(self, main_ui, tmp_path):
        ui, microscope, exp = _window_with_magazine(main_ui, tmp_path)
        _name_on_the_sample_view(ui, 0, "Grid-02")
        assert [g.name for g in exp.grids] == ["Grid-01", "Grid-02", "Grid-03"]
        assert microscope._stage.loader.slots["Slot-01"].loaded_grid.name == "Grid-01"

    def test_nothing_is_renamed_while_a_workflow_runs(self, main_ui, tmp_path):
        """The run's queue holds grids by name."""
        ui, microscope, exp = _window_with_magazine(main_ui, tmp_path)

        class _Running:
            def is_alive(self):
                return True

        ui._task_worker_thread = _Running()
        try:
            _name_on_the_sample_view(ui, 1, "grid-birch")
        finally:
            ui._task_worker_thread = None  # or closing the window waits on it
        assert [g.name for g in exp.grids] == ["Grid-01", "Grid-02", "Grid-03"]
        assert "while a workflow is running" in (
            ui.sample_widget.loader_widget.status_label.text()
        )


class TestReport:
    """Writes the grid screening PDF under the experiment and opens it. Tools →
    Reporting calls this; see test_grid_report_menu.py for the menu."""

    def test_needs_a_record_but_no_hardware(self, qapp, experiment):
        widget = GridsTabWidget(synchronous=True)
        widget.set_experiment(experiment)
        widget.generate_report()
        assert widget.status_label.text() == "No grids to report. Run inventory first."

    def test_writes_under_the_experiment_and_opens_it(
        self, tab, experiment, monkeypatch
    ):
        pytest.importorskip("reportlab")
        tab.btn_inventory.click()
        opened = []
        monkeypatch.setattr(tab, "open_report", opened.append)
        tab.generate_report()
        (path,) = opened
        assert path == os.path.join(str(experiment.path), "grid-screening-report.pdf")
        assert os.path.getsize(path) > 0
        assert tab.status_label.text() == "Report written: grid-screening-report.pdf"
        assert not tab.busy

    def test_a_missing_reporting_extra_is_said_plainly(self, tab, monkeypatch):
        import fibsem.applications.autolamella.ui.grids_tab_widget as module

        tab.btn_inventory.click()

        def refuse(*_args, **_kwargs):
            raise ImportError("No module named 'reportlab'")

        monkeypatch.setattr(module, "generate_grid_report", refuse)
        tab.generate_report()
        assert "pip install fibsem-os[reporting]" in tab.status_label.text()


class TestNote:
    """The operator's note on a grid (FIB-1132): written from the card's actions
    menu, kept on the record, and printed by the grid screening report."""

    NOTE = "Even ice across the centre, cells on the east half."

    @pytest.fixture
    def card(self, tab):
        tab.btn_inventory.click()
        card = tab.cards.cards[0]
        tab.select_grid(card.grid)
        return card

    @staticmethod
    def type_note(monkeypatch, text, ok=True):
        import fibsem.applications.autolamella.ui.grid_card_widget as module

        monkeypatch.setattr(
            module.QInputDialog,
            "getMultiLineText",
            lambda *args, **kwargs: (text, ok),
        )

    def test_it_is_saved_and_shown(self, tab, card, experiment, monkeypatch):
        changed = []
        tab.experiment_changed.connect(lambda: changed.append(True))
        self.type_note(monkeypatch, f"  {self.NOTE}\n")
        card._action_note.trigger()

        assert card.grid.description == self.NOTE  # trimmed
        assert changed == [True]
        assert card._card.toolTip() == self.NOTE
        assert self.NOTE in tab.results_widget.subtitle_label.text()
        loaded = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert loaded.get_grid_by_name(card.grid.name).description == self.NOTE

    def test_a_verdict_keeps_the_note(self, card, monkeypatch):
        self.type_note(monkeypatch, self.NOTE)
        card._action_note.trigger()
        card.set_quality(GridQuality.GOOD)
        assert card.grid.description == self.NOTE

    def test_an_empty_note_clears_it(self, card, monkeypatch):
        self.type_note(monkeypatch, self.NOTE)
        card._action_note.trigger()
        self.type_note(monkeypatch, "   ")
        card._action_note.trigger()
        assert card.grid.description == ""
        assert card._card.toolTip() == ""

    def test_cancel_changes_nothing(self, tab, card, monkeypatch):
        changed = []
        tab.experiment_changed.connect(lambda: changed.append(True))
        self.type_note(monkeypatch, self.NOTE, ok=False)
        card._action_note.trigger()
        assert card.grid.description == ""
        assert changed == []

    def test_it_reaches_the_report(self, tab, card, experiment, monkeypatch):
        from fibsem.applications.autolamella.tools.grid_report import (
            collect_grid_report,
        )

        self.type_note(monkeypatch, self.NOTE)
        card._action_note.trigger()
        experiment.task_protocol = AutoLamellaTaskProtocol()
        report = collect_grid_report(experiment)
        assert report.sections[0].description == self.NOTE

        pytest.importorskip("reportlab")
        from fibsem.applications.autolamella.tools.grid_report_pdf import (
            generate_grid_report,
        )

        path = generate_grid_report(
            experiment,
            output_path=str(Path(experiment.path) / "report.pdf"),
            compress=False,
        )
        # Once on the cover, once under the grid's name. The cover's column wraps
        # the note, so look for its two ends rather than the whole string.
        pdf = Path(path).read_bytes()
        assert pdf.count(b"Even ice across") == 2
        assert pdf.count(b"east half.") == 2

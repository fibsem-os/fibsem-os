"""Workflow · Grids: select grids and tasks, confirm, run on the window's worker."""

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtTest import QTest

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskStatus,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.task_outputs import grid_outputs
from fibsem.applications.autolamella.ui.grid_workflow_widget import (
    GridRunPreflightDialog,
    GridWorkflowWidget,
)
from fibsem.applications.autolamella.workflows.tasks.grid import (
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.microscopes._stage import SampleGrid, SlotCalibration, _create_sample_stage
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
)


def _small_settings(beam=BeamType.ELECTRON) -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(resolution=(128, 128), hfw=200e-6, beam_type=beam),
        nrows=1,
        ncols=1,
    )


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
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_sem", settings=_small_settings())
    )
    exp.grid_protocol.add(FluorescenceOverviewGridTaskConfig(task_name="overview_fm"))
    return exp


@pytest.fixture
def view(qapp, arctis, experiment):
    experiment.sync_grids_from_inventory(arctis._stage)
    experiment.add_grid(GridRecord(name="grid-oak"))  # not in the magazine
    widget = GridWorkflowWidget()
    widget.set_microscope(arctis)
    widget.set_experiment(experiment)
    return widget


class TestSelection:
    def test_empty_lists_say_what_to_do(self, qapp, arctis, tmp_path):
        exp = Experiment(path=tmp_path, name="exp")
        (tmp_path / "exp").mkdir()
        exp.task_protocol = AutoLamellaTaskProtocol()
        widget = GridWorkflowWidget()
        widget.set_microscope(arctis)
        widget.set_experiment(exp)
        assert (
            not widget.grid_empty.isHidden() and "inventory" in widget.grid_empty.text()
        )
        assert (
            not widget.task_empty.isHidden()
            and "Protocol tab" in widget.task_empty.text()
        )
        assert not widget.btn_screen_all.isEnabled()  # nothing to run yet

    def test_rows_and_defaults(self, view):
        assert view.grid_empty.isHidden() and view.task_empty.isHidden()
        assert list(view._grid_rows) == ["Grid-01", "Grid-02", "Grid-03", "grid-oak"]
        assert view.grid_header.trailing.text() == "3 of 4 present"
        assert not view._grid_rows["grid-oak"].checkbox.isEnabled()
        # every task ticked by default, in the protocol's order
        assert view.get_selected_task_names() == ["overview_sem", "overview_fm"]
        assert view.get_selected_grids() == []
        assert view.summary_label.text() == "0 grids, 2 tasks selected"

    def test_select_all_ticks_only_present_grids(self, view):
        view.grid_header.select_all.setChecked(True)
        assert [g.name for g in view.get_selected_grids()] == [
            "Grid-01",
            "Grid-02",
            "Grid-03",
        ]
        assert view.summary_label.text() == "3 grids, 2 tasks selected · 3 exchanges"

    def test_a_grid_in_the_beam_costs_no_exchange(self, view, arctis):
        arctis._stage.ensure_loaded("Grid-02")
        view.refresh()
        view.grid_header.select_all.setChecked(True)
        assert view.exchanges_for(view.get_selected_grids()) == 2
        assert [c.text() for c in view._grid_rows["Grid-02"]._chip_widgets] == [
            "slot 02",
            "in beam",
        ]

    def test_the_fm_task_is_greyed_without_a_fluorescence_microscope(
        self, qapp, experiment
    ):
        plain, _ = utils.setup_session(manufacturer="Demo")
        # Said outright: a Demo session set up after an Arctis one in the same
        # process keeps the Arctis FM (the default configuration is shared), so
        # the plain Demo is not reliably FM-less by itself.
        plain.fm = None
        widget = GridWorkflowWidget()
        widget.set_microscope(plain)
        widget.set_experiment(experiment)
        row = widget._task_rows["overview_fm"]
        assert not row.checkbox.isEnabled()
        assert "no fluorescence microscope" in row.detail_label.text()
        assert widget.get_selected_task_names() == ["overview_sem"]

    def test_reordering_tasks_writes_the_protocol(self, view, experiment):
        view._task_rows["overview_fm"].btn_up.click()
        assert view.get_selected_task_names() == ["overview_fm", "overview_sem"]
        assert experiment.grid_protocol.ordered_task_names == [
            "overview_fm",
            "overview_sem",
        ]
        again = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert again.grid_protocol.ordered_task_names == ["overview_fm", "overview_sem"]

    def test_screen_all_needs_a_task_and_a_stage(self, view):
        assert view.btn_screen_all.isEnabled()
        assert view.task_header.select_all.isChecked()  # reads the rows
        view.task_header.select_all.setChecked(False)
        assert view.get_selected_task_names() == []
        assert not view.btn_screen_all.isEnabled()
        view.set_all_tasks_selected(True)
        view.set_controls_enabled(False)
        assert not view.btn_screen_all.isEnabled()


def test_the_preflight_says_what_a_run_does(qapp):
    dialog = GridRunPreflightDialog(
        ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"], 2, "/exp"
    )
    assert dialog.windowTitle() == "Run grid workflow"
    dialog = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", screen_all=True
    )
    assert dialog.windowTitle() == "Screen all grids"


@pytest.fixture
def main_ui(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    original = module.AutoLamellaSingleWindowUI.add_minimap_tab
    module.AutoLamellaSingleWindowUI.add_minimap_tab = lambda self: None
    try:
        window = module.AutoLamellaSingleWindowUI()
    finally:
        module.AutoLamellaSingleWindowUI.add_minimap_tab = original
    yield window
    if window.autolamella_ui.microscope is not None:
        window.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        window.close()
    finally:
        qapp.quit = original_quit


def _wait_for_run(ui, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while ui.is_workflow_running and time.monotonic() < deadline:
        QTest.qWait(100)
    assert not ui.is_workflow_running, "the grid run did not finish in time"


def test_the_grids_view_sits_beside_lamella_behind_the_flag(main_ui):
    left = main_ui.workflow_left_tabs
    index = left.indexOf(main_ui.grid_workflow_widget)
    assert left.tabText(index) == "Grids" and left.tabText(0) == "Lamella"
    was = main_ui._preferences.features.grid_workflow
    try:
        main_ui._preferences.features.grid_workflow = False
        main_ui._apply_grid_workflow_visibility()
        assert not left.isTabVisible(index)
        main_ui._preferences.features.grid_workflow = True
        main_ui._apply_grid_workflow_visibility()
        assert left.isTabVisible(index)
    finally:
        main_ui._preferences.features.grid_workflow = was


def test_an_inventory_on_the_grids_tab_reaches_the_run_view(main_ui, tmp_path):
    """The Grids tab creates the records; the Workflow view's rows must follow
    without a reload."""
    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    slot = microscope._stage.holder.slots["Slot-01"]
    slot.position = FibsemStagePosition(
        name=slot.name, x=-4e-3, y=1e-3, z=4e-3, r=0, t=0.61
    )
    slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "test")
    slot.loaded_grid = SampleGrid(name="grid-aspen")
    main_ui._refresh_grids_tab_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.grids_tab.set_experiment(exp)
    main_ui.grid_workflow_widget.set_experiment(exp)
    # the app enables the tab on experiment load; a click on a disabled tab's
    # button is swallowed
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )
    assert main_ui.grid_workflow_widget._grid_rows == {}

    main_ui.grids_tab._synchronous = True
    assert main_ui.grids_tab.btn_inventory.isEnabled()
    main_ui.grids_tab.btn_inventory.click()
    assert [g.name for g in exp.grids] == ["grid-aspen"]
    assert list(main_ui.grid_workflow_widget._grid_rows) == ["grid-aspen"]
    assert main_ui.grid_workflow_widget.grid_empty.isHidden()


def test_a_grid_run_from_the_window_on_a_fixed_holder(main_ui, tmp_path, monkeypatch):
    """End to end: the Run button on the Grids view, the worker, the manager, the
    shared timeline and the record. A fixed holder, so no exchange."""
    ui = main_ui.autolamella_ui
    # The post-run summary is modal; under offscreen it would block forever.
    from fibsem.applications.autolamella.ui import AutoLamellaUI as ui_module

    class _NoDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr(ui_module, "WorkflowSummaryDialog", _NoDialog)
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    slot = microscope._stage.holder.slots["Slot-01"]
    slot.position = FibsemStagePosition(
        name=slot.name, x=-4e-3, y=1e-3, z=4e-3, r=0, t=0.61
    )
    slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "test")
    slot.loaded_grid = SampleGrid(name="grid-aspen")
    main_ui._refresh_grids_tab_microscope()

    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_sem", settings=_small_settings())
    )
    exp.sync_grids_from_inventory(microscope._stage)
    ui.experiment = exp
    # The pieces of _on_experiment_update this test needs. The whole handler also
    # rebuilds the lamella task editor, which cannot take a protocol with no
    # lamella tasks (a pre-existing gap; real protocols always have some).
    main_ui.grids_tab.set_experiment(exp)
    main_ui.grid_workflow_widget.set_experiment(exp)
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )

    view = main_ui.grid_workflow_widget
    main_ui.workflow_left_tabs.setCurrentWidget(view)
    view.grid_header.select_all.setChecked(True)
    assert main_ui.run_workflow_btn.isEnabled()
    assert "1 grid, 1 task" in main_ui.run_workflow_btn.toolTip()

    main_ui._start_grid_run(["overview_sem"], ["grid-aspen"], inventory_first=False)
    assert ui.is_workflow_running
    assert not main_ui.grids_tab.btn_inventory.isEnabled()  # locked during the run
    _wait_for_run(ui)
    QTest.qWait(200)  # let the finished signal land

    grid = exp.get_grid_by_name("grid-aspen")
    # a fixed holder: the grid was in the beam already, so no load entry
    assert [t.name for t in grid.task_history] == ["overview_sem"]
    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert len(grid_outputs(exp, grid, "overview_sem")) == 1
    assert main_ui.grids_tab.btn_inventory.isEnabled()
    assert ui._last_run_summary is not None  # the grid summary, for the agent server

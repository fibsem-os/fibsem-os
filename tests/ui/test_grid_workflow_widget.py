"""Workflow · Grids: select grids and tasks, confirm, run on the window's worker."""

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtGui import QFontMetrics
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLabel

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    Attention,
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
from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
    LOAD_ENTRY_NAME,
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
        row = view._grid_rows["Grid-02"]
        assert [c.text() for c in row._chip_widgets] == ["Loaded"]
        assert row.slot_label.text() == "slot 02"

    def test_a_loaded_row_keeps_its_slot_inside_the_list(self, view, arctis):
        """The Loaded pill once pushed the slot column past the list's edge, and
        the slot read as "s". The row must ask for less than the panel gives."""
        arctis._stage.ensure_loaded("Grid-02")
        view.refresh()
        row = view._grid_rows["Grid-02"]
        assert [c.text() for c in row._chip_widgets] == ["Loaded"]
        assert row.sizeHint().width() <= 380

    def test_an_unscanned_magazine_says_so_on_every_row(self, view, arctis):
        """Before an inventory the rows are UNKNOWN: the chip says the magazine
        has not been read, not that the grids are missing, and none can be run."""
        arctis._stage.loader.scanned = False
        view.refresh()
        rows = [view._grid_rows[n] for n in ("Grid-01", "Grid-02", "Grid-03")]
        assert all([c.text() for c in r._chip_widgets] == ["not scanned"] for r in rows)
        assert not any(r.checkbox.isEnabled() for r in rows)
        arctis._stage.loader.scanned = True
        view.refresh()
        assert all(r.checkbox.isEnabled() for r in rows)

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
        # The reason is longer than its column. It is elided from the right so
        # the start survives; a plain right-aligned label lost it off the left.
        widget.resize(420, 600)  # the Workflow tab's left panel, near enough
        widget.show()
        QTest.qWait(50)
        drawn = QLabel.text(row.detail_label)
        assert drawn != row.detail_label.text() and drawn.endswith("\u2026")
        assert drawn.startswith(row.detail_label.text()[:8])
        assert (
            QFontMetrics(row.detail_label.font()).horizontalAdvance(drawn)
            <= row.detail_label.width()
        )
        # And the column sits inside the row, before the drag handle: left to
        # its Ignored policy the layout pushed it past the row's right edge.
        assert row.detail_label.geometry().right() <= row.drag_handle.geometry().left()
        assert row.drag_handle.geometry().right() <= row.width()
        widget.close()

    def test_reordering_tasks_writes_the_protocol(self, view, experiment):
        view._on_reordered(["overview_fm", "overview_sem"])  # what a drop reports
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


class TestAttentionChip:
    """Automated or Review per grid task, as on the lamella task list; shown
    only while the review preference is on."""

    def test_hidden_while_review_is_off(self, view, monkeypatch):
        import fibsem.applications.autolamella.ui.grid_workflow_widget as module

        monkeypatch.setattr(module, "_review_available", lambda: False)
        row = view._task_rows["overview_sem"]
        row.refresh()
        assert row.btn_attention.isHidden()

    def test_a_click_toggles_review_and_saves_the_protocol(
        self, view, experiment, monkeypatch
    ):
        import fibsem.applications.autolamella.ui.grid_workflow_widget as module

        monkeypatch.setattr(module, "_review_available", lambda: True)
        row = view._task_rows["overview_sem"]
        row.refresh()
        assert not row.btn_attention.isHidden()
        assert row.btn_attention.text() == "Automated"
        changed = []
        view.protocol_changed.connect(lambda: changed.append(True))

        row.btn_attention.click()

        config = experiment.grid_protocol.task_config["overview_sem"]
        assert config.attention is Attention.review
        assert row.btn_attention.text() == "Review"
        assert changed == [True]
        again = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert again.grid_protocol.task_config["overview_sem"].attention is (
            Attention.review
        )

        row.btn_attention.click()
        assert config.attention is Attention.automated


def test_review_on_a_task_nothing_requires_says_nothing_waits(
    qapp, arctis, experiment, monkeypatch
):
    """As on the lamella list: a Review chip on a task no other task requires
    holds nothing, and the row says so until something requires it."""
    import fibsem.applications.autolamella.ui.grid_workflow_widget as module

    monkeypatch.setattr(module, "_review_available", lambda: True)
    experiment.grid_protocol.task_config["overview_sem"].attention = Attention.review
    widget = GridWorkflowWidget()
    widget.set_microscope(arctis)
    widget.set_experiment(experiment)
    row = widget._task_rows["overview_sem"]
    assert row.detail_label.text() == "nothing waits on this"
    assert "nothing waits" in row.btn_attention.toolTip()

    experiment.grid_protocol.task_config["overview_fm"].requires = ["overview_sem"]
    widget._rebuild()
    row = widget._task_rows["overview_sem"]
    assert row.detail_label.text() != "nothing waits on this"
    assert "the tasks that require it wait" in row.btn_attention.toolTip()
    widget.close()


def test_the_review_tab_loads_a_grid_proposals_image_from_the_grid_directory(
    experiment,
):
    """A grid's recorded outputs are relative to its grid_path, which the record
    does not store; the loader resolves it through the experiment."""
    from fibsem.applications.autolamella.proposals import TASK_RESULT, Proposal
    from fibsem.applications.autolamella.ui.review_tab_widget import (
        _load_reference_image,
    )
    from fibsem.structures import FibsemImage

    grid = experiment.add_grid(GridRecord(name="grid-oak"))
    directory = experiment.grid_path(grid) / "overview_sem"
    directory.mkdir(parents=True)
    FibsemImage.generate_blank_image(resolution=(64, 64)).save(
        str(directory / "overview.tif")
    )
    proposal = Proposal(
        kind=TASK_RESULT, provenance={"reference_image": "overview_sem/overview.tif"}
    )

    image = _load_reference_image(experiment, grid, proposal)

    assert image is not None and image.data.shape == (64, 64)


def test_the_preflight_says_what_a_run_does(qapp):
    dialog = GridRunPreflightDialog(
        ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"], 2, "/exp"
    )
    assert dialog.windowTitle() == "Run grid workflow"
    dialog = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", screen_all=True
    )
    assert dialog.windowTitle() == "Screen all grids"


def test_the_preflight_says_which_beams_are_off(qapp):
    """Screening starts on an empty stage, so the beams are off more often than
    not. The run turns them on; the dialog says so first, so it is no surprise
    and an operator who does not want that can cancel."""
    dialog = GridRunPreflightDialog(["overview_sem"], ["Grid-01"], 1, "/exp")
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert not any("beam" in t for t in labels)

    dialog = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", beams_off=[BeamType.ION]
    )
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert "The ion beam is off. The run turns it on when it starts." in labels

    dialog = GridRunPreflightDialog(
        ["overview_sem"],
        ["Grid-01"],
        1,
        "/exp",
        beams_off=[BeamType.ELECTRON, BeamType.ION],
    )
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert (
        "The electron and ion beams are off. The run turns them on when it starts."
        in labels
    )


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
        assert left.tabBar().isHidden()  # one page: no bar, as the tab was before
        main_ui._preferences.features.grid_workflow = True
        main_ui._apply_grid_workflow_visibility()
        assert left.isTabVisible(index) and not left.tabBar().isHidden()
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


def test_an_inventory_on_the_sample_view_reaches_the_grids_tab_and_run_view(
    main_ui, tmp_path, monkeypatch
):
    """The Sample view's inventory updates the loader; the experiment records the
    grids it lists, and the Grids tab and the Workflow view's rows follow, with
    nothing pressed on the Grids tab. An autoloader, so the Sample view has a
    loader panel to fire from."""
    ui = main_ui.autolamella_ui
    config = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
    monkeypatch.setattr(
        ui.system_widget, "load_configuration", lambda configuration_name=None: config
    )
    ui.system_widget.connect_to_microscope()
    main_ui._refresh_grids_tab_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    ui.experiment = exp
    main_ui.grids_tab.set_experiment(exp)
    main_ui.grid_workflow_widget.set_experiment(exp)
    assert exp.grids == []
    assert main_ui.grid_workflow_widget._grid_rows == {}

    loader = ui.sample_widget.loader_widget
    assert loader is not None
    ui.microscope._stage.get_inventory()
    loader.loader_changed.emit()  # what the Sample view does after an inventory
    names = [g.name for g in exp.grids]
    assert names and names == [
        e.name for e in ui.microscope._stage.grid_inventory() if e.present
    ]
    assert [c.grid.name for c in main_ui.grids_tab.cards.cards] == names
    assert list(main_ui.grid_workflow_widget._grid_rows) == names
    assert names[0] in (tmp_path / "exp" / "experiment.yaml").read_text()


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

    # A beam that is off is turned on before the first task; the preflight
    # said it would be.
    microscope.turn_off(BeamType.ION)
    assert main_ui._beams_off() == [BeamType.ION]
    main_ui._start_grid_run(["overview_sem"], ["grid-aspen"], inventory_first=False)
    assert ui.is_workflow_running
    assert not main_ui.grids_tab.btn_inventory.isEnabled()  # locked during the run
    _wait_for_run(ui)
    QTest.qWait(200)  # let the finished signal land

    grid = exp.get_grid_by_name("grid-aspen")
    # a fixed holder: the grid was loaded already, so no load entry
    assert [t.name for t in grid.task_history] == ["overview_sem"]
    assert grid.task_history[-1].status is AutoLamellaTaskStatus.Completed
    assert len(grid_outputs(exp, grid, "overview_sem")) == 1
    assert main_ui.grids_tab.btn_inventory.isEnabled()
    assert ui._last_run_summary is not None  # the grid summary, for the agent server
    assert microscope.is_on(BeamType.ION) and main_ui._beams_off() == []


def test_adding_grids_to_a_running_queue_appends_their_blocks(
    main_ui, tmp_path, monkeypatch
):
    """The Grids view's selection goes onto the end of a grid run's queue as
    load-plus-tasks blocks, after a confirmation; a lamella run refuses it."""
    from fibsem.applications.autolamella.ui import grid_workflow_widget as view_module
    from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
        GridTaskManager,
        plan_grid_run,
    )
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager

    ui = main_ui.autolamella_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    for i, name in enumerate(["grid-aspen", "grid-birch"]):
        slot = microscope._stage.holder.slots[f"Slot-{i + 1:02d}"]
        slot.position = FibsemStagePosition(
            name=slot.name, x=-4e-3 + i * 8e-3, y=1e-3, z=4e-3, r=0, t=0.61
        )
        slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "t")
        slot.loaded_grid = SampleGrid(name=name)
    main_ui._refresh_grids_tab_microscope()
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    exp.grid_protocol.add(
        BeamOverviewGridTaskConfig(task_name="overview_sem", settings=_small_settings())
    )
    exp.sync_grids_from_inventory(microscope._stage)
    ui.experiment = exp
    main_ui.grid_workflow_widget.set_experiment(exp)
    main_ui.workflow_left_tabs.setCurrentWidget(main_ui.grid_workflow_widget)
    monkeypatch.setattr(
        view_module.GridRunPreflightDialog, "exec_", lambda self: QDialog.Accepted
    )

    # a grid run under way on grid-aspen, with the manager where the window looks
    manager = GridTaskManager(microscope, exp, parent_ui=ui)
    manager.queue.build_from_pairs(plan_grid_run(["overview_sem"], ["grid-aspen"]))
    manager.queue.next()  # grid-aspen's load is active
    ui._task_manager = manager

    view = main_ui.grid_workflow_widget
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui._on_workflow_selection_changed()  # the Add button follows the selection
    main_ui._on_add_to_queue(run_next=True)  # "run next" is still the end for grids
    assert [(i.item_name, i.task_name) for i in manager.queue.items] == [
        ("grid-aspen", LOAD_ENTRY_NAME),
        ("grid-aspen", "overview_sem"),
        ("grid-birch", LOAD_ENTRY_NAME),
        ("grid-birch", "overview_sem"),
    ]
    assert not view._grid_rows["grid-birch"].checkbox.isChecked()  # cleared once added

    # adding the same again: nothing new to queue
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui._on_add_to_queue(run_next=False)
    assert len(manager.queue.items) == 4

    # a lamella run refuses grid tasks, and the Add button says so instead of
    # offering an add the handler would refuse (seen on the bench: it was
    # enabled with a grid ticked during a lamella run)
    ui._task_manager = TaskManager(microscope, exp, parent_ui=ui)
    main_ui._on_workflow_selection_changed()
    # The button's own flag: the timeline as a whole is only enabled once an
    # experiment is loaded through the window, which this harness skips.
    add = main_ui.workflow_timeline._btn_add
    own = lambda: add.isEnabledTo(add.parentWidget())  # noqa: E731
    assert not own() and "lamella run is going" in add.toolTip()
    main_ui._on_add_to_queue(run_next=False)
    assert len(ui._task_manager.queue.items) == 0
    ui._task_manager = manager  # back to the grid run: a ticked grid can join
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui._on_workflow_selection_changed()
    assert own() and add.toolTip().startswith("Add to the end of the queue")
    ui._task_manager = None

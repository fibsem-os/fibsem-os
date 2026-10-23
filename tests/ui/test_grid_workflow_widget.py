"""Workflow · Grids: select grids and tasks, confirm, run on the window's worker."""

import os
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QDialog, QLabel

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.grids.run import GridWorkflowWidget
from fibsem.applications.autolamella.ui.workflow.preflight import (
    GridRunPreflightDialog,
)
from fibsem.applications.autolamella.workflows.estimate import (
    exchange_seconds,
    exchanges_for,
)
from fibsem.services.sample_handling import bind_sample_handling
from fibsem.structures import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    OverviewAcquisitionSettings,
    SampleGrid,
    SlotCalibration,
)
from fibsem.workflows.attention import Attention
from fibsem.workflows.grid import (
    BeamOverviewGridTaskConfig,
    FluorescenceOverviewGridTaskConfig,
)
from fibsem.workflows.grid.manager import (
    LOAD_ENTRY_NAME,
)
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.outputs import item_outputs
from fibsem.workflows.status import TaskState, TaskStatus


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
    experiment.sync_grids_from_inventory(arctis.sample_handling)
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
        assert view.grid_header.count_label.text() == "4 · 3 present"
        assert not view._grid_rows["grid-oak"].checkbox.isEnabled()
        # every task ticked by default, in the protocol's order
        assert view.get_selected_task_names() == ["overview_sem", "overview_fm"]
        assert view.get_selected_grids() == []
        assert view.selection_summary() == "0 grids, 2 tasks"

    def test_select_all_ticks_only_present_grids(self, view):
        view.grid_header.checkbox_all.setChecked(True)
        assert [g.name for g in view.get_selected_grids()] == [
            "Grid-01",
            "Grid-02",
            "Grid-03",
        ]
        assert view.selection_summary() == "3 grids, 2 tasks · 3 exchanges"

    def test_clearing_from_the_window_unticks_the_header_too(self, view):
        view.grid_header.checkbox_all.setChecked(True)
        view.set_all_grids_selected(False)  # as a run starting does
        assert view.get_selected_grids() == []
        assert not view.grid_header.checkbox_all.isChecked()
        view.grid_header.checkbox_all.setChecked(True)  # one click selects all again
        assert len(view.get_selected_grids()) == 3

    def test_a_grid_in_the_beam_costs_no_exchange(self, view, arctis):
        arctis.sample_handling.ensure_loaded("Grid-02")
        view.refresh()
        view.grid_header.checkbox_all.setChecked(True)
        names = [g.name for g in view.get_selected_grids()]
        assert exchanges_for(arctis.sample_handling, names) == 2
        row = view._grid_rows["Grid-02"]
        assert [c.text() for c in row._chip_widgets] == ["Loaded"]
        assert row.slot_label.text() == "slot 02"

    def test_a_loaded_row_keeps_its_slot_inside_the_list(self, view, arctis):
        """The Loaded pill once pushed the slot column past the list's edge, and
        the slot read as "s". The row must ask for less than the panel gives."""
        arctis.sample_handling.ensure_loaded("Grid-02")
        view.refresh()
        row = view._grid_rows["Grid-02"]
        assert [c.text() for c in row._chip_widgets] == ["Loaded"]
        assert row.sizeHint().width() <= 380

    def test_an_unscanned_magazine_says_so_on_every_row(self, view, arctis):
        """Before an inventory the rows are UNKNOWN: the chip says the magazine
        has not been read, not that the grids are missing, and none can be run."""
        arctis.sample_handling.magazine.scanned = False
        view.refresh()
        rows = [view._grid_rows[n] for n in ("Grid-01", "Grid-02", "Grid-03")]
        assert all([c.text() for c in r._chip_widgets] == ["not scanned"] for r in rows)
        assert not any(r.checkbox.isEnabled() for r in rows)
        arctis.sample_handling.magazine.scanned = True
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
        assert view.task_header.checkbox_all.isChecked()  # reads the rows
        view.task_header.checkbox_all.setChecked(False)
        assert view.get_selected_task_names() == []
        assert not view.btn_screen_all.isEnabled()
        view.set_all_tasks_selected(True)
        view.set_controls_enabled(False)
        assert not view.btn_screen_all.isEnabled()


class TestAttentionChip:
    """Automated or Review per grid task, as on the lamella task list."""

    def test_a_click_toggles_review_and_saves_the_protocol(self, view, experiment):
        row = view._task_rows["overview_sem"]
        row.refresh()
        assert not row.btn_attention.isHidden()
        assert row.btn_attention.text() == "Automated"
        changed = []
        view.protocol_changed.connect(lambda: changed.append(True))

        row.btn_attention.click()

        config = experiment.grid_protocol.workflow_config.get("overview_sem")
        assert config.attention is Attention.supervised
        assert row.btn_attention.text() == "Supervised"
        assert changed == [True]
        again = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert again.grid_protocol.workflow_config.get("overview_sem").attention is (
            Attention.supervised
        )

        row.btn_attention.click()
        assert config.attention is Attention.automated


class TestRequiresOnTheTaskList:
    """What a grid task requires is edited on its Workflow row, as a lamella
    task's is: a pencil, an Edit Task dialog, and "after ..." on the row."""

    def test_the_dialog_lists_every_other_task(self, view, experiment):
        dialog = view.edit_dialog("overview_fm")
        assert list(dialog.checks) == ["overview_sem"]
        assert dialog.requires == []

    def test_apply_saves_and_the_row_says_after(self, view, experiment, monkeypatch):
        import fibsem.applications.autolamella.ui.grids.run as module

        def accept_with_fm(dialog):
            dialog.checks["overview_fm"].setChecked(True)
            return module.QDialog.Accepted

        monkeypatch.setattr(module.GridTaskEditDialog, "exec_", accept_with_fm)
        changed = []
        view.protocol_changed.connect(lambda: changed.append(True))
        row = view._task_rows["overview_sem"]
        assert row.detail_label.text() == "Beam overview"

        row.btn_edit.click()

        assert experiment.grid_protocol.requirements("overview_sem") == ["overview_fm"]
        again = Experiment.load(Path(experiment.path) / "experiment.yaml")
        assert again.grid_protocol.requirements("overview_sem") == ["overview_fm"]
        assert changed == [True]
        assert row.detail_label.text() == "after overview_fm"
        assert "Beam overview" in row.detail_label.toolTip(), "the kind is a hover away"

    def test_cancel_changes_nothing(self, view, experiment, monkeypatch):
        import fibsem.applications.autolamella.ui.grids.run as module

        def tick_then_cancel(dialog):
            dialog.checks["overview_fm"].setChecked(True)
            return module.QDialog.Rejected

        monkeypatch.setattr(module.GridTaskEditDialog, "exec_", tick_then_cancel)
        view._task_rows["overview_sem"].btn_edit.click()
        assert experiment.grid_protocol.requirements("overview_sem") == []
        assert view._task_rows["overview_sem"].detail_label.text() == "Beam overview"

    def test_a_supervised_requirement_reads_as_any_other(self, view, experiment):
        view.set_requires("overview_sem", ["overview_fm"])
        row = view._task_rows["overview_sem"]
        assert row.detail_label.text() == "after overview_fm"

        view._task_rows["overview_fm"].btn_attention.click()  # FM to Supervised

        assert row.detail_label.text() == "after overview_fm"


def test_a_grid_proposals_image_loads_from_the_grid_directory(experiment):
    """A grid's recorded outputs are relative to its grid_path, which the record
    does not store; the loader resolves it through the experiment."""
    from fibsem.structures import FibsemImage
    from fibsem.workflows.proposals import STATE, Proposal
    from fibsem.workflows.reference_images import load_reference_image

    grid = experiment.add_grid(GridRecord(name="grid-oak"))
    directory = experiment.grid_path(grid) / "overview_sem"
    directory.mkdir(parents=True)
    FibsemImage.generate_blank_image(resolution=(64, 64)).save(
        str(directory / "overview.tif")
    )
    proposal = Proposal(
        kind=STATE, provenance={"reference_image": "overview_sem/overview.tif"}
    )

    image = load_reference_image(experiment, grid, proposal)

    assert image is not None and image.data.shape == (64, 64)


def test_the_preflight_says_what_a_run_does(qapp):
    dialog = GridRunPreflightDialog(
        ["overview_sem", "overview_fib"], ["Grid-01", "Grid-02"], 2, "/exp"
    )
    assert dialog.windowTitle() == "Run grid workflow"


def test_the_preflight_names_the_grids_whose_names_it_will_fix(qapp):
    """A grid's name is fixed once it has run, so the run that does it says so,
    marking a default name; nothing is said when every grid has run before."""
    dialog = GridRunPreflightDialog(["overview_sem"], ["Grid-01"], 1, "/exp")
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert not any("Names are fixed" in t for t in labels)

    dialog = GridRunPreflightDialog(
        ["overview_sem"],
        ["Grid-01", "grid-aspen"],
        2,
        "/exp",
        first_run=["Grid-01", "grid-aspen"],
    )
    labels = [w.text() for w in dialog.findChildren(QLabel)]
    assert (
        "Names are fixed once a grid has run. Running for the first time: "
        "Grid-01 (default name), grid-aspen. Rename on the Grids tab first if "
        "needed."
    ) in labels


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


def test_the_preflight_says_how_the_run_ends(qapp):
    """Set on the ⚙, said on the confirmation; not when adding to a running
    queue: the added work ends the way that run was started."""
    both = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", unload_at_end=True, report_at_end=True
    )
    assert (
        "When the run ends: unload the last grid and write the grid screening report."
        in _labels(both)
    )
    neither = GridRunPreflightDialog(["overview_sem"], ["Grid-01"], 1, "/exp")
    assert any(
        t.startswith("When the run ends: nothing more") for t in _labels(neither)
    )
    adding = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", adding=True, unload_at_end=True
    )
    assert not any(t.startswith("When the run ends") for t in _labels(adding))


def _wait_for_run(ui, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while ui.session.runner.is_running and time.monotonic() < deadline:
        QTest.qWait(100)
    assert not ui.session.runner.is_running, "the grid run did not finish in time"


class TestPanelChrome:
    """The Lamella page's list chrome: one header style, Screen all on the grid
    list's, settings and hints on the task list's, and no footer."""

    def test_headers_count_and_carry_screen_all(self, view):
        assert view.grid_header.count_label.text() == "4 \u00b7 3 present"
        assert view.task_header.count_label.text() == "2"
        assert view.btn_screen_all.parent() is view.grid_header
        for gone in ("summary_label", "task_hint"):
            assert not hasattr(view, gone)

    def test_the_task_header_reads_partly_ticked(self, view):
        view._task_rows["overview_fm"].checkbox.setChecked(False)
        box = view.task_header.checkbox_all
        assert box.checkState() == Qt.CheckState.PartiallyChecked
        box.click()  # partly ticked: one click ticks them all
        assert view.get_selected_task_names() == ["overview_sem", "overview_fm"]
        assert box.checkState() == Qt.CheckState.Checked

    def test_the_gear_opens_the_grid_workflow_options(self, view):
        # on the Lamella page ⚙ opens the workflow's settings; here, how a run
        # ends. Unload off and the report on until asked otherwise.
        assert view.btn_settings.isEnabled()
        assert view.btn_settings.toolTip() == "Grid workflow options"
        view.btn_settings.click()
        assert view.options.isVisible()
        assert not view.unload_at_end()
        assert view.report_at_end()
        view.options.hide()
        hints = view.btn_help.toolTip()
        assert "order they run" in hints and "Protocol tab's Grid page" in hints

    def test_unload_is_offered_only_with_a_loader(self, view):
        options = view.options
        options.set_availability(can_unload=False)
        options.unload_check.setChecked(True)
        assert options.unload_check.isHidden()
        assert not view.unload_at_end(), "a fixed holder has nothing to unload"
        options.set_availability(can_unload=True)
        assert not options.unload_check.isHidden()
        assert view.unload_at_end(), "the choice is kept"

    def test_the_gear_is_accent_while_an_option_is_off_its_default(self, view):
        view.options.set_availability(can_unload=True)
        assert "some set" not in view.btn_settings.toolTip()
        view.options.unload_check.setChecked(True)
        assert "some set" in view.btn_settings.toolTip()
        view.options.unload_check.setChecked(False)
        view.options.report_check.setChecked(False)
        assert "some set" in view.btn_settings.toolTip()


def test_the_grids_view_sits_beside_lamella(main_ui):
    left = main_ui.workflow_tab.left_tabs
    index = left.indexOf(main_ui.workflow_tab.grid_workflow_widget)
    assert left.tabText(index) == "Grids" and left.tabText(0) == "Lamella"
    assert left.isTabVisible(index) and not left.tabBar().isHidden()
    # each list is named by its own header, with no section label over it
    lamella = main_ui.workflow_tab.lamella_workflow_widget
    assert not hasattr(lamella, "_lamella_header")
    assert lamella.lamella_list._header.title_label.text() == "Lamella"


def test_an_inventory_on_the_grids_tab_reaches_the_run_view(main_ui, tmp_path):
    """The Grids tab creates the records; the Workflow view's rows must follow
    without a reload."""
    ui = main_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_device.compustage = False
    microscope.sample_handling = bind_sample_handling(microscope)
    slot = microscope.sample_handling.holder.slots["Slot-01"]
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
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    # the app enables the tab on experiment load; a click on a disabled tab's
    # button is swallowed
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )
    assert main_ui.workflow_tab.grid_workflow_widget._grid_rows == {}

    main_ui.grids_tab._synchronous = True
    assert main_ui.grids_tab.btn_inventory.isEnabled()
    main_ui.grids_tab.btn_inventory.click()
    assert [g.name for g in exp.grids] == ["grid-aspen"]
    assert list(main_ui.workflow_tab.grid_workflow_widget._grid_rows) == ["grid-aspen"]
    assert main_ui.workflow_tab.grid_workflow_widget.grid_empty.isHidden()


def test_an_inventory_on_the_sample_view_reaches_the_grids_tab_and_run_view(
    main_ui, tmp_path, monkeypatch
):
    """The Sample view's inventory updates the loader; the experiment records the
    grids it lists, and the Grids tab and the Workflow view's rows follow, with
    nothing pressed on the Grids tab. An autoloader, so the Sample view has a
    loader panel to fire from."""
    ui = main_ui
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
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    assert exp.grids == []
    assert main_ui.workflow_tab.grid_workflow_widget._grid_rows == {}

    loader = ui.panel.sample_widget.loader_widget
    assert loader is not None
    ui.microscope.sample_handling.get_inventory()
    loader.loader_changed.emit()  # what the Sample view does after an inventory
    names = [g.name for g in exp.grids]
    assert names and names == [
        e.name for e in ui.microscope.sample_handling.grid_inventory() if e.present
    ]
    assert [c.grid.name for c in main_ui.grids_tab.cards.cards] == names
    assert list(main_ui.workflow_tab.grid_workflow_widget._grid_rows) == names
    assert names[0] in (tmp_path / "exp" / "experiment.yaml").read_text()


def test_a_grid_run_from_the_window_on_a_fixed_holder(main_ui, tmp_path, monkeypatch):
    """End to end: the Run button on the Grids view, the worker, the manager, the
    shared timeline and the record. A fixed holder, so no exchange."""
    ui = main_ui
    # The post-run summary is modal; under offscreen it would block forever.
    from fibsem.applications.autolamella.ui import window as ui_module
    from fibsem.ui.run import view as run_view_module

    class _NoDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr(run_view_module, "WorkflowSummaryDialog", _NoDialog)
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_device.compustage = False
    microscope.sample_handling = bind_sample_handling(microscope)
    slot = microscope.sample_handling.holder.slots["Slot-01"]
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
    exp.sync_grids_from_inventory(microscope.sample_handling)
    ui.experiment = exp
    # The pieces of _on_experiment_update this test needs. The whole handler also
    # rebuilds the lamella task editor, which cannot take a protocol with no
    # lamella tasks (a pre-existing gap; real protocols always have some).
    main_ui.grids_tab.set_experiment(exp)
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    main_ui.tab_widget.setTabEnabled(
        main_ui.tab_widget.indexOf(main_ui.grids_tab), True
    )

    view = main_ui.workflow_tab.grid_workflow_widget
    main_ui.workflow_tab.left_tabs.setCurrentWidget(view)
    view.grid_header.checkbox_all.setChecked(True)
    assert main_ui.workflow_controls.run_btn.isEnabled()
    assert "1 grid, 1 task" in main_ui.workflow_controls.run_btn.toolTip()

    # A beam that is off is turned on before the first task; the preflight
    # said it would be.
    microscope.turn_off(BeamType.ION)
    assert main_ui.workflow_tab._beams_off() == [BeamType.ION]
    main_ui.workflow_tab._start_grid_run(["overview_sem"], ["grid-aspen"])
    assert ui.session.runner.is_running
    assert not main_ui.grids_tab.btn_inventory.isEnabled()  # locked during the run
    _wait_for_run(ui)
    QTest.qWait(200)  # let the finished signal land

    # The start cleared the grid ticks, header too, so an Add to Queue mid-run
    # could not queue grid-aspen again; the task ticks stay for the next add.
    # Checked after the run: a failure mid-run would leave its worker going.
    assert view.get_selected_grids() == []
    assert not view.grid_header.checkbox_all.isChecked()
    assert view.get_selected_task_names() == ["overview_sem"]

    grid = exp.get_grid_by_name("grid-aspen")
    # a fixed holder: the grid was loaded already, so no load entry
    assert [t.name for t in grid.task_history] == ["overview_sem"]
    assert grid.task_history[-1].status is TaskStatus.Completed
    assert len(item_outputs(exp, grid, "overview_sem")) == 1
    assert main_ui.grids_tab.btn_inventory.isEnabled()
    assert (
        ui.session.runner.last_summary is not None
    )  # the grid summary, for the App Server
    assert microscope.is_on(BeamType.ION) and main_ui.workflow_tab._beams_off() == []


def test_adding_grids_to_a_running_queue_appends_their_blocks(
    main_ui, tmp_path, monkeypatch
):
    """The Grids view's selection goes onto the end of a grid run's queue as
    load-plus-tasks blocks, after a confirmation; a lamella run refuses it."""
    from fibsem.applications.autolamella.workflows.tasks.manager import TaskManager
    from fibsem.workflows.grid.manager import (
        GridTaskManager,
        plan_grid_run,
    )

    ui = main_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_device.compustage = False
    microscope.sample_handling = bind_sample_handling(microscope)
    for i, name in enumerate(["grid-aspen", "grid-birch"]):
        slot = microscope.sample_handling.holder.slots[f"Slot-{i + 1:02d}"]
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
    exp.sync_grids_from_inventory(microscope.sample_handling)
    ui.experiment = exp
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    main_ui.workflow_tab.left_tabs.setCurrentWidget(
        main_ui.workflow_tab.grid_workflow_widget
    )
    monkeypatch.setattr(GridRunPreflightDialog, "exec_", lambda self: QDialog.Accepted)

    # a grid run under way on grid-aspen, with the manager where the window looks
    manager = GridTaskManager(microscope, exp)
    manager.queue.build_from_pairs(plan_grid_run(["overview_sem"], ["grid-aspen"]))
    manager.queue.next()  # grid-aspen's load is active
    ui.session.runner.manager = manager

    view = main_ui.workflow_tab.grid_workflow_widget
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui.workflow_tab.selection_changed()  # the Add button follows the selection
    main_ui.workflow_tab._on_add_to_queue(
        run_next=True
    )  # "run next" is still the end for grids
    assert [(i.item_name, i.task_name) for i in manager.queue.items] == [
        ("grid-aspen", LOAD_ENTRY_NAME),
        ("grid-aspen", "overview_sem"),
        ("grid-birch", LOAD_ENTRY_NAME),
        ("grid-birch", "overview_sem"),
    ]
    assert not view._grid_rows["grid-birch"].checkbox.isChecked()  # cleared once added

    # adding the same again: nothing new to queue
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui.workflow_tab._on_add_to_queue(run_next=False)
    assert len(manager.queue.items) == 4

    # a lamella run refuses grid tasks, and the Add button says so instead of
    # offering an add the handler would refuse (seen on the bench: it was
    # enabled with a grid ticked during a lamella run)
    ui.session.runner.manager = TaskManager(microscope, exp)
    main_ui.workflow_tab.selection_changed()
    # The button's own flag: the timeline as a whole is only enabled once an
    # experiment is loaded through the window, which this harness skips.
    add = main_ui.workflow_tab.timeline._btn_add
    own = lambda: add.isEnabledTo(add.parentWidget())  # noqa: E731
    assert not own() and "lamella run is going" in add.toolTip()
    main_ui.workflow_tab._on_add_to_queue(run_next=False)
    assert len(ui.session.runner.manager.queue.items) == 0
    ui.session.runner.manager = manager  # back to the grid run: a ticked grid can join
    view._grid_rows["grid-birch"].checkbox.setChecked(True)
    main_ui.workflow_tab.selection_changed()
    assert own() and add.toolTip().startswith("Add to the end of the queue")
    ui.session.runner.manager = None


def test_run_and_screen_all_name_the_grids_running_for_the_first_time(
    main_ui, tmp_path, monkeypatch
):
    """The window hands Run's confirmation the ticked grids that have never run;
    Screen all grids runs on the grids its own dialog names."""
    from fibsem.applications.autolamella.ui.workflow import tab as workflow_tab

    ui = main_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_device.compustage = False
    microscope.sample_handling = bind_sample_handling(microscope)
    for i, name in enumerate(["grid-aspen", "grid-birch"]):
        slot = microscope.sample_handling.holder.slots[f"Slot-{i + 1:02d}"]
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
    exp.sync_grids_from_inventory(microscope.sample_handling)
    # grid-aspen has run once; grid-birch has not
    ran = exp.get_grid_by_name("grid-aspen")
    ran.task_history.append(TaskState(name="overview_sem"))
    ui.experiment = exp
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    main_ui.workflow_tab.left_tabs.setCurrentWidget(
        main_ui.workflow_tab.grid_workflow_widget
    )

    shown = []
    offered = []

    class _Capture:
        def __init__(self, *args, **kwargs):
            shown.append(kwargs.get("first_run"))
            offered.append(kwargs.get("unload_at_end"))

        def exec_(self):
            return QDialog.Rejected  # look, do not run

    monkeypatch.setattr(workflow_tab, "GridRunPreflightDialog", _Capture)
    view = main_ui.workflow_tab.grid_workflow_widget
    view.grid_header.checkbox_all.setChecked(True)
    main_ui.workflow_tab._run_grid_workflow()
    assert shown == [["grid-birch"]]
    assert offered == [False], "a fixed holder has nothing to unload"
    assert not ui.session.runner.is_running

    # Screen all grids asks its own dialog, which names and records the grids,
    # and runs on exactly the grids it hands back (FIB-1138).
    started = []
    told = []

    class _Screen:
        def __init__(self, stage, experiment, task_names, *args, **kwargs):
            self.grid_names = ["grid-birch"]
            told.append((kwargs["unload_at_end"], kwargs["report_at_end"]))

        def exec_(self):
            return QDialog.Accepted

    monkeypatch.setattr(workflow_tab, "ScreenGridsDialog", _Screen)
    monkeypatch.setattr(
        main_ui.workflow_tab,
        "_start_grid_run",
        lambda *args, **kwargs: started.append((args, kwargs)),
    )
    main_ui.workflow_tab._on_screen_all_grids()
    # the run ends as the Grids page's ⚙ says, and the dialog says so (FIB-1133)
    ending = main_ui.workflow_tab._end_of_run()
    assert told == [(ending["unload_at_end"], ending["report_at_end"])]
    assert started == [((["overview_sem"], ["grid-birch"]), ending)]


def _labels(dialog) -> list:
    return [w.text() for w in dialog.findChildren(QLabel)]


def test_the_preflight_quotes_the_time_per_task_and_the_finish(
    qapp, view, experiment, arctis
):
    """FIB-1134: the run confirmation shows a time per task, the exchanges as a
    row of their own priced at the loader's figure, a total and a finish."""
    from datetime import datetime

    from fibsem.util.durations import format_duration_rounded
    from fibsem.workflows.grid.estimate import (
        estimate_grid_run,
    )

    view.grid_header.checkbox_all.setChecked(True)
    grids = view.get_selected_grids()
    names = [g.name for g in grids]
    stage = arctis.sample_handling
    exchanges = exchanges_for(stage, names)
    assert exchange_seconds(stage) == stage.magazine.exchange_seconds > 0
    estimate = estimate_grid_run(
        experiment,
        ["overview_sem", "overview_fm"],
        names,
        exchanges,
        exchange_seconds(stage),
        now=datetime(2026, 10, 5, 14, 0),
    )
    dialog = GridRunPreflightDialog(
        ["overview_sem", "overview_fm"], names, exchanges, "/exp", estimate=estimate
    )
    labels = _labels(dialog)
    assert "Estimated duration" in labels and "Expected finish" in labels
    assert format_duration_rounded(estimate.work_seconds, pad=True) in labels
    rows = [t for t in labels if "×" in t]
    assert [r.split("<")[0].strip() for r in rows] == [
        LOAD_ENTRY_NAME,
        "overview_sem",
        "overview_fm",
    ]
    assert all("×3" in r for r in rows)
    assert "Estimates round up." in labels

    # no estimate, no time: what a caller that could not price gets
    dialog = GridRunPreflightDialog(["overview_sem"], names, 3, "/exp")
    assert "Estimated duration" not in _labels(dialog)


def test_the_preflight_says_what_adding_costs_and_moves_the_finish(qapp):
    from datetime import datetime, timedelta

    from fibsem.workflows.estimate import (
        AdditionEstimate,
    )

    before = datetime(2026, 10, 5, 14, 0)
    addition = AdditionEstimate(
        work_seconds=600,
        delay_seconds=600,
        finish_before=before,
        finish_after=before + timedelta(minutes=10),
        priced_count=2,
        total_count=2,
    )
    dialog = GridRunPreflightDialog(
        ["overview_sem"], ["Grid-01"], 1, "/exp", adding=True, addition=addition
    )
    labels = _labels(dialog)
    assert "Adds" in labels and "Expected finish" in labels
    assert any(t.startswith("was ") for t in labels)


def test_the_window_prices_a_grid_run_for_the_confirmation_and_the_timeline(
    main_ui, tmp_path, monkeypatch
):
    """On a fixed holder: no exchange is priced, each task is, and the timeline
    gets a figure for every step of the plan."""
    from fibsem.applications.autolamella.ui.workflow import tab as workflow_tab

    ui = main_ui
    ui.system_widget.connect_to_microscope()
    microscope = ui.microscope
    microscope.stage_device.compustage = False
    microscope.sample_handling = bind_sample_handling(microscope)
    for i, name in enumerate(["grid-aspen", "grid-birch"]):
        slot = microscope.sample_handling.holder.slots[f"Slot-{i + 1:02d}"]
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
    exp.sync_grids_from_inventory(microscope.sample_handling)
    ui.experiment = exp
    main_ui.workflow_tab.grid_workflow_widget.set_experiment(exp)
    main_ui.workflow_tab.left_tabs.setCurrentWidget(
        main_ui.workflow_tab.grid_workflow_widget
    )

    shown = []

    class _Capture:
        def __init__(self, *args, **kwargs):
            shown.append(kwargs.get("estimate"))

        def exec_(self):
            return QDialog.Rejected  # look, do not run

    monkeypatch.setattr(workflow_tab, "GridRunPreflightDialog", _Capture)
    main_ui.workflow_tab.grid_workflow_widget.grid_header.checkbox_all.setChecked(True)
    main_ui.workflow_tab._run_grid_workflow()
    (estimate,) = shown
    assert [(t.name, t.count) for t in estimate.tasks] == [("overview_sem", 2)]
    per_grid = exp.grid_protocol.task_config["overview_sem"].estimated_duration
    assert estimate.work_seconds == pytest.approx(2 * per_grid)

    pairs = [
        (g, s)
        for g in ["grid-aspen", "grid-birch"]
        for s in [LOAD_ENTRY_NAME, "overview_sem"]
    ]
    main_ui.workflow_tab._push_timeline_estimates(pairs)
    estimates = main_ui.workflow_tab.timeline._estimates
    assert estimates[("grid-aspen", LOAD_ENTRY_NAME)] == 0.0  # a fixed holder
    assert estimates[("grid-birch", "overview_sem")] == pytest.approx(per_grid)

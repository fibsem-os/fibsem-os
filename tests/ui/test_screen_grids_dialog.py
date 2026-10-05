"""Screen all grids' dialog: scan the magazine, name the grids, then run (FIB-1138).

On the Arctis simulator started unscanned it opens at "not scanned"; names typed
during the scan are held and applied only at Screen; a slot whose description is
set keeps it and its row says what it replaced; Cancel writes nothing. Already
scanned, and on a fixed holder, it opens at the review.
"""

import os
import time

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    Experiment,
    GridRecord,
)
from fibsem.applications.autolamella.ui.screen_grids_dialog import (
    NOT_SCANNED,
    REVIEW,
    SCANNING,
    ScreenGridsDialog,
)
from fibsem.microscopes._stage import (
    DemoSampleLoader,
    GridExchangeError,
    SampleGrid,
    SlotCalibration,
    _create_sample_stage,
)
from fibsem.structures import FibsemStagePosition


@pytest.fixture
def arctis():
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml"),
    )
    return microscope


@pytest.fixture
def unscanned(arctis):
    """The magazine as an Arctis reads it after it has been opened: slots 1, 2
    and 5 hold grids, slot 2's description is set in xT, and nothing is known
    until a scan."""
    arctis._stage.loader = DemoSampleLoader(
        arctis,
        capacity=12,
        occupied=(1, 2, 5),
        names={2: "grid-elm"},
        start_unscanned=True,
    )
    return arctis


@pytest.fixture
def experiment(tmp_path):
    exp = Experiment(path=tmp_path, name="exp")
    (tmp_path / "exp").mkdir()
    exp.task_protocol = AutoLamellaTaskProtocol()
    return exp


def _dialog(microscope, experiment):
    return ScreenGridsDialog(
        microscope._stage,
        experiment,
        ["overview_sem"],
        str(experiment.path),
        synchronous=True,
    )


def _slot_names(microscope):
    return {
        e.slot_name: e.name for e in microscope._stage.grid_inventory() if e.present
    }


def test_unscanned_opens_at_not_scanned_with_every_slot_unknown(
    qapp, unscanned, experiment
):
    dialog = _dialog(unscanned, experiment)
    assert dialog.stage_name == NOT_SCANNED
    assert "has not been scanned since it was opened" in dialog.banner.text()
    assert {r.state.text() for r in dialog._rows.values()} == {"unknown"}
    assert dialog.btn_scan.text() == "Scan magazine" and dialog.btn_scan.isEnabled()
    assert dialog.btn_scan.isDefault()
    assert not dialog.btn_screen.isEnabled()


def test_names_typed_during_the_scan_are_held_until_screen(
    qapp, unscanned, experiment, monkeypatch
):
    stage = unscanned._stage
    dialog = _dialog(unscanned, experiment)
    scan = stage.run_inventory
    seen = {}

    def scanning():
        # The operator types while the scan runs: the fields are open, Screen
        # is not, and nothing reaches the slots or the experiment.
        seen["stage"] = dialog.stage_name
        seen["screen"] = dialog.btn_screen.isEnabled()
        dialog._rows["Slot-01"].field.setText("grid-birch")
        dialog._rows["Slot-02"].field.setText("grid-oak")
        dialog._rows["Slot-04"].field.setText("grid-ash")
        result = scan()
        seen["records"] = list(experiment.grids)
        return result

    monkeypatch.setattr(stage, "run_inventory", scanning)
    dialog.btn_scan.click()
    assert seen == {"stage": SCANNING, "screen": False, "records": []}
    assert dialog.stage_name == REVIEW
    assert _slot_names(unscanned) == {
        "Slot-01": "Grid-01",
        "Slot-02": "grid-elm",
        "Slot-05": "Grid-05",
    }

    rows = dialog._rows
    # Slot 2's xT name wins, and its row says what it replaced.
    assert rows["Slot-02"].field.text() == "grid-elm"
    assert rows["Slot-02"].note.text().startswith("From xT · replaced 'grid-oak'")
    assert rows["Slot-01"].field.text() == "grid-birch"
    assert rows["Slot-04"].state.text() == "empty"
    assert rows["Slot-04"].note.text() == "Empty slot: this name will not be used."
    assert rows["Slot-05"].field.placeholderText() == "Grid-05"
    assert rows["Slot-05"].note.text() == "Default name"
    assert dialog.btn_screen.text() == "Screen 3 grids"
    assert dialog.btn_scan.text() == "Scan again"

    # Taking the typed name back warns that it replaces the xT one.
    dialog._on_link("Slot-02", "typed")
    assert rows["Slot-02"].field.text() == "grid-oak"
    assert "Will replace the xT name 'grid-elm'" in rows["Slot-02"].note.text()

    dialog.btn_screen.click()
    assert dialog.result() == dialog.Accepted
    assert dialog.grid_names == ["grid-birch", "grid-oak", "Grid-05"]
    assert _slot_names(unscanned) == {
        "Slot-01": "grid-birch",
        "Slot-02": "grid-oak",
        "Slot-05": "Grid-05",
    }
    assert [g.name for g in experiment.grids] == ["grid-birch", "grid-oak", "Grid-05"]


def test_cancel_writes_nothing(qapp, unscanned, experiment):
    dialog = _dialog(unscanned, experiment)
    dialog.btn_scan.click()
    dialog._rows["Slot-01"].field.setText("grid-birch")
    dialog._apply()
    dialog.btn_cancel.click()
    assert dialog.result() == dialog.Rejected
    assert dialog.grid_names == []
    assert experiment.grids == []
    assert _slot_names(unscanned)["Slot-01"] == "Grid-01"


def test_already_scanned_opens_at_the_review(qapp, arctis, experiment):
    dialog = _dialog(arctis, experiment)
    assert dialog.stage_name == REVIEW
    assert not dialog.banner.isVisible()
    assert dialog.btn_scan.text() == "Scan again" and dialog.btn_scan.isEnabled()
    assert (
        dialog.btn_screen.text() == "Screen 3 grids" and dialog.btn_screen.isEnabled()
    )
    # The plan lists the grids the read found, not the ones known before it.
    assert experiment.grids == []
    labels = [w.text() for w in dialog.summary_box.findChildren(type(dialog.banner))]
    assert "Grid-01  ·  load" in labels
    assert any(
        t.startswith(
            "Names are fixed once a grid has run. Running for the first "
            "time: Grid-01 (default name)"
        )
        for t in labels
    )


def test_a_grid_that_has_run_keeps_its_name(qapp, arctis, experiment):
    ran = experiment.add_grid(GridRecord(name="Grid-01"))
    ran.task_history.append(AutoLamellaTaskState(name="overview_sem"))
    dialog = _dialog(arctis, experiment)
    row = dialog._rows["Slot-01"]
    assert row.field.isReadOnly()
    assert row.note.text() == "Has run: its name is fixed."


def test_a_name_two_slots_share_is_refused_before_anything_is_written(
    qapp, arctis, experiment
):
    dialog = _dialog(arctis, experiment)
    dialog._rows["Slot-01"].field.setText("grid-oak")
    dialog._rows["Slot-02"].field.setText("grid-oak")
    dialog._apply()
    assert not dialog.btn_screen.isEnabled()
    labels = [w.text() for w in dialog.summary_box.findChildren(type(dialog.banner))]
    assert "2 slots are named grid-oak." in labels
    assert _slot_names(arctis)["Slot-01"] == "Grid-01"


def test_a_name_that_does_not_stick_is_said_at_screen(
    qapp, arctis, experiment, monkeypatch
):
    loader = arctis._stage.loader

    def refuse(slot):
        raise GridExchangeError("Autoloader slot 1 reads back '', not 'grid-oak'.")

    monkeypatch.setattr(loader, "_write_slot_description", refuse)
    dialog = _dialog(arctis, experiment)
    dialog._rows["Slot-01"].field.setText("grid-oak")
    dialog._apply()
    dialog.btn_screen.click()
    assert dialog.result() != dialog.Accepted
    assert "reads back" in dialog.error_label.text()
    assert _slot_names(arctis)["Slot-01"] == "Grid-01"
    assert experiment.grids == []


def test_a_fixed_holder_opens_at_the_review_without_a_scan(qapp, experiment):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    microscope.stage_is_compustage = False
    microscope._stage = _create_sample_stage(microscope)
    for i, name in enumerate(["grid-aspen", "Grid-02"]):
        slot = microscope._stage.holder.slots[f"Slot-{i + 1:02d}"]
        slot.position = FibsemStagePosition(
            name=slot.name, x=-4e-3 + i * 8e-3, y=0, z=4e-3, r=0, t=0
        )
        slot.calibration = SlotCalibration("SEM", 35.0, 0.0, "2026-09-02T11:24:09", "t")
        slot.loaded_grid = SampleGrid(name=name)
    dialog = _dialog(microscope, experiment)
    assert dialog.stage_name == REVIEW
    assert not dialog.btn_scan.isVisible() and dialog.btn_scan.isHidden()
    dialog._rows["Slot-02"].field.setText("grid-birch")
    dialog._apply()
    dialog.btn_screen.click()
    assert dialog.grid_names == ["grid-aspen", "grid-birch"]
    assert microscope._stage.holder.slots["Slot-02"].loaded_grid.name == "grid-birch"


def test_screen_all_grids_from_the_window_runs_on_the_named_grids(
    qapp, tmp_path, monkeypatch
):
    """End to end on an unscanned magazine: Screen all grids opens the dialog,
    the scan runs on its own thread, a name typed during it is applied at
    Screen, and the run screens exactly the grids found, under those names.
    The tasks themselves are stood in for: this is about the wiring."""
    from PyQt5.QtTest import QTest

    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module
    from fibsem.applications.autolamella.ui import AutoLamellaUI as ui_module
    from fibsem.applications.autolamella.workflows.tasks.grid import (
        BeamOverviewGridTaskConfig,
    )
    from fibsem.applications.autolamella.workflows.tasks.grid.manager import (
        LOAD_ENTRY_NAME,
        GridTaskManager,
    )

    class _NoDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec_(self):
            return 0

    monkeypatch.setattr(ui_module, "WorkflowSummaryDialog", _NoDialog)
    ran = []

    def _run_single_task(self, task_name, grid):
        ran.append((grid.name, task_name))
        grid.task_state.name = task_name
        grid.task_state.status = AutoLamellaTaskStatus.Completed
        grid.task_history.append(AutoLamellaTaskState(name=task_name))
        return None

    monkeypatch.setattr(GridTaskManager, "_run_single_task", _run_single_task)

    def _wait(condition, timeout_s=30.0):
        deadline = time.monotonic() + timeout_s
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(50)
        assert condition()

    def _drive(dialog):
        """What the operator does: scan, type a name while it runs, Screen."""
        dialog.show()
        _wait(lambda: dialog.stage_name == NOT_SCANNED)  # the read, on its thread
        dialog.btn_scan.click()
        assert dialog.stage_name == SCANNING
        dialog._rows["Slot-01"].field.setText("grid-birch")
        _wait(lambda: dialog.stage_name == REVIEW)
        assert dialog._rows["Slot-02"].note.text() == "From xT"
        dialog.btn_screen.click()
        return dialog.result()

    monkeypatch.setattr(module.ScreenGridsDialog, "exec_", _drive)

    window = module.AutoLamellaSingleWindowUI()
    ui = window.autolamella_ui
    try:
        ui.system_widget.connect_to_microscope()
        microscope = ui.microscope
        microscope._stage.loader = DemoSampleLoader(
            microscope,
            capacity=12,
            occupied=(1, 2, 5),
            names={2: "grid-elm"},
            start_unscanned=True,
        )
        window._refresh_grids_tab_microscope()
        exp = Experiment(path=tmp_path, name="exp")
        (tmp_path / "exp").mkdir()
        exp.task_protocol = AutoLamellaTaskProtocol()
        exp.grid_protocol.add(BeamOverviewGridTaskConfig(task_name="overview_sem"))
        ui.experiment = exp
        window.grids_tab.set_experiment(exp)
        window.grid_workflow_widget.set_experiment(exp)
        window.workflow_left_tabs.setCurrentWidget(window.grid_workflow_widget)

        window._on_screen_all_grids()
        assert ui.is_workflow_running
        _wait(lambda: not ui.is_workflow_running, 90.0)
        QTest.qWait(200)  # let the finished signal land

        assert ran == [
            ("grid-birch", "overview_sem"),
            ("grid-elm", "overview_sem"),
            ("Grid-05", "overview_sem"),
        ]
        assert [g.name for g in exp.grids] == ["grid-birch", "grid-elm", "Grid-05"]
        for grid in exp.grids:
            assert grid.task_history[0].name == LOAD_ENTRY_NAME
    finally:
        if ui.microscope is not None:
            ui.microscope.disconnect()
        original_quit = qapp.quit
        qapp.quit = lambda: None
        try:
            window.close()
        finally:
            qapp.quit = original_quit

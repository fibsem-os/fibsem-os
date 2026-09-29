"""Edits to a lamella's plan, on the experiment's record (FIB-1034).

The real main window and editors on Demo, with the app's own ``EventRecorder``
(the operator for anything unmarked, as in the app). Each edit is driven through
the handler its widget calls, and read back from the recorder's buffer.
"""

import os
from concurrent.futures import Future
from copy import deepcopy
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from fibsem.acting import OPERATOR  # noqa: E402
from fibsem.applications.autolamella.event_recording import EventRecorder  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_lamella_protocol_editor as lamella_editor_module,
)
from fibsem.applications.autolamella.ui import (  # noqa: E402
    autolamella_task_config_editor as protocol_editor_module,
)
from fibsem.applications.autolamella.ui.edit_recording import SETTLE_MS  # noqa: E402
from fibsem.applications.autolamella.workflows.tasks.rough import (  # noqa: E402
    MillRoughTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.tasks import (  # noqa: E402
    SpotBurnFiducialTaskConfig,
)
from fibsem.structures import MicroscopeState, Point  # noqa: E402

TASK = "Rough Milling"
KEY = "mill_rough"
SPOT = "Spot Burn Fiducial"


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    win.autolamella_ui.system_widget.connect_to_microscope()
    win._set_border_state("idle")
    yield win
    if win.autolamella_ui.microscope is not None:
        win.autolamella_ui.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(window, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="plan-edits")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        exp.add_new_lamella(MicroscopeState(), EventedDict())
    for lamella in exp.positions:
        lamella.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    exp.task_protocol.task_config[TASK] = MillRoughTaskConfig(task_name=TASK)
    window.autolamella_ui.experiment = exp
    window.lamella_widget.set_experiment()
    return exp


@pytest.fixture
def edits(window, experiment, tmp_path):
    """The edit events recorded while the test runs."""
    recorder = EventRecorder(
        window.autolamella_ui.microscope,
        experiment_path=tmp_path,
        default_actor=OPERATOR,
    )

    def read():
        return [
            e for e in recorder.buffer.events_since(0)["events"] if e["kind"] == "edit"
        ]

    yield read
    window.lamella_widget.flush_pending_save()
    recorder.close()


@pytest.fixture
def editor(window, experiment):
    editor = window.lamella_widget
    assert editor._selected_lamella is experiment.positions[0]
    editor.listWidget_selected_task.set_tasks([TASK], preferred=TASK)
    assert editor.listWidget_selected_task.selected_task == TASK
    editor._current_milling_key = KEY
    return editor


def _deeper(config, depth):
    config = deepcopy(config)
    config.stages[0].pattern.depth = depth
    return config


def _depth(milling_dict):
    return milling_dict["stages"][0]["pattern"]["depth"]


# ── the lamella editor ───────────────────────────────────────────────────────


def test_a_burst_of_pattern_edits_is_one_event_from_its_start_to_its_end(
    editor, experiment, edits
):
    lamella = experiment.positions[0]
    original = lamella.task_config[TASK].milling[KEY]
    start = original.stages[0].pattern.depth
    for depth in (2e-6, 3e-6, 4e-6):  # a drag, step by step
        editor._on_milling_task_config_updated(_deeper(original, depth))
    assert edits() == [], "recorded before the edit settled"

    editor.flush_pending_save()

    ((event),) = edits()
    assert event["actor"] == "operator"
    payload = event["payload"]
    assert payload["item"] == {"id": lamella.id, "name": lamella.name}
    assert (payload["task"], payload["target"]) == (TASK, f"milling.{KEY}")
    assert payload["via"] == "lamella editor"
    assert _depth(payload["before"]) == start
    assert _depth(payload["after"]) == 4e-6


def test_an_edit_settles_on_its_own(editor, edits):
    editor._on_task_parameters_config_changed("reacquire_alignment_reference", True)
    assert edits() == []

    QTest.qWait(SETTLE_MS + 250)

    ((event),) = edits()
    payload = event["payload"]
    assert payload["target"] == "parameters.reacquire_alignment_reference"
    assert (payload["before"], payload["after"]) == (False, True)


def test_an_edit_put_back_records_nothing(editor, edits):
    editor._on_task_parameters_config_changed("reacquire_alignment_reference", True)
    editor._on_task_parameters_config_changed("reacquire_alignment_reference", False)
    editor.flush_pending_save()
    assert edits() == []


def test_leaving_the_lamella_records_its_edit(editor, experiment, edits, monkeypatch):
    monkeypatch.setattr(editor, "alignment_area_editable", True, raising=False)
    editor._on_alignment_area_updated(
        _moved_area(experiment.positions[0].alignment_area)
    )
    editor._on_selected_lamella_changed()  # a switch flushes, as it saves

    ((event),) = edits()
    assert event["payload"]["target"] == "alignment_area"


def _moved_area(area):
    area = deepcopy(area)
    area.left = min(area.left + 0.05, 1 - area.width)
    return area


def test_moving_the_point_of_interest_records_the_patterns_it_moved(
    editor, experiment, edits
):
    lamella = experiment.positions[0]
    assert lamella.task_config[TASK].sync_to_poi
    before_point = deepcopy(lamella.poi)

    editor._on_point_of_interest_updated(Point(x=2e-6, y=-1e-6))
    editor.flush_pending_save()

    by_target = {e["payload"]["target"]: e for e in edits()}
    poi = by_target["poi"]["payload"]
    assert poi["before"] == before_point.to_dict()
    assert (poi["after"]["x"], poi["after"]["y"]) == (2e-6, -1e-6)
    patterns = by_target[f"milling.{KEY}"]["payload"]
    assert patterns["via"] == "point of interest"
    assert patterns["task"] == TASK
    assert patterns["before"] != patterns["after"]


def test_applying_to_other_lamellae_records_what_each_was_and_became(
    editor, experiment, edits, monkeypatch
):
    source, other = experiment.positions
    source.task_config[TASK].milling[KEY] = _deeper(
        source.task_config[TASK].milling[KEY], 5e-6
    )
    other_before = other.task_config[TASK].to_dict()

    class _Dialog:  # the dialog, answered: this task, to the other lamella
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return lamella_editor_module.QDialog.Accepted

        def get_selected_lamella_names(self):
            return [other.name]

        def get_selected_tasks(self):
            return [TASK]

        def get_update_base_protocol(self):
            return False

    monkeypatch.setattr(lamella_editor_module, "ApplyLamellaConfigDialog", _Dialog)
    monkeypatch.setattr(
        lamella_editor_module.QMessageBox, "information", lambda *a, **k: None
    )

    editor._on_apply_to_other_clicked()
    editor.flush_pending_save()

    ((event),) = edits()  # the other lamella's; the source is unchanged
    payload = event["payload"]
    assert payload["via"] == "apply to other lamellae"
    assert payload["item"]["name"] == other.name
    assert (payload["task"], payload["target"]) == (TASK, "task_config")
    assert payload["before"] == other_before
    assert _depth(payload["after"]["milling"][KEY]) == 5e-6


def test_adding_a_spot_burn_point_records_the_points(editor, experiment, edits):
    lamella = experiment.positions[0]
    lamella.task_config[SPOT] = SpotBurnFiducialTaskConfig(task_name=SPOT)
    editor.listWidget_selected_task.set_tasks([TASK, SPOT])
    editor.listWidget_selected_task.select(SPOT)
    assert editor.listWidget_selected_task.selected_task == SPOT

    editor.spot_burn_coordinates_widget._add_coordinate()  # as its Add button does
    editor.flush_pending_save()

    ((event),) = edits()
    payload = event["payload"]
    assert payload["item"]["name"] == lamella.name
    assert (payload["task"], payload["target"], payload["via"]) == (
        SPOT,
        "parameters.coordinates",
        "lamella editor",
    )
    assert payload["before"] == []
    assert payload["after"] == [
        p.to_dict() for p in lamella.task_config[SPOT].coordinates
    ]
    assert len(payload["after"]) == 1


# ── the protocol editor ──────────────────────────────────────────────────────


@pytest.fixture
def protocol_editor(window, experiment):
    editor = window.task_widget
    editor.set_experiment(experiment)
    editor.task_list_widget.select(TASK)
    assert editor.task_list_widget.selected_task == TASK
    QTest.qWait(50)  # the stage list selects its first stage on the next tick
    return editor


def test_a_protocol_edit_and_a_sync_to_the_lamellae(
    window, experiment, edits, monkeypatch
):
    protocol_editor = window.task_widget
    protocol_editor.set_experiment(experiment)
    protocol_editor.task_list_widget.select(TASK)
    assert protocol_editor.task_list_widget.selected_task == TASK

    protocol_editor._on_task_parameters_config_changed("sync_to_poi", False)
    QTest.qWait(SETTLE_MS + 250)

    monkeypatch.setattr(
        protocol_editor_module.QMessageBox,
        "question",
        lambda *a, **k: protocol_editor_module.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        protocol_editor_module.QMessageBox, "information", lambda *a, **k: None
    )
    protocol_editor._on_sync_to_lamella_clicked()
    QTest.qWait(SETTLE_MS + 250)

    edit, *synced = (e["payload"] for e in edits())
    assert (edit["item"], edit["task"]) == (None, TASK)
    assert edit["target"] == "protocol.parameters.sync_to_poi"
    assert (edit["before"], edit["after"], edit["via"]) == (
        True,
        False,
        "protocol editor",
    )
    assert sorted(e["item"]["name"] for e in synced) == sorted(
        p.name for p in experiment.positions
    )
    for e in synced:
        assert (e["via"], e["task"], e["target"]) == (
            "sync to lamellae",
            TASK,
            "task_config",
        )
        assert e["before"]["parameters"]["sync_to_poi"] is True
        assert e["after"]["parameters"]["sync_to_poi"] is False


def test_a_protocol_pattern_edit_is_recorded_from_what_it_was(
    protocol_editor, experiment, edits
):
    stage = experiment.task_protocol.task_config[TASK].milling[KEY].stages[0]
    start = stage.pattern.depth
    deeper = deepcopy(stage.pattern)
    deeper.depth = 2 * start

    # The pattern panel's own signal, as its depth box sends it. The panel sets
    # the pattern on the stage it was given before the editor hears of it.
    stages = protocol_editor.milling_task_editor.config_widget.milling_stages_widget
    stages._pattern_widget.pattern_changed.emit(deeper)
    QTest.qWait(SETTLE_MS + 250)

    ((event),) = edits()
    payload = event["payload"]
    assert (payload["item"], payload["task"]) == (None, TASK)
    assert (payload["target"], payload["via"]) == (
        f"protocol.milling.{KEY}",
        "protocol editor",
    )
    assert _depth(payload["before"]) == start
    assert _depth(payload["after"]) == 2 * start
    assert (
        experiment.task_protocol.task_config[TASK].milling[KEY].stages[0].pattern.depth
        == 2 * start
    )


def test_a_global_edit_records_the_protocol_and_each_lamella(
    protocol_editor, experiment, edits, monkeypatch
):
    def accept(dialog):  # every task, a wider milling field of view, lamellae too
        dialog._select_all_tasks()
        dialog.spinbox_milling_fov.setValue(dialog.spinbox_milling_fov.value() + 10)
        dialog.checkbox_update_existing.setChecked(True)
        return protocol_editor_module.QDialog.Accepted

    monkeypatch.setattr(
        protocol_editor_module.AutoLamellaGlobalTaskEditDialog, "exec_", accept
    )
    monkeypatch.setattr(
        protocol_editor_module.QMessageBox, "information", lambda *a, **k: None
    )
    fov = experiment.task_protocol.task_config[TASK].milling[KEY].field_of_view

    protocol_editor._on_global_edit_clicked()
    QTest.qWait(SETTLE_MS + 250)

    protocol, *lamellae = (e["payload"] for e in edits())
    assert (protocol["item"], protocol["task"], protocol["target"]) == (
        None,
        TASK,
        "protocol.task_config",
    )
    assert sorted(e["item"]["name"] for e in lamellae) == sorted(
        p.name for p in experiment.positions
    )
    for e in (protocol, *lamellae):
        assert e["via"] == "global edit"
        assert e["before"]["milling"][KEY]["field_of_view"] == fov
        assert e["after"]["milling"][KEY]["field_of_view"] == pytest.approx(fov + 10e-6)


def test_adding_a_task_records_it_on_the_protocol_and_each_lamella(
    protocol_editor, experiment, edits, monkeypatch
):
    def accept(dialog):
        dialog.comboBox_task_type.setCurrentIndex(
            dialog.comboBox_task_type.findData("MILL_ROUGH")
        )
        dialog.lineEdit_task_name.setText("Extra Rough")
        return protocol_editor_module.QDialog.Accepted

    monkeypatch.setattr(protocol_editor_module.AddTaskDialog, "exec_", accept)

    protocol_editor._on_add_task_clicked()
    QTest.qWait(SETTLE_MS + 250)

    protocol, *lamellae = (e["payload"] for e in edits())
    assert (protocol["item"], protocol["target"]) == (None, "protocol.task_config")
    assert protocol["after"] == (
        experiment.task_protocol.task_config["Extra Rough"].to_dict()
    )
    for e in (protocol, *lamellae):
        assert (e["task"], e["via"], e["before"]) == ("Extra Rough", "add task", None)
    added = {
        p.name: p.task_config["Extra Rough"].to_dict() for p in experiment.positions
    }
    assert {e["item"]["name"]: e["after"] for e in lamellae} == added


def test_removing_a_task_records_what_the_protocol_had(
    protocol_editor, experiment, edits, monkeypatch
):
    monkeypatch.setattr(
        protocol_editor_module.QMessageBox,
        "question",
        lambda *a, **k: protocol_editor_module.QMessageBox.Yes,
    )
    had = experiment.task_protocol.task_config[TASK].to_dict()

    protocol_editor._on_remove_task_clicked()
    QTest.qWait(SETTLE_MS + 250)

    ((event),) = edits()
    payload = event["payload"]
    assert (payload["item"], payload["task"], payload["target"]) == (
        None,
        TASK,
        "protocol.task_config",
    )
    assert (payload["via"], payload["before"], payload["after"]) == (
        "remove task",
        had,
        None,
    )


def test_the_protocol_s_spot_burn_points_are_recorded(
    window, experiment, edits, monkeypatch
):
    from fibsem.ui.widgets.spot_burn_coordinates_widget import (
        SpotBurnCoordinatesWidget,
    )

    config = SpotBurnFiducialTaskConfig(task_name=SPOT)
    experiment.task_protocol.task_config[SPOT] = config
    had = config.to_dict()
    editor = window.task_widget
    editor.set_experiment(experiment)
    editor.task_list_widget.select(SPOT)
    assert editor.task_list_widget.selected_task == SPOT

    def accept(dialog):  # a point added, then OK
        dialog.findChild(SpotBurnCoordinatesWidget)._add_coordinate()
        return protocol_editor_module.QDialog.Accepted

    monkeypatch.setattr(protocol_editor_module.QDialog, "exec_", accept)

    editor._on_spot_burn_coordinates_clicked()
    QTest.qWait(SETTLE_MS + 250)

    ((event),) = edits()
    payload = event["payload"]
    assert (payload["item"], payload["task"], payload["target"]) == (
        None,
        SPOT,
        "protocol.task_config",
    )
    assert (payload["via"], payload["before"]) == ("protocol editor", had)
    assert payload["after"] == config.to_dict()
    assert len(config.coordinates) == 1


# ── a correlation ────────────────────────────────────────────────────────────

_FIB_PIXEL_SIZE = 20e-9


def _correlation_result(poi):
    """A result as the correlation dialog hands it back: seeded, with the
    refractive-index correction applied before the fit."""
    from fibsem.correlation.structures import (
        CorrelationInputData,
        CorrelationPointOfInterest,
        CorrelationResult,
    )

    return CorrelationResult(
        poi=[CorrelationPointOfInterest(px_m=poi)],
        rms_error=1.5,
        delta_2d=[Point(0.5, -0.5)] * 4,
        input_data=CorrelationInputData(stored_fib_image_pixel_size=_FIB_PIXEL_SIZE),
        refractive_index_correction_mode="pre",
        refractive_index_correction_factor=1.3,
        seed={"rotation": [0.0, 0.0, 0.0]},
        diagnostics={
            "rms_um": 0.03,
            "pairs": [],
            "mirror_ratio": 10.0,
            "depth_span_um": 5.0,
            "scale_ratio": 1.0,
            "n_pairs": 4,
            "n_accepted": 4,  # every fiducial a prediction accepted: "check"
        },
    )


def _accept_correlation(editor, monkeypatch, result):
    """Open the editor's correlation dialog and accept *result*; returns the
    run folder the dialog was given."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    given = {}

    class _Dialog:
        def __init__(self, parent=None):
            self.correlation_config = None
            self.result = result

        def set_project_dir(self, path):
            given["folder"] = path

        def set_correlation_config(self, config):
            self.correlation_config = config

        def set_fib_image(self, image):
            pass

        def set_fm_image(self, image):
            pass

        def add_lamella_setup(self, **kwargs):
            return SimpleNamespace(emit_current_seed=lambda: None)

        def set_prior_runs(self, runs):
            pass

        def exec_(self):
            return lamella_editor_module.QDialog.Accepted

    monkeypatch.setattr(ctw, "CorrelationTabDialog", _Dialog)
    editor._open_correlation_dialog()
    return given["folder"]


def test_an_accepted_correlation_is_recorded_with_the_point_it_moved(
    window, editor, experiment, tmp_path, monkeypatch
):
    recorder = EventRecorder(
        window.autolamella_ui.microscope,
        experiment_path=tmp_path,
        default_actor=OPERATOR,
    )
    lamella = experiment.positions[0]
    try:
        folder = _accept_correlation(
            editor, monkeypatch, _correlation_result(Point(x=2e-6, y=-3e-6))
        )
        editor.flush_pending_save()
        events = recorder.buffer.events_since(0)["events"]
    finally:
        recorder.close()

    assert (lamella.poi.x, lamella.poi.y) == (2e-6, -3e-6)
    kinds = [e["kind"] for e in events]
    assert kinds.count("correlation") == 1
    correlation = events[kinds.index("correlation")]
    assert correlation["actor"] == "operator"
    payload = correlation["payload"]
    assert payload["item"] == {"id": lamella.id, "name": lamella.name}
    assert (payload["poi"]["x"], payload["poi"]["y"]) == (2e-6, -3e-6)
    assert payload["rms_px"] == 1.5
    assert payload["rms_nm"] == pytest.approx(1.5 * _FIB_PIXEL_SIZE * 1e9)
    assert payload["fiducials"] == 4
    assert payload["refractive_index"] == {"mode": "pre", "factor": 1.3}
    assert (payload["verdict"], payload["seeded"]) == ("check", True)
    assert payload["folder"] == os.path.relpath(folder, str(experiment.path))
    assert payload["folder"].startswith(os.path.join(lamella.name, "Correlation"))

    # The point it moved, and the patterns that followed, say where they came from.
    edits = [e for e in events if e["kind"] == "edit"]
    assert {e["payload"]["target"] for e in edits} >= {"poi", f"milling.{KEY}"}
    assert {e["payload"]["via"] for e in edits} == {"correlation"}
    assert all(events.index(e) > events.index(correlation) for e in edits)


def test_a_correlation_that_cannot_be_recorded_is_still_applied(
    editor, experiment, edits, monkeypatch
):
    def _broken(*args, **kwargs):
        raise RuntimeError("cannot describe it")

    monkeypatch.setattr(lamella_editor_module, "correlation_record", _broken)
    lamella = experiment.positions[0]

    _accept_correlation(editor, monkeypatch, _correlation_result(Point(x=1e-6, y=1e-6)))

    assert (lamella.poi.x, lamella.poi.y) == (1e-6, 1e-6)
    editor.flush_pending_save()
    assert "poi" in {e["payload"]["target"] for e in edits()}


def test_the_replay_shows_a_correlation_before_the_point_it_moved(
    qapp, window, editor, experiment, tmp_path, monkeypatch
):
    """From the file the app writes, through the reader, to the replay window."""
    from fibsem.applications.autolamella.tools.replay import EventKind, load_replay
    from fibsem.applications.autolamella.ui.experiment_replay_widget import (
        ExperimentReplayWidget,
    )

    record = tmp_path / "record"
    record.mkdir()
    recorder = EventRecorder(
        window.autolamella_ui.microscope,
        experiment_path=record,
        default_actor=OPERATOR,
    )
    lamella = experiment.positions[0]
    try:
        _accept_correlation(
            editor, monkeypatch, _correlation_result(Point(x=2e-6, y=-3e-6))
        )
        editor.flush_pending_save()
    finally:
        recorder.close()

    kinds = (EventKind.CORRELATION, EventKind.EDIT)
    correlation, *edits = [e for e in load_replay(record).events if e.kind in kinds]
    assert (correlation.kind, correlation.item) == (EventKind.CORRELATION, lamella.name)
    assert correlation.summary == (
        "Correlation: point of interest x=2.0 µm, y=-3.0 µm — RMS 30 nm over 4"
        " fiducials, check fit, refractive index ×1.30 before the fit"
    )
    assert correlation.actor == "operator"
    assert edits
    assert {(e.kind, e.item, e.data["via"]) for e in edits} == {
        (EventKind.EDIT, lamella.name, "correlation")
    }

    widget = ExperimentReplayWidget.from_directory(record)
    try:
        assert widget.filter_boxes[EventKind.CORRELATION].text() == "Correlation (1)"
        row = widget.replay.events.index(correlation)
        cells = [widget.table.item(row, col).text() for col in (1, 2)]
        assert cells == [lamella.name, "Correlation"]
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()


# ── the agent ────────────────────────────────────────────────────────────────


def _agent_patch(window, lamella, patch):
    from fibsem.applications.autolamella.server.context import config_version

    outcome = Future()
    window.autolamella_ui._apply_agent_config_patch(
        "item",
        lamella.name,
        TASK,
        patch,
        config_version(lamella.task_config[TASK]),
        outcome,
    )
    return outcome.result(timeout=5)


def test_an_agent_patch_is_the_agent_s_edit(window, experiment, edits):
    lamella = experiment.positions[0]
    path = f"milling.{KEY}.stages.0.pattern.depth"
    assert _agent_patch(window, lamella, {path: 2.7e-6})["applied"] is True

    ((event),) = edits()
    assert event["actor"] == "agent"
    payload = event["payload"]
    assert (payload["target"], payload["via"]) == ("task_config", "agent patch")
    assert payload["item"]["name"] == lamella.name
    assert _depth(payload["after"]["milling"][KEY]) == 2.7e-6
    assert _depth(payload["before"]["milling"][KEY]) != 2.7e-6


def test_an_agent_moving_the_point_of_interest_records_the_patterns_it_moved(
    window, experiment, edits
):
    from fibsem.applications.autolamella.server.context import item_fields_version

    lamella = experiment.positions[0]
    lamella.poi = Point(x=1e-6, y=2e-6)  # floats: the patch keeps each field's type
    outcome = Future()
    window.autolamella_ui._apply_agent_config_patch(
        "item_fields",
        lamella.name,
        None,
        {"poi.x": 4e-6},
        item_fields_version(lamella),
        outcome,
    )
    result = outcome.result(timeout=5)
    assert result["applied"] is True, result

    by_target = {e["payload"]["target"]: e for e in edits()}
    assert by_target["poi"]["payload"]["after"]["x"] == 4e-6
    assert by_target[f"milling.{KEY}"]["payload"]["via"] == "agent patch"
    assert {e["actor"] for e in by_target.values()} == {"agent"}


def test_an_operator_edit_still_settling_when_the_agent_patches_is_the_operator_s(
    window, editor, experiment, edits
):
    """The patch flushes the editor first, under the agent's mark: the edit
    still says who made it."""
    editor._on_task_parameters_config_changed("reacquire_alignment_reference", True)
    path = f"milling.{KEY}.stages.0.pattern.depth"
    assert _agent_patch(window, experiment.positions[0], {path: 3.1e-6})["applied"]

    operator_edit, agent_edit = edits()
    assert operator_edit["actor"] == "operator"
    assert operator_edit["payload"]["via"] == "lamella editor"
    assert agent_edit["actor"] == "agent"


# ── the record never costs the edit ──────────────────────────────────────────


def test_an_edit_whose_record_cannot_be_taken_is_still_made(
    window, editor, experiment, edits, monkeypatch
):
    """Snapshots are taken before an edit applies: one that fails must cost the
    record, never the edit."""
    from fibsem.applications.autolamella.ui import edit_recording

    def broken(value):
        raise RuntimeError("cannot describe it")

    monkeypatch.setattr(edit_recording, "serialise", broken)
    source, other = experiment.positions

    editor._on_point_of_interest_updated(Point(x=3e-6, y=0.0))
    assert (source.poi.x, source.poi.y) == (3e-6, 0.0)

    class _Dialog:
        def __init__(self, **kwargs):
            pass

        def exec_(self):
            return lamella_editor_module.QDialog.Accepted

        def get_selected_lamella_names(self):
            return [other.name]

        def get_selected_tasks(self):
            return [TASK]

        def get_update_base_protocol(self):
            return False

    monkeypatch.setattr(lamella_editor_module, "ApplyLamellaConfigDialog", _Dialog)
    monkeypatch.setattr(
        lamella_editor_module.QMessageBox, "information", lambda *a, **k: None
    )
    source.task_config[TASK].milling[KEY] = _deeper(
        source.task_config[TASK].milling[KEY], 6e-6
    )
    editor._on_apply_to_other_clicked()
    assert other.task_config[TASK].milling[KEY].stages[0].pattern.depth == 6e-6

    path = f"milling.{KEY}.stages.0.pattern.depth"
    assert _agent_patch(window, source, {path: 1.9e-6})["applied"] is True

    editor.flush_pending_save()
    assert edits() == []  # nothing could be described, and nothing broke


# ── the replay ───────────────────────────────────────────────────────────────


def test_the_replay_shows_each_edit_on_the_lamella_edited(
    qapp, window, editor, experiment, tmp_path
):
    """From the file the app writes, through the reader, to the replay window."""
    from fibsem.applications.autolamella.tools.replay import EventKind, load_replay
    from fibsem.applications.autolamella.ui.experiment_replay_widget import (
        ExperimentReplayWidget,
    )

    record = tmp_path / "record"
    record.mkdir()
    recorder = EventRecorder(
        window.autolamella_ui.microscope,
        experiment_path=record,
        default_actor=OPERATOR,
    )
    first, second = experiment.positions
    original = first.task_config[TASK].milling[KEY]
    start = original.stages[0].pattern.depth
    other_start = second.task_config[TASK].milling[KEY].stages[0].pattern.depth
    try:
        for depth in (2e-6, 3e-6):
            editor._on_milling_task_config_updated(_deeper(original, depth))
        editor.flush_pending_save()
        path = f"milling.{KEY}.stages.0.pattern.depth"
        assert _agent_patch(window, second, {path: 2.5e-6})["applied"] is True
    finally:
        recorder.close()

    rows = [e for e in load_replay(record).events if e.kind == EventKind.EDIT]
    assert [(e.item, e.task) for e in rows] == [
        (first.name, TASK),
        (second.name, TASK),
    ]
    assert [e.summary for e in rows] == [
        f"milling.{KEY}: stages.0.pattern.depth {start:.4g} → 3e-06 (lamella editor)",
        f"task_config: milling.{KEY}.stages.0.pattern.depth {other_start:.4g}"
        " → 2.5e-06 (agent patch)",
    ]
    assert [e.actor for e in rows] == ["operator", "agent"]

    widget = ExperimentReplayWidget.from_directory(record)
    try:
        assert widget.filter_boxes[EventKind.EDIT].text() == "Edit (2)"
        row = widget.replay.events.index(rows[0])
        cells = [widget.table.item(row, col).text() for col in (1, 2)]
        assert cells == [first.name, "Edit"]
    finally:
        widget.close()
        widget.deleteLater()
        qapp.processEvents()

"""Edits to a lamella's plan, on the experiment's record (FIB-1034).

The real main window and editors on Demo, with the app's own ``EventRecorder``
(the operator for anything unmarked, as in the app). Each edit is driven through
the handler its widget calls, and read back from the recorder's buffer.
"""

import os
from concurrent.futures import Future
from copy import deepcopy

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
from fibsem.structures import MicroscopeState, Point  # noqa: E402

TASK = "Rough Milling"
KEY = "mill_rough"


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


# ── the protocol editor ──────────────────────────────────────────────────────


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
        f"milling.{KEY}: stages.0.pattern.depth {start:.4g} → 3e-06"
        " — by the operator (lamella editor)",
        f"task_config: milling.{KEY}.stages.0.pattern.depth {other_start:.4g}"
        " → 2.5e-06 — by the agent (agent patch)",
    ]

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

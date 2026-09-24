"""The record's tables, read from ``events.jsonl``, which the second version of
the report is built from (FIB-1036).

The first tests run real tasks through ``run_tasks`` on Demo and read the file
the run wrote, so a producer changing the shape of a record the tables read
fails them. What a headless Demo run cannot produce -- a question someone
waited on, a failed or cancelled run, a decision that moved something -- is in
records written here, in the shape the recorder writes.
"""

import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.event_recording import (
    EVENTS_FILENAME,
    read_events,
)
from fibsem.applications.autolamella.proposals import (
    ALIGNMENT_AREA,
    DETECTION,
    POINT_OF_INTEREST,
    STATE,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.tools.event_tables import (
    ACTOR_COLUMNS,
    DECISION_COLUMNS,
    EDIT_COLUMNS,
    MILLING_COLUMNS,
    RUN_COLUMNS,
    STEP_COLUMNS,
    WAIT_COLUMNS,
    event_tables,
    read_event_tables,
)
from fibsem.applications.autolamella.workflows.tasks import manager as manager_module
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)

SETUP = "Setup Lamella Position"
ROUGH = "Rough Milling"
CONFIG = os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=CONFIG)
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(microscope, tmp_path):
    exp = Experiment(path=tmp_path, name="tables")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=SETUP, required=True),
                AutoLamellaTaskDescription(name=ROUGH, required=True),
            ]
        )
    )
    exp.add_new_lamella(
        microscope.get_microscope_state(),
        EventedDict(
            {
                SETUP: SelectMillingPositionTaskConfig(
                    task_name=SETUP, use_autofocus=False
                ),
                ROUGH: MillRoughTaskConfig(task_name=ROUGH),
            }
        ),
    )
    exp.positions[0].path.mkdir(parents=True, exist_ok=True)
    exp.positions[0].milling_pose = microscope.get_microscope_state()
    return exp


# ── a real run ───────────────────────────────────────────────────────────────


def test_a_run_s_tasks_steps_and_milling_are_read(microscope, experiment):
    run_tasks(microscope, experiment, [SETUP, ROUGH])

    tables = read_event_tables(experiment.path)

    lamella = experiment.positions[0]
    runs = tables.runs
    assert list(runs["task"]) == [SETUP, ROUGH]
    assert set(runs["item"]) == {lamella.name}
    assert set(runs["outcome"]) == {"completed"}
    # the runs are the ones on the lamella's own record
    assert set(runs["task_id"]) == {h.task_id for h in lamella.task_history}
    assert (runs["duration"] > 0).all()
    # nobody was asked anything
    assert (runs["waiting"] == 0).all()
    assert (runs["machine"] == runs["duration"]).all()

    for run in runs.itertuples():
        steps = tables.steps[tables.steps["task_id"] == run.task_id]
        assert len(steps) > 1
        # each step ends where the next starts, and the last where the run ends
        assert list(steps["end"])[:-1] == list(steps["start"])[1:]
        assert steps["end"].iloc[-1] == run.end
        assert steps["start"].iloc[0] >= run.start
        assert set(steps["task_type"]) == {run.task_type}

    rough = lamella.task_config[ROUGH].milling
    planned = [stage for config in rough.values() for stage in config.stages]
    milling = tables.milling
    assert list(milling["stage"]) == [stage.name for stage in planned]
    assert milling["finished"].all() and (milling["duration"] > 0).all()
    assert list(milling["milling_current"]) == [
        stage.milling.milling_current for stage in planned
    ]
    assert list(milling["depth"]) == [stage.pattern.depth for stage in planned]
    assert set(milling["task_id"]) == set(runs[runs["task"] == ROUGH]["task_id"])

    assert tables.decisions.empty

    # every record counted once, and a run's steps are the task's
    actors = tables.actors
    records = list(read_events(Path(experiment.path) / EVENTS_FILENAME))
    assert actors["count"].sum() == len(records)
    steps = actors[actors["kind"] == "task_step"]
    assert list(steps["actor"]) == ["task"]
    assert steps["count"].iloc[0] == len(tables.steps)


def test_decisions_nobody_was_asked_are_read_as_the_task_s(
    microscope, experiment, monkeypatch
):
    """With review on and no window to ask in, Setup's questions go straight
    on the record as Unreviewed: nobody waited, and nothing moved."""
    monkeypatch.setattr(manager_module, "review_enabled", lambda: True)

    run_tasks(microscope, experiment, [SETUP])

    decisions = read_event_tables(experiment.path).decisions
    assert len(decisions) > 0
    assert set(decisions["outcome"]) == {"Unreviewed"}
    assert set(decisions["actor"]) == {"task"}
    assert set(decisions["item"]) == {experiment.positions[0].name}
    assert decisions["asked"].isna().all() and decisions["waited"].isna().all()
    assert not decisions["changed"].any()


def test_every_table_has_its_columns_even_empty(tmp_path):
    (tmp_path / EVENTS_FILENAME).write_text("", encoding="utf-8")

    tables = read_event_tables(tmp_path)

    assert list(tables.runs.columns) == RUN_COLUMNS
    assert list(tables.steps.columns) == STEP_COLUMNS
    assert list(tables.milling.columns) == MILLING_COLUMNS
    assert list(tables.decisions.columns) == DECISION_COLUMNS
    assert list(tables.edits.columns) == EDIT_COLUMNS
    assert list(tables.actors.columns) == ACTOR_COLUMNS
    assert list(tables.waits.columns) == WAIT_COLUMNS
    assert tables.runs.empty and tables.decisions.empty and tables.actors.empty


# ── records the recorder writes ──────────────────────────────────────────────

T0 = datetime(2026, 9, 24, 10, 0, 0)
RUN = "run-1"
TASK = "Mill Rough"
ITEM = {"id": "L1", "name": "01-lamella"}


def _record(kind, second, payload=None, run=RUN, actor="task", session="s1"):
    """One record, stamped the way the recorder stamps it: on a run of TASK
    on ITEM (or on no run), with its UTC offset."""
    return {
        "session": session,
        "actor": actor,
        "kind": kind,
        "t": (T0 + timedelta(seconds=second)).isoformat() + "-03:30",
        "item": ITEM if run else None,
        "task": {"id": run, "name": TASK} if run else None,
        "payload": payload or {},
    }


def _step(second, step, run=RUN):
    return _record("task_step", second, {"step": step, "task_type": "MILL_ROUGH"}, run)


def _started(second, run=RUN):
    return _record("task_started", second, {"task_type": "MILL_ROUGH"}, run)


def _seconds(*values):
    return [T0 + timedelta(seconds=s) for s in values]


def test_times_are_the_instrument_s_clock():
    tables = event_tables([_started(0), _record("task_completed", 5)])

    ((run),) = tables.runs.to_dict("records")
    assert (run["start"], run["end"]) == tuple(_seconds(0, 5))


def test_waiting_for_an_answer_is_not_the_machine_s_time():
    """A prompt raised in the second step, answered 30 s later. The same
    question is on the record too, asked and decided over the same span: it
    counts once."""
    records = [
        _started(0),
        _step(0, "ALIGN_REFERENCE_IMAGE"),
        _step(10, "MILL_LAMELLA"),
        _record("prompt_raised", 20, {"type": "RunMillingTask", "nonce": 3}),
        _record("proposal_asked", 20, {"proposal_id": "p1", "kind": STATE}),
        _record(
            "proposal_decided",
            50,
            {"proposal_id": "p1", "decision": 0, "kind": STATE},
            actor="operator",
        ),
        _record(
            "prompt_answered",
            50,
            {"type": "RunMillingTask", "nonce": 3, "answered_by": "operator"},
            actor="operator",
        ),
        _step(60, "ACQUIRE_REFERENCE_IMAGES"),
        _record("task_completed", 70),
    ]

    tables = event_tables(records)

    ((run),) = tables.runs.to_dict("records")
    assert (run["duration"], run["waiting"], run["machine"]) == (70.0, 30.0, 40.0)
    steps = tables.steps.to_dict("records")
    assert [s["step"] for s in steps] == [
        "ALIGN_REFERENCE_IMAGE",
        "MILL_LAMELLA",
        "ACQUIRE_REFERENCE_IMAGES",
    ]
    assert [s["duration"] for s in steps] == [10.0, 50.0, 10.0]
    assert [s["waiting"] for s in steps] == [0.0, 30.0, 0.0]
    assert [s["machine"] for s in steps] == [10.0, 20.0, 10.0]
    ((decision),) = tables.decisions.to_dict("records")
    assert decision["asked"] == T0 + timedelta(seconds=20)
    assert decision["waited"] == 30.0
    # each wait where it was, for a timeline to draw: both, as recorded
    waits = tables.waits.to_dict("records")
    assert [(w["source"], w["item"], w["task_id"]) for w in waits] == [
        ("question", ITEM["name"], RUN),
        ("prompt", ITEM["name"], RUN),
    ]
    assert {(w["start"], w["end"], w["duration"]) for w in waits} == {
        (T0 + timedelta(seconds=20), T0 + timedelta(seconds=50), 30.0)
    }


def test_a_prompt_is_paired_with_its_own_answer():
    """Two prompts in turn, the second withdrawn when the run stopped; a
    prompt from another session with the same number is not this one's."""
    records = [
        _started(0),
        _step(0, "MILL_LAMELLA"),
        _record("prompt_raised", 10, {"nonce": 1}),
        _record("prompt_raised", 11, {"nonce": 1}, session="s0"),
        _record("prompt_answered", 15, {"nonce": 1}),
        _record("prompt_raised", 20, {"nonce": 2}),
        _record("prompt_cancelled", 22, {"nonce": 2}),
        _record("task_cancelled", 22),
    ]

    ((run),) = event_tables(records).runs.to_dict("records")

    assert run["waiting"] == 7.0  # 5 s, then 2 s


def test_how_each_run_ended():
    records = [
        _started(0, "done"),
        _record("task_completed", 10, run="done"),
        _started(10, "broke"),
        _step(11, "MILL_LAMELLA", "broke"),
        _record("task_failed", 20, {"error": "milling failed"}, "broke"),
        _started(20, "stopped"),
        _record(
            "task_cancelled", 25, {"error": "Workflow aborted by user."}, "stopped"
        ),
        _record("task_skipped", 25, {"skip_reason": "defect"}, "skipped"),
        _started(30, "running"),
        _step(31, "MILL_LAMELLA", "running"),
    ]

    tables = event_tables(records)

    runs = {r["task_id"]: r for r in tables.runs.to_dict("records")}
    assert {k: r["outcome"] for k, r in runs.items()} == {
        "done": "completed",
        "broke": "failed",
        "stopped": "cancelled",
        "skipped": "skipped",
        "running": "unfinished",
    }
    assert runs["broke"]["reason"] == "milling failed"
    assert runs["stopped"]["reason"] == "Workflow aborted by user."
    assert runs["skipped"]["reason"] == "defect"
    assert pd.isna(runs["done"]["reason"])
    assert runs["skipped"]["duration"] == 0.0
    assert pd.isna(runs["running"]["end"]) and pd.isna(runs["running"]["duration"])
    steps = {s["task_id"]: s for s in tables.steps.to_dict("records")}
    assert steps["broke"]["end"] == T0 + timedelta(seconds=20)  # the run's end
    assert pd.isna(steps["running"]["duration"])


def test_a_milling_stage_that_never_finished():
    stage = {
        "name": "Rough 01",
        "milling": {"milling_current": 2e-9},
        "pattern": {"depth": 1e-6},
    }
    records = [
        _record("milling_stage_started", 0, {"task_id": "m1", "stage": stage}),
        _record(
            "milling_progress",
            40,
            {"task_id": "m1", "stage_name": "Rough 01", "status": "stage-finished"},
        ),
        _record("milling_stage_started", 40, {"task_id": "m2", "stage": stage}),
    ]

    milling = event_tables(records).milling.to_dict("records")

    assert [m["finished"] for m in milling] == [True, False]
    assert milling[0]["duration"] == 40.0 and pd.isna(milling[1]["duration"])
    assert (milling[0]["milling_current"], milling[0]["depth"]) == (2e-9, 1e-6)


def _decided(second, kind, proposed, decided, proposal_id="p1", index=0, **fields):
    payload = {
        "item": ITEM,
        "task": TASK,
        "proposal_id": proposal_id,
        "decision": index,
        "kind": kind,
        "proposed": proposed,
        "decided": decided,
        "outcome": "Confirmed",
        "author": "human:op",
        "via": "review",
        **fields,
    }
    return _record("proposal_decided", second, payload, run=None, actor="operator")


def _point(x, y):
    return {"x": x, "y": y}


def _feature(name, x, y):
    return {"name": name, "px": _point(x, y)}


def test_how_far_a_decision_moved_what_was_proposed():
    records = [
        # a point of interest moved 3 µm right and 4 µm up
        _decided(
            0,
            POINT_OF_INTEREST,
            {"poi": _point(0.0, 0.0)},
            {"poi": _point(3e-6, 4e-6)},
            "poi",
        ),
        # a detection: one feature left alone, the other moved 6 px
        _decided(
            1,
            DETECTION,
            {"features": [_feature("A", 10, 10), _feature("B", 20, 20)]},
            {"features": [_feature("A", 10, 10), _feature("B", 26, 20)]},
            "detection",
            checkpoint="autolamella-mega.pt",
        ),
        # an alignment area: changed, with no distance to give
        _decided(
            2,
            ALIGNMENT_AREA,
            {"alignment_area": {"left": 0.1, "top": 0.1, "width": 0.3, "height": 0.3}},
            {"alignment_area": {"left": 0.2, "top": 0.1, "width": 0.3, "height": 0.3}},
            "area",
        ),
        # a point confirmed where it was, read back a hair off
        _decided(
            3,
            POINT_OF_INTEREST,
            {"poi": _point(1e-6, 0.0)},
            {"poi": _point(1e-6 * (1 + 1e-12), 0.0)},
        ),
    ]

    decisions = {
        d["proposal_id"]: d for d in event_tables(records).decisions.to_dict("records")
    }

    assert decisions["poi"]["moved"] == pytest.approx(5e-6)
    assert (decisions["poi"]["unit"], decisions["poi"]["changed"]) == ("m", True)
    assert (decisions["detection"]["moved"], decisions["detection"]["unit"]) == (
        6.0,
        "px",
    )
    assert decisions["detection"]["checkpoint"] == "autolamella-mega.pt"
    area = decisions["area"]
    assert area["changed"] and pd.isna(area["moved"]) and pd.isna(area["unit"])
    assert (decisions["p1"]["changed"], decisions["p1"]["moved"]) == (False, 0.0)
    assert {d["actor"] for d in decisions.values()} == {"operator"}


def test_a_confirmation_is_completed_by_its_fill_in():
    """Confirmed "as it stands", then filled in with where the stage was: one
    decision, which moved it 2 µm."""
    asked_at = {"x": 1e-3, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0}
    moved_to = dict(asked_at, x=1.002e-3)
    records = [
        _record("proposal_asked", 0, {"proposal_id": "s", "kind": STATE}),
        _decided(5, STATE, {"stage_position": asked_at}, {}, "s"),
        _decided(
            6,
            STATE,
            {"stage_position": asked_at},
            {"stage_position": moved_to},
            "s",
            filled_in=True,
        ),
    ]

    ((decision),) = event_tables(records).decisions.to_dict("records")

    assert decision["time"] == T0 + timedelta(seconds=5)
    assert decision["waited"] == 5.0
    assert decision["changed"]
    assert decision["moved"] == pytest.approx(2e-6)


def test_a_second_look_did_not_wait():
    records = [
        _record("proposal_asked", 0, {"proposal_id": "p", "kind": POINT_OF_INTEREST}),
        _decided(4, POINT_OF_INTEREST, {"poi": _point(0.0, 0.0)}, {}, "p"),
        _decided(
            100,
            POINT_OF_INTEREST,
            {"poi": _point(0.0, 0.0)},
            {"poi": _point(1e-6, 0.0)},
            "p",
            1,
        ),
    ]

    decisions = event_tables(records).decisions.to_dict("records")

    assert decisions[0]["waited"] == 4.0 and pd.isna(decisions[1]["waited"])
    assert [d["changed"] for d in decisions] == [False, True]


def test_read_from_the_file_or_its_folder(microscope, experiment):
    run_tasks(microscope, experiment, [SETUP])
    folder = Path(experiment.path)

    by_folder = read_event_tables(folder).runs
    by_file = read_event_tables(folder / EVENTS_FILENAME).runs

    assert by_folder.equals(by_file) and len(by_folder) == 1


def _edit(second, before, after, actor="operator", via="lamella editor", run=None):
    """An edit to the second lamella's rough milling pattern, as the editor
    records it: on the item and task it edited, whatever run was going."""
    payload = {
        "item": {"id": "L2", "name": "02-lamella"},
        "task": ROUGH,
        "target": "milling.mill_rough",
        "via": via,
        "before": before,
        "after": after,
    }
    return _record("edit", second, payload, run=run, actor=actor)


def test_an_edit_names_the_fields_it_changed():
    milling = MillRoughTaskConfig(task_name=ROUGH).milling["mill_rough"]
    edited = deepcopy(milling)
    edited.stages[0].pattern.depth = 2.5e-6
    edited.stages[1].milling.milling_current = 1e-9
    # read back through a widget in µm: a hair off, and not a change
    rounded = deepcopy(milling)
    rounded.field_of_view = milling.field_of_view * (1 + 1e-12)
    records = [
        # made while a run on the first lamella was going
        _edit(0, milling.to_dict(), edited.to_dict(), run=RUN),
        _edit(
            5, milling.to_dict(), rounded.to_dict(), actor="agent", via="agent patch"
        ),
    ]

    edits = event_tables(records).edits.to_dict("records")

    first, second = edits
    assert (first["item"], first["item_id"], first["task"]) == (
        "02-lamella",
        "L2",
        ROUGH,
    )
    assert (first["target"], first["via"], first["actor"]) == (
        "milling.mill_rough",
        "lamella editor",
        "operator",
    )
    assert first["fields"] == [
        "milling.mill_rough.stages.0.pattern.depth",
        "milling.mill_rough.stages.1.milling.milling_current",
    ]
    assert first["changes"] == 2
    assert (second["changes"], second["fields"]) == (0, [])
    assert (second["actor"], second["via"]) == ("agent", "agent patch")


def test_an_edit_to_one_setting_names_it():
    payload = {
        "item": {"id": "L2", "name": "02-lamella"},
        "task": ROUGH,
        "target": "parameters.reacquire_alignment_reference",
        "via": "lamella editor",
        "before": False,
        "after": True,
    }

    ((edit),) = event_tables([_record("edit", 0, payload, run=None)]).edits.to_dict(
        "records"
    )

    assert edit["fields"] == ["parameters.reacquire_alignment_reference"]
    assert edit["changes"] == 1


def test_who_did_what():
    records = [
        _started(0),
        _step(0, "MILL_LAMELLA"),
        _step(5, "ACQUIRE_REFERENCE_IMAGES"),
        _edit(6, {"a": 1}, {"a": 2}),
        _edit(7, {"a": 2}, {"a": 3}),
        _edit(8, {"a": 3}, {"a": 4}, actor="agent"),
        _record("task_completed", 9),
        {"kind": "task_step", "t": T0.isoformat(), "payload": {}},  # no actor
    ]

    actors = event_tables(records).actors
    # a record from before actors were recorded has none (NaN on pandas 3)
    counts = {
        (None if pd.isna(a) else a, k): n for a, k, n in actors.itertuples(index=False)
    }

    assert counts == {
        ("task", "task_started"): 1,
        ("task", "task_step"): 2,
        ("operator", "edit"): 2,
        ("agent", "edit"): 1,
        ("task", "task_completed"): 1,
        (None, "task_step"): 1,
    }

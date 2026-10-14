"""The History tab's rows: one per task run, with its operations (FIB-1256).

``history_rows`` joins a lamella's ``task_history`` to the operation events its
runs recorded. The first test is a real run on Demo read back from the
``events.jsonl`` it wrote; the rest build the history and records by hand for the
cases a Demo run cannot produce: a failed run, an old experiment, a re-run.
"""

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.event_recording import EVENTS_FILENAME, read_events
from fibsem.applications.autolamella.history_rows import (
    OUTSIDE_A_TASK,
    HistoryFilter,
    filter_runs,
    history_rows,
    milling_stages,
    operation_facts,
    operations_need_a_look,
    operations_summary,
)
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    AutoLamellaWorkflowConfig,
    Experiment,
    Lamella,
)
from fibsem.applications.autolamella.workflows.tasks.manager import run_tasks
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)

SETUP, ROUGH = "Setup Lamella Position", "Rough Milling"
T0 = datetime(2026, 7, 29, 18, 0, tzinfo=timezone(timedelta(hours=10)))


# --- a real run, read back --------------------------------------------------


def test_a_real_run_reads_back_as_one_row_per_run_with_its_alignment(tmp_path):
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
        setup_logging=False,
    )
    try:
        exp = Experiment(path=tmp_path, name="history")
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
        lamella = exp.positions[0]
        lamella.path.mkdir(parents=True, exist_ok=True)
        lamella.milling_pose = microscope.get_microscope_state()

        run_tasks(microscope, exp, [SETUP, ROUGH])
        rows = history_rows(lamella, read_events(Path(exp.path) / EVENTS_FILENAME))
    finally:
        microscope.disconnect()

    assert [r.task_name for r in rows] == [SETUP, ROUGH]
    assert all(r.status is AutoLamellaTaskStatus.Completed for r in rows)
    setup, rough = rows
    assert setup.operations == []
    (align,) = [op for op in rough.operations if op.kind == "alignment"]
    assert align.label == "Align" and align.status == "completed"
    assert " µm · 3 steps" in align.detail
    # the mill, after the task's own alignment, each stage with its drift
    # correction under it rather than a line of its own (FIB-1257, FIB-1351)
    (mill,) = [op for op in rough.operations if op.kind == "milling"]
    assert rough.operations == [align, mill]
    stages = [
        stage.name
        for m in MillRoughTaskConfig(task_name=ROUGH).milling.values()
        for stage in m.enabled_stages
    ]
    assert mill.label == "Mill" and mill.status == "completed"
    assert mill.detail.startswith(f"{len(stages)} stages · alignment ≤ ")
    lines = milling_stages(mill)
    assert [line.name for line in lines] == stages
    for line in lines:
        (drift,) = line.drift
        assert drift.status == "completed" and drift.plot is not None
    assert align.duration is not None and align.path and os.path.isdir(align.path)
    assert rough.images and all(os.path.isfile(p) for p in rough.images)
    assert rough.duration is not None and rough.duration >= align.duration


def test_a_task_run_twice_keeps_both_runs_images_in_their_own_folders(tmp_path):
    """Each run saves into ``<lamella>/<task>/<YYYYMMDD-HHMMSS>/`` (FIB-1275): a
    re-run overwrites nothing, and the History tab shows each run's own images
    rather than "replaced by the later run"."""
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(
        manufacturer="Demo",
        config_path=os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml"),
        setup_logging=False,
    )
    try:
        exp = Experiment(path=tmp_path, name="rerun")
        os.makedirs(exp.path, exist_ok=True)
        exp.task_protocol = AutoLamellaTaskProtocol(
            workflow_config=AutoLamellaWorkflowConfig(
                tasks=[AutoLamellaTaskDescription(name=SETUP, required=True)]
            )
        )
        exp.add_new_lamella(
            microscope.get_microscope_state(),
            EventedDict(
                {
                    SETUP: SelectMillingPositionTaskConfig(
                        task_name=SETUP, use_autofocus=False
                    )
                }
            ),
        )
        lamella = exp.positions[0]
        lamella.path.mkdir(parents=True, exist_ok=True)
        lamella.milling_pose = microscope.get_microscope_state()

        run_tasks(microscope, exp, [SETUP])
        run_tasks(microscope, exp, [SETUP])
        rows = history_rows(lamella, read_events(Path(exp.path) / EVENTS_FILENAME))
    finally:
        microscope.disconnect()

    first, second = rows
    assert first.images and second.images
    assert set(first.images).isdisjoint(second.images), "nothing overwritten"
    assert first.images_replaced_by is None
    for row in rows:
        folders = {Path(p).relative_to(lamella.path).parts[:2] for p in row.images}
        assert len(folders) == 1, "one run, one folder"
        ((task_folder, stamp),) = folders
        assert task_folder == SETUP and len(stamp) == len("20261014-093012")
    # the lamella's shared state stays in its folder, once
    assert (lamella.path / "ref_alignment_ib.tif").is_file()


# --- built by hand ------------------------------------------------------------


def _lamella(tmp_path) -> Lamella:
    lamella = Lamella(path=tmp_path / "lam", number=1, petname="lam")
    lamella.path.mkdir(parents=True, exist_ok=True)
    return lamella


def _image(lamella, name):
    (lamella.path / name).write_bytes(b"")
    return name


def _run(name, task_id, minute, status=AutoLamellaTaskStatus.Completed, **outputs):
    return AutoLamellaTaskState(
        name=name,
        task_id=task_id,
        status=status,
        start_timestamp=T0 + timedelta(minutes=minute),
        end_timestamp=T0 + timedelta(minutes=minute, seconds=30),
        outputs=dict(outputs),
    )


def _event(lamella, task_id, kind, **payload):
    return {
        "kind": kind,
        "t": (T0 + timedelta(minutes=5)).isoformat(),
        "item": {"id": lamella.id, "name": lamella.name},
        "task": {"id": task_id, "name": "?"},
        "payload": payload,
    }


def test_a_rerun_takes_the_reference_images_it_wrote_over(tmp_path):
    """The same filename, written by both runs, is the later run's picture."""
    lamella = _lamella(tmp_path)
    ref = _image(lamella, f"ref_{ROUGH}_final_res_01_ib.tif")
    lamella.task_history = [
        _run(ROUGH, "first", 0, final_fib=[ref]),
        _run(ROUGH, "second", 10, final_fib=[ref]),
    ]

    first, second = history_rows(lamella, [])

    assert first.images == [] and first.images_replaced_by == "second"
    assert second.images == [str(lamella.path / ref)]
    assert second.images_replaced_by is None


def test_a_failed_run_does_not_borrow_an_earlier_runs_images(tmp_path):
    """It recorded no final images, so it made none: the filename fallback, for
    experiments older than recorded outputs, would hand it the earlier run's."""
    lamella = _lamella(tmp_path)
    ref = _image(lamella, f"ref_{ROUGH}_final_res_01_ib.tif")
    lamella.task_history = [
        _run(ROUGH, "ok", 0, final_fib=[ref]),
        _run(ROUGH, "broke", 10, status=AutoLamellaTaskStatus.Failed),
    ]

    ok, broke = history_rows(lamella, [])

    assert ok.images == [str(lamella.path / ref)]
    assert broke.images == [] and broke.status is AutoLamellaTaskStatus.Failed
    assert ok.images_replaced_by is None, "it recorded none, so it replaced none"


def test_an_experiment_from_before_recorded_outputs_still_shows_its_images(tmp_path):
    """Found by filename, on the last run of the task."""
    lamella = _lamella(tmp_path)
    ref = _image(lamella, f"ref_{ROUGH}_final_res_01_ib.tif")
    lamella.task_history = [_run(ROUGH, "a", 0), _run(ROUGH, "b", 10)]

    a, b = history_rows(lamella, [])

    assert a.images == [] and b.images == [str(lamella.path / ref)]
    assert a.images_replaced_by == "b"


def test_the_run_still_going_is_a_row(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(SETUP, "done", 0)]
    lamella.task_state = _run(ROUGH, "now", 10, status=AutoLamellaTaskStatus.InProgress)

    rows = history_rows(lamella, [_event(lamella, "now", "alignment", results=[])])

    assert [(r.task_name, r.status) for r in rows] == [
        (SETUP, AutoLamellaTaskStatus.Completed),
        (ROUGH, AutoLamellaTaskStatus.InProgress),
    ]
    assert len(rows[1].operations) == 1


def test_each_operation_says_how_it_ended(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    shift = {"shift": {"x": 3e-7, "y": 4e-7}}
    events = [
        _event(
            lamella,
            "r",
            "alignment",
            status="skipped",
            reason="reference image x does not exist",
        ),
        _event(
            lamella, "r", "alignment", status="failed", error="RuntimeError: no signal"
        ),
        _event(lamella, "r", "autofocus", beam_type="ION", status="cancelled"),
        _event(
            lamella,
            "r",
            "alignment",
            results=[shift, shift],
            status="completed",
            started_at=(T0 + timedelta(minutes=4, seconds=55)).isoformat(),
        ),
        # written before operations said how they ended: only ever for a completed run
        _event(
            lamella,
            "r",
            "autofocus",
            beam_type="ELECTRON",
            initial_working_distance=0.004,
            working_distance=0.00412,
            steps=11,
        ),
    ]

    (run,) = history_rows(lamella, events)

    # in the order they started; all end at once, and only one says when it began
    assert [(op.status, op.detail) for op in run.operations] == [
        ("completed", "1.00 µm · 2 steps"),
        ("skipped", "skipped: reference image x does not exist"),
        ("failed", "failed: RuntimeError: no signal"),
        ("cancelled", "cancelled"),
        ("completed", "SEM WD 4.000 → 4.120 mm · 11 probes"),
    ]
    assert run.operations[0].duration == pytest.approx(5.0)


def test_an_expanded_operation_lists_each_step_and_why_it_stopped(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    saved = tmp_path / "Alignment" / "lam - Rough Milling-18-05-00"
    saved.mkdir(parents=True)
    (saved / "figure.png").write_bytes(b"")
    events = [
        _event(
            lamella,
            "r",
            "alignment",
            status="completed",
            beam_type="ION",
            method="cross-correlation",
            subsystem="beam-shift",
            steps=3,
            results=[
                {"shift": {"x": 3e-8, "y": -1e-8}, "score": 0.8},
                {"shift": {"x": 0.0, "y": 0.0}, "score": 0.2, "success": False},
            ],
            validation={"agreement": False, "max_disagreement_px": 70.04},
            path=str(saved),
        ),
        _event(lamella, "r", "alignment", status="failed", error="RuntimeError: x"),
        _event(
            lamella,
            "r",
            "autofocus",
            status="completed",
            beam_type="ELECTRON",
            hfw=1.5e-4,
            steps=22,
            initial_working_distance=0.004,
            working_distance=0.0042,
            path=str(tmp_path / "moved away"),
        ),
    ]

    (run,) = history_rows(lamella, events)
    align, failed, focus = run.operations

    assert operation_facts(align) == [
        ("Beam", "FIB"),
        ("Method", "cross-correlation"),
        ("Corrected by", "beam-shift"),
        ("Step 1", "(+30, -10) nm · score 0.80"),
        ("Step 2", "(+0, +0) nm · score 0.20 · too weak, not applied"),
        ("Steps", "2 of 3"),
        ("Methods", "disagree · 70.0 px apart at most"),
    ]
    assert align.plot == str(saved / "figure.png")
    assert operation_facts(failed)[0] == ("Error", "RuntimeError: x")
    assert ("Working distance", "4.000 → 4.200 mm") in operation_facts(focus)
    assert focus.plot is None, "its folder is not where the record says"


def test_only_this_lamellas_operations_are_its_own(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    other = dict(_event(lamella, "r", "alignment"), item={"id": "someone-else"})
    unstamped = dict(_event(lamella, "r", "alignment"), task=None)
    not_an_operation = _event(lamella, "r", "stage_moved")

    run, outside = history_rows(lamella, [other, unstamped, not_an_operation])

    assert run.operations == []
    # no task ran it: a row of its own, not the run's (FIB-1375)
    assert outside.outside_task and len(outside.operations) == 1


def _outside(lamella, minute, kind="autofocus", actor="agent", session="s1"):
    return dict(
        _event(lamella, "", kind, status="completed"),
        t=(T0 + timedelta(minutes=minute)).isoformat(),
        item=None,
        task=None,
        actor=actor,
        session=session,
    )


def test_operations_outside_a_task_are_a_row_per_session_among_the_runs(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(SETUP, "s", 0), _run(ROUGH, "r", 20)]
    events = [
        _outside(lamella, 12, actor="operator"),
        _outside(lamella, 10),
        _outside(lamella, 30, session="s2"),
    ]

    rows = history_rows(lamella, events)

    assert [row.task_name for row in rows] == [
        SETUP,
        OUTSIDE_A_TASK,
        ROUGH,
        OUTSIDE_A_TASK,
    ]
    first = rows[1]
    assert [op.label for op in first.operations] == [
        "Autofocus (agent)",
        "Autofocus (you)",
    ]
    assert first.started_at == T0 + timedelta(minutes=10)
    assert first.ended_at == T0 + timedelta(minutes=12)
    assert first.images == [] and rows[3].task_id == "session:s2"


# --- the filter ---------------------------------------------------------------


def _runs(tmp_path):
    lamella = _lamella(tmp_path)
    ref = _image(lamella, f"ref_{SETUP}_final_res_01_ib.tif")
    lamella.task_history = [
        _run(SETUP, "s", 0, final_fib=[ref]),
        _run(ROUGH, "r1", 10, status=AutoLamellaTaskStatus.Failed),
        _run(ROUGH, "r2", 20, status=AutoLamellaTaskStatus.Cancelled),
    ]
    return history_rows(lamella, [_event(lamella, "r1", "alignment", results=[])])


@pytest.mark.parametrize(
    "flt, expected",
    [
        (HistoryFilter(), ["s", "r1", "r2"]),
        (HistoryFilter(status="failed"), ["r1"]),
        (HistoryFilter(status="cancelled"), ["r2"]),
        (HistoryFilter(task=ROUGH), ["r1", "r2"]),
        (HistoryFilter(task="Polishing"), []),
        (HistoryFilter(show="images"), ["s"]),
        (HistoryFilter(show="operations"), ["r1"]),
        (HistoryFilter(status="failed", task=SETUP), []),
    ],
)
def test_the_filter_lets_through_what_it_names(tmp_path, flt, expected):
    assert [r.task_id for r in filter_runs(_runs(tmp_path), flt)] == expected


def test_a_saved_filter_round_trips_and_anything_unknown_shows_everything():
    flt = HistoryFilter(show="operations", status="failed", task=ROUGH)

    assert HistoryFilter.from_dict(flt.to_dict()) == flt and flt.active
    assert HistoryFilter.from_dict({"show": "pictures", "status": 3, "task": ""}) == (
        HistoryFilter()
    )
    assert HistoryFilter.from_dict(None) == HistoryFilter()
    assert not HistoryFilter().active


# --- milling (FIB-1257) -------------------------------------------------------


def _at(minute, second=0):
    return (T0 + timedelta(minutes=minute, seconds=second)).isoformat()


def _stage(name, index, status, start, end, **more):
    return {
        "name": name,
        "index": index,
        "status": status,
        "started_at": start,
        "ended_at": end,
        "milling_current": 7.4e-10,
        **more,
    }


def test_a_mill_says_how_it_ended_and_each_stage_is_a_line(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    turned_off = {
        "name": "Notch",
        "status": "skipped",
        "reason": "turned off in the protocol",
    }
    completed = _event(
        lamella,
        "r",
        "milling",
        status="completed",
        started_at=_at(1),
        stages=[
            _stage("Rough Mill 01", 0, "completed", _at(1), _at(3, 13)),
            turned_off,
            _stage("Rough Mill 02", 1, "completed", _at(3, 13), _at(3, 40)),
        ],
    )
    failed = _event(
        lamella,
        "r",
        "milling",
        status="failed",
        error="RuntimeError: rejected",
        started_at=_at(6),
        stages=[
            _stage("Rough Mill 01", 0, "failed", _at(6), _at(6, 2)),
            {
                "name": "Rough Mill 02",
                "index": 1,
                "status": "skipped",
                "reason": "the run failed before this stage",
            },
        ],
    )
    stopped = _event(
        lamella,
        "r",
        "milling",
        status="cancelled",
        started_at=_at(7),
        stages=[
            _stage("Rough Mill 01", 0, "completed", _at(7), _at(7, 30)),
            _stage("Rough Mill 02", 1, "cancelled", _at(7, 30), _at(7, 31)),
        ],
    )
    skipped = _event(
        lamella,
        "r",
        "milling",
        status="skipped",
        reason="Continue was pressed without running it",
        stages=[{"name": "Rough Mill 01", "status": "skipped"}],
    )
    events = [completed, failed, stopped, skipped]
    for event, minute in zip(events, (4, 6, 8, 9)):
        event["t"] = _at(minute)

    (run,) = history_rows(lamella, events)

    assert [(op.label, op.status, op.detail) for op in run.operations] == [
        ("Mill", "completed", "2 stages"),
        ("Mill", "failed", "failed at Rough Mill 01: RuntimeError: rejected"),
        ("Mill", "cancelled", "stopped at Rough Mill 02 · 1 of 2 stages milled"),
        ("Mill", "skipped", "skipped: Continue was pressed without running it"),
    ]
    assert [(line.name, line.text) for line in milling_stages(run.operations[0])] == [
        ("Rough Mill 01", "completed · 2 m 13 s · 740 pA"),
        ("Notch", "skipped: turned off in the protocol"),
        ("Rough Mill 02", "completed · 27 s · 740 pA"),
    ]
    assert operation_facts(run.operations[1]) == [("Error", "RuntimeError: rejected")]
    assert [(line.name, line.text) for line in milling_stages(run.operations[1])] == [
        ("Rough Mill 01", "failed · 2 s · 740 pA"),
        ("Rough Mill 02", "skipped: the run failed before this stage"),
    ]


def test_a_mill_comes_before_an_alignment_it_started_before(tmp_path):
    """Recorded after it, as it ends after it; listed first, as it started
    first. (No stage times, so nothing is folded under it.)"""
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    drift = _event(
        lamella, "r", "alignment", name="Rough Mill 01-18-01-05", started_at=_at(1, 1)
    )
    drift["t"] = _at(1, 4)
    mill = _event(lamella, "r", "milling", status="completed", started_at=_at(1))
    mill["t"] = _at(3)

    (run,) = history_rows(lamella, [drift, mill])

    assert [op.label for op in run.operations] == ["Mill", "Align (Rough Mill 01)"]


# --- drift corrections, under their stage (FIB-1351) -------------------------


def _drift(lamella, name, start, end, x=3e-8, **payload):
    event = _event(
        lamella,
        "r",
        "alignment",
        name=f"{name}-18-00-00",
        started_at=start,
        results=[{"shift": {"x": x, "y": 0.0}, "success": True}],
        status="completed",
        **payload,
    )
    event["t"] = end
    return event


def _mill(lamella, status="completed", **payload):
    event = _event(
        lamella,
        "r",
        "milling",
        status=status,
        started_at=_at(1),
        stages=[
            _stage("Rough Mill 01", 0, "completed", _at(1), _at(2)),
            _stage("Rough Mill 02", 1, "completed", _at(2), _at(3)),
        ],
        **payload,
    )
    event["t"] = _at(3)
    return event


def test_each_stage_s_drift_correction_folds_under_its_mill(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    own = _drift(lamella, "lam - Rough Milling", _at(0, 30), _at(0, 40))
    first = _drift(lamella, "Rough Mill 01", _at(1, 1), _at(1, 4), x=7e-8)
    second = _drift(lamella, "Rough Mill 02", _at(2, 1), _at(2, 4))
    after = _drift(lamella, "something later", _at(4), _at(4, 5))

    (run,) = history_rows(lamella, [own, first, second, _mill(lamella), after])

    # the task's own alignment and one after the mill keep their lines
    assert [op.label for op in run.operations] == [
        "Align",
        "Mill",
        "Align (something later)",
    ]
    mill = run.operations[1]
    assert mill.detail == "2 stages · alignment ≤ 0.07 µm"
    one, two = milling_stages(mill)
    assert [d.label for d in one.drift] == ["Align (Rough Mill 01)"]
    assert [d.label for d in two.drift] == ["Align (Rough Mill 02)"]


def test_a_drift_correction_on_a_stage_boundary_goes_to_the_stage_it_starts(tmp_path):
    """On a coarse clock (Windows, 15.6 ms) one stage's end, the next stage's
    start and its drift correction's start read the same instant."""
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    first = _drift(lamella, "Rough Mill 01", _at(1), _at(1, 4))
    second = _drift(lamella, "Rough Mill 02", _at(2), _at(2, 4))

    (run,) = history_rows(lamella, [first, second, _mill(lamella)])

    (mill,) = run.operations
    one, two = milling_stages(mill)
    assert [d.label for d in one.drift] == ["Align (Rough Mill 01)"]
    assert [d.label for d in two.drift] == ["Align (Rough Mill 02)"]


def test_a_drift_correction_that_went_wrong_shows_on_the_folded_line(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    weak = _drift(lamella, "Rough Mill 01", _at(1, 1), _at(1, 4))
    weak["payload"]["results"][0]["success"] = False
    failed = _event(
        lamella,
        "r",
        "alignment",
        name="Rough Mill 02",
        started_at=_at(2, 1),
        status="failed",
        error="RuntimeError: no signal",
    )
    failed["t"] = _at(2, 2)

    (run,) = history_rows(lamella, [weak, _mill(lamella)])
    assert run.operations[0].detail == "2 stages · alignment too weak at Rough Mill 01"

    (run,) = history_rows(lamella, [failed, _mill(lamella)])
    assert run.operations[0].detail == "2 stages · alignment failed at Rough Mill 02"


def test_a_mill_that_did_not_complete_does_not_add_the_drift(tmp_path):
    """It already says why it stopped; the drift is in its opened stages."""
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    first = _drift(lamella, "Rough Mill 01", _at(1, 1), _at(1, 4))
    mill = _mill(lamella, status="failed", error="RuntimeError: rejected")
    mill["payload"]["stages"][1]["status"] = "failed"

    (run,) = history_rows(lamella, [first, mill])

    (only,) = run.operations
    assert only.detail == "failed at Rough Mill 02: RuntimeError: rejected"
    assert milling_stages(only)[0].drift[0].label == "Align (Rough Mill 01)"


def test_an_fm_autofocus_has_its_duration_and_figure(tmp_path):
    """It records when it started and where it saved, as the beam autofocus
    does (FIB-1257)."""
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run("Acquire Fluorescence Image", "f", 0)]
    saved = tmp_path / "autofunctions" / "af"
    saved.mkdir(parents=True)
    (saved / "plot.png").write_bytes(b"")
    event = _event(
        lamella,
        "f",
        "fm_autofocus",
        status="completed",
        started_at=_at(4, 40),
        passes=2,
        completed_passes=2,
        initial_position=1e-6,
        position=3e-6,
        path=str(saved),
    )

    (run,) = history_rows(lamella, [event])

    (op,) = run.operations
    assert op.label == "Autofocus (FM)" and op.detail == "objective +2.00 µm"
    assert op.duration == pytest.approx(20.0)
    assert op.plot == str(saved / "plot.png")


# --- what each stage's strategy said (FIB-1257) --------------------------------


def test_a_strategy_with_only_facts_says_them_on_its_stage_s_line(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    mill = _mill(lamella)
    for stage in mill["payload"]["stages"]:
        stage["strategy"] = {"name": "Standard", "facts": {"Pattern": "Trench"}}

    (run,) = history_rows(lamella, [mill])

    (op,) = run.operations
    assert op.detail == "2 stages"
    first, _ = milling_stages(op)
    assert first.text == "completed · 1 m 00 s · 740 pA · Standard · Pattern Trench"
    assert first.strategy == "" and first.said == ""


def test_a_strategy_that_says_why_it_stopped_has_its_own_line(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    summary = tmp_path / "run_summary.png"
    summary.write_bytes(b"")
    mill = _mill(lamella)
    mill["payload"]["stages"] = mill["payload"]["stages"][:1]
    mill["payload"]["stages"][0]["strategy"] = {
        "name": "CoincidenceMilling",
        "end_reason": "intensity drop",
        "facts": {"Mill time (s)": 250.0, "Intensity drop seen": True},
        "figure": str(summary),
    }

    (run,) = history_rows(lamella, [mill])

    (op,) = run.operations
    # folded, the line still says why it stopped
    assert op.detail == "1 stage · intensity drop"
    (stage,) = milling_stages(op)
    assert stage.text == "completed · 1 m 00 s · 740 pA"
    assert stage.strategy == "Coincidence Milling"
    assert stage.said == "intensity drop · Mill time (s) 250 · Intensity drop seen yes"
    assert stage.figure == str(summary)


# --- a run's operations, folded unless one needs a look (FIB-1361) ------------


def _aligned(lamella, minute, **payload):
    event = _event(
        lamella,
        "r",
        "alignment",
        status="completed",
        started_at=_at(minute),
        results=[{"shift": {"x": 1e-8, "y": 0.0}, "success": True}],
        **payload,
    )
    event["t"] = _at(minute, 5)
    return event


def test_a_run_that_went_to_plan_folds(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    disagree = {"agreement": False, "max_disagreement_px": 9.0}

    (run,) = history_rows(
        lamella, [_aligned(lamella, 0, validation=disagree), _mill(lamella)]
    )

    assert operations_summary(run) == "2 operations · all completed"
    # methods disagreeing is routine on the Demo, and not a reason to open
    assert not operations_need_a_look(run)


def test_a_run_that_did_not_opens_and_says_what(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    skipped = _event(lamella, "r", "alignment", status="skipped", reason="no image")
    failed = _mill(lamella, status="failed", error="RuntimeError: rejected")

    (run,) = history_rows(lamella, [skipped, failed])

    assert operations_summary(run) == "2 operations · 1 failed · 1 skipped"
    assert operations_need_a_look(run)


def test_a_weak_or_failed_alignment_opens_its_run(tmp_path):
    """Its own, or one a mill's stage ran: both completed operations otherwise."""
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    weak = _aligned(lamella, 0)
    weak["payload"]["results"][0]["success"] = False
    (run,) = history_rows(lamella, [weak, _mill(lamella)])
    assert operations_need_a_look(run)

    in_stage = _event(
        lamella,
        "r",
        "alignment",
        status="failed",
        error="RuntimeError: no signal",
        started_at=_at(1, 1),
    )
    in_stage["t"] = _at(1, 2)
    (run,) = history_rows(lamella, [_aligned(lamella, 0), in_stage, _mill(lamella)])
    assert len(run.operations) == 2  # the failed one is under its stage
    assert operations_need_a_look(run)

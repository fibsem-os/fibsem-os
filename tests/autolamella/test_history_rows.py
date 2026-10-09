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
    HistoryFilter,
    filter_runs,
    history_rows,
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
    align, *drift = [op for op in rough.operations if op.kind == "alignment"]
    assert align.label == "Align" and align.status == "completed"
    # each milling stage's drift correction is stamped with the same run, and named
    # for its stage
    stages = {
        stage.name
        for m in MillRoughTaskConfig(task_name=ROUGH).milling.values()
        for stage in m.stages
    }
    assert drift and {op.label for op in drift} <= {
        f"Align ({name})" for name in stages
    }
    assert " µm · 3 steps" in align.detail
    assert align.duration is not None and align.path and os.path.isdir(align.path)
    assert rough.images and all(os.path.isfile(p) for p in rough.images)
    assert rough.duration is not None and rough.duration >= align.duration


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

    assert first.images == []
    assert second.images == [str(lamella.path / ref)]


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


def test_an_experiment_from_before_recorded_outputs_still_shows_its_images(tmp_path):
    """Found by filename, on the last run of the task."""
    lamella = _lamella(tmp_path)
    ref = _image(lamella, f"ref_{ROUGH}_final_res_01_ib.tif")
    lamella.task_history = [_run(ROUGH, "a", 0), _run(ROUGH, "b", 10)]

    a, b = history_rows(lamella, [])

    assert a.images == [] and b.images == [str(lamella.path / ref)]


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

    assert [(op.status, op.detail) for op in run.operations] == [
        ("skipped", "skipped: reference image x does not exist"),
        ("failed", "failed: RuntimeError: no signal"),
        ("cancelled", "cancelled"),
        ("completed", "1.00 µm · 2 steps"),
        ("completed", "SEM WD 4.000 → 4.120 mm · 11 probes"),
    ]
    assert run.operations[3].duration == pytest.approx(5.0)


def test_only_this_lamellas_operations_are_its_own(tmp_path):
    lamella = _lamella(tmp_path)
    lamella.task_history = [_run(ROUGH, "r", 0)]
    other = dict(_event(lamella, "r", "alignment"), item={"id": "someone-else"})
    unstamped = dict(_event(lamella, "r", "alignment"), task=None)
    not_an_operation = _event(lamella, "r", "stage_moved")

    (run,) = history_rows(lamella, [other, unstamped, not_an_operation])

    assert run.operations == []


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

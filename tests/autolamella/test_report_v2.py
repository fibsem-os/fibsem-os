"""Report v2, the page built from ``events.jsonl`` (FIB-1036).

The first tests write the report of a real run through ``run_tasks`` on Demo.
The rest render records written here in the recorder's shape, for what a
headless Demo run cannot produce: time spent waiting, idle gaps, failed and
retried runs.
"""

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from psygnal.containers import EventedDict

import fibsem.config as cfg
from fibsem import utils
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaTaskProtocol,
    AutoLamellaWorkflowConfig,
    Experiment,
)
from fibsem.applications.autolamella.tools.event_tables import event_tables
from fibsem.applications.autolamella.tools.report_v2 import (
    REPORT_DIRNAME,
    REPORT_FILENAME,
    render_report,
    summarise,
    write_report,
)
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
    exp = Experiment(path=tmp_path, name="report")
    os.makedirs(exp.path, exist_ok=True)
    exp.task_protocol = AutoLamellaTaskProtocol(
        workflow_config=AutoLamellaWorkflowConfig(
            tasks=[
                AutoLamellaTaskDescription(name=SETUP, required=True),
                AutoLamellaTaskDescription(name=ROUGH, required=True),
            ]
        )
    )
    for _ in range(2):
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
    for lamella in exp.positions:
        lamella.path.mkdir(parents=True, exist_ok=True)
        lamella.milling_pose = microscope.get_microscope_state()
    return exp


def _count(page, pattern):
    return len(re.findall(pattern, page))


# ── a real run ───────────────────────────────────────────────────────────────


def test_the_report_of_a_run_is_written_beside_its_record(microscope, experiment):
    run_tasks(microscope, experiment, [SETUP, ROUGH])

    path = write_report(experiment)

    assert path == Path(experiment.path) / REPORT_DIRNAME / REPORT_FILENAME
    page = path.read_text(encoding="utf-8")
    first, second = (p.name for p in experiment.positions)
    for name in (experiment.name, first, second, SETUP, ROUGH):
        assert name in page
    # every lamella finished every task: four runs, each a completed cell and a bar
    assert _count(page, r'class="cell completed"') == 4
    assert _count(page, r'class="run"') == 4
    assert "2 of 2" in page
    # one file, with nothing to fetch
    assert not re.search(r"(src|href)=\"(https?:)?//", page)
    assert "<script src" not in page and "<link" not in page


def test_an_experiment_recorded_before_the_event_stream_has_no_v2_report(tmp_path):
    experiment = Experiment(path=tmp_path, name="older")
    os.makedirs(experiment.path, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="recorded before the event stream"):
        write_report(experiment)
    assert not (Path(experiment.path) / REPORT_DIRNAME).exists()


# ── records the recorder writes ──────────────────────────────────────────────

T0 = datetime(2026, 9, 24, 9, 0, 0)
A, B, C = "01-lamella", "02-lamella", "03-lamella"


def _record(kind, second, item, task, run, payload=None, actor="task"):
    return {
        "session": "s1",
        "actor": actor,
        "kind": kind,
        "t": (T0 + timedelta(seconds=second)).isoformat() + "+10:00",
        "item": {"id": item, "name": item},
        "task": {"id": run, "name": task},
        "payload": payload or {},
    }


def _run(item, task, start, end, ending="task_completed", error=None, wait=None):
    run = f"{item}/{task}/{start}"
    records = [
        _record("task_started", start, item, task, run),
        _record("task_step", start, item, task, run, {"step": "MILL_LAMELLA"}),
    ]
    if wait is not None:  # a prompt raised at wait[0], answered at wait[1]
        nonce = f"{run}/prompt"
        records += [
            _record("prompt_raised", wait[0], item, task, run, {"nonce": nonce}),
            _record("prompt_answered", wait[1], item, task, run, {"nonce": nonce}),
        ]
    if ending:
        payload = {"error": error} if error else {}
        records.append(_record(ending, end, item, task, run, payload))
    return records


def _render(records, items=(A, B, C), tasks=(SETUP, ROUGH), name="session"):
    tables = event_tables(records)
    page = render_report(
        tables, name=name, items=items, tasks=tasks, generated=T0 + timedelta(hours=9)
    )
    return tables, page


def test_where_the_time_went():
    """Two runs with a gap of two minutes between them, the first waiting 10 s
    for an answer: 190 s machine, 10 s waiting, 120 s idle, of 320 s."""
    records = _run(A, SETUP, 0, 100, wait=(20, 30)) + _run(A, ROUGH, 220, 320)

    tables, page = _render(records, items=(A,))

    summary = summarise(tables, [A], [SETUP, ROUGH])
    assert (summary.span, summary.machine, summary.waiting, summary.idle) == (
        320.0,
        190.0,
        10.0,
        120.0,
    )
    assert summary.idle_gaps == [
        (T0 + timedelta(seconds=100), T0 + timedelta(seconds=220))
    ]
    assert summary.finished == [A]
    assert summary.throughput == pytest.approx(1 / (320 / 3600))
    for shown in ("59%", "3 min", "3%", "10 s", "38%", "2 min, nothing running"):
        assert shown in page
    assert "idle 2 min" in page  # the gap, labelled on the timeline
    # a minute apart, across the five minutes
    ticks = re.findall(r'text-anchor="middle">(\d\d:\d\d)</text>', page)
    assert ticks == ["09:00", "09:01", "09:02", "09:03", "09:04", "09:05"]
    assert _count(page, r'class="wait"') == 1


def test_a_gap_that_is_the_queue_moving_on_is_not_idle_time_worth_showing():
    records = _run(A, SETUP, 0, 100) + _run(A, ROUGH, 120, 220)

    tables, page = _render(records, items=(A,))

    assert summarise(tables, [A], [SETUP, ROUGH]).idle_gaps == []
    assert 'text-anchor="middle">idle' not in page


def test_each_lamella_s_outcome():
    records = (
        _run(A, SETUP, 0, 60)
        + _run(A, ROUGH, 60, 245, "task_failed", "Alignment failed")
        + _run(A, ROUGH, 300, 900)
        + _run(B, SETUP, 900, 960)
        + _run(B, ROUGH, 960, 1168, "task_cancelled", "Workflow aborted by user.")
    )

    _, page = _render(records)

    assert "10:00 · after 1 failed" in page  # the retry, and how the first went
    assert "cancelled at 3:28" in page
    assert 'title="Rough Milling: cancelled — Workflow aborted by user."' in page
    # C never ran, and B never finished
    assert _count(page, r'class="cell none">not run') == 2
    assert "1 of 3" in page
    # the failed run is on the timeline, and so is its key
    assert "Alignment failed" in page
    assert _count(page, r"<i style=\"background:#E24B4A\"></i>failed") == 1


def test_the_key_names_only_what_happened():
    _, page = _render(_run(A, SETUP, 0, 60))

    assert "</i>failed" not in page and "</i>cancelled" not in page
    assert "</i>waiting for an answer" not in page


def test_a_run_that_never_ended_is_drawn_to_where_it_was_last_heard_from():
    records = _run(A, SETUP, 0, 60) + _run(A, ROUGH, 60, None, ending=None)
    records.append(
        _record("task_step", 400, A, ROUGH, f"{A}/{ROUGH}/60", {"step": "MILL"})
    )

    tables, page = _render(records, items=(A,))

    assert summarise(tables, [A], [SETUP, ROUGH]).span == 400.0
    assert re.search(r'class="cell unfinished"[^>]*>unfinished</td>', page)
    assert _count(page, r'class="run unfinished"') == 1


def test_a_lamella_or_task_outside_the_workflow_is_still_shown():
    records = _run("99-extra", "Spot Burn Fiducial", 0, 60)

    _, page = _render(records, items=(A,), tasks=(SETUP,))

    assert "99-extra" in page and "Spot Burn Fiducial" in page


def test_names_are_escaped():
    records = _run("<b>lamella</b>", SETUP, 0, 60)

    _, page = _render(records, items=(), name='grid "A" & <co>')

    assert "<b>lamella</b>" not in page
    assert "&lt;b&gt;lamella&lt;/b&gt;" in page
    assert "grid &quot;A&quot; &amp; &lt;co&gt;" in page


def test_nothing_ran():
    _, page = _render([])

    assert "No task runs were recorded." in page
    assert "0 of 3" in page

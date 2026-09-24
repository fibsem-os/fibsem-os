"""Tools → Reporting → Generate Report v2 (preview) writes the page from the
experiment's events.jsonl and opens it (FIB-1036).

The real main window, with no microscope: the report reads records only. What
the browser is asked to open is caught, not opened.
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtGui import QDesktopServices  # noqa: E402

from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EVENTS_FILENAME,
)
from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.tools.report_v2 import (  # noqa: E402
    REPORT_DIRNAME,
    REPORT_FILENAME,
)


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import AutoLamellaMainUI as module

    win = module.AutoLamellaSingleWindowUI()
    yield win
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def seen(window, monkeypatch):
    """What the window opened, and what it said."""
    seen = {"opened": [], "toasts": []}
    monkeypatch.setattr(
        QDesktopServices, "openUrl", lambda url: seen["opened"].append(url) or True
    )
    monkeypatch.setattr(
        window,
        "show_toast",
        lambda message, level="info": seen["toasts"].append((level, message)),
    )
    return seen


def _experiment(tmp_path, name, recorded=True):
    exp = Experiment(path=tmp_path, name=name)
    os.makedirs(exp.path, exist_ok=True)
    if recorded:
        t0 = datetime(2026, 9, 24, 9, 0, 0)
        records = [
            {
                "session": "s1",
                "actor": "task",
                "kind": kind,
                "t": (t0 + timedelta(seconds=second)).isoformat() + "+10:00",
                "item": {"id": "L1", "name": "01-lamella"},
                "task": {"id": "r1", "name": "Rough Milling"},
                "payload": {},
            }
            for kind, second in (("task_started", 0), ("task_completed", 600))
        ]
        with open(Path(exp.path) / EVENTS_FILENAME, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    return exp


def test_the_report_is_written_and_opened(window, seen, tmp_path):
    experiment = _experiment(tmp_path, "recorded")
    window.autolamella_ui.experiment = experiment

    window.action_generate_report_v2.trigger()

    path = Path(experiment.path) / REPORT_DIRNAME / REPORT_FILENAME
    assert path.is_file()
    assert "Rough Milling" in path.read_text(encoding="utf-8")
    ((url),) = seen["opened"]
    assert Path(url.toLocalFile()) == path
    assert seen["toasts"] == [("success", f"Report written: {REPORT_FILENAME}")]


def test_an_older_experiment_says_why_there_is_no_v2_report(window, seen, tmp_path):
    window.autolamella_ui.experiment = _experiment(tmp_path, "older", recorded=False)

    window.action_generate_report_v2.trigger()

    assert seen["opened"] == []
    ((level, message),) = seen["toasts"]
    assert level == "warning" and "recorded before the event stream" in message


def test_with_no_experiment_open(window, seen):
    window.autolamella_ui.experiment = None

    window.action_generate_report_v2.trigger()

    assert seen["opened"] == []
    assert seen["toasts"] == [("warning", "Open an experiment to report on.")]

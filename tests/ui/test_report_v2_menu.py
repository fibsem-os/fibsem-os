"""Tools → Reporting → Generate Report v2 (preview) writes the page from the
experiment's events.jsonl, opens it, and prints it to a PDF (FIB-1036).

The real main window, with no microscope: the report reads records only. What
the browser is asked to open is caught, not opened, and the printing is a
stand-in (``tests/autolamella/test_pdf_export.py`` prints for real).
"""

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtGui import QDesktopServices  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402

from fibsem.applications.autolamella.event_recording import (  # noqa: E402
    EVENTS_FILENAME,
)
from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.tools import pdf_export  # noqa: E402
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
    """What the window opened, printed and said. Printing makes the PDF,
    unless ``seen["no_pdf"]`` says why it can't."""
    seen = {"opened": [], "printed": [], "toasts": [], "no_pdf": None}

    def print_to_pdf(path):
        seen["printed"].append(Path(path))
        if seen["no_pdf"]:
            raise pdf_export.PdfExportError(seen["no_pdf"])
        return Path(path).with_suffix(".pdf")

    monkeypatch.setattr(pdf_export, "html_to_pdf", print_to_pdf)
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


def _toasts(seen, count):
    """The toasts, once there are ``count``: the PDF's comes from a worker."""
    for _ in range(100):
        if len(seen["toasts"]) >= count:
            break
        QTest.qWait(20)
    return seen["toasts"]


def test_the_report_is_written_opened_and_printed(window, seen, tmp_path):
    experiment = _experiment(tmp_path, "recorded")
    window.autolamella_ui.experiment = experiment

    window.action_generate_report_v2.trigger()

    # written on a worker, then opened, then printed on another
    assert _toasts(seen, 2) == [
        ("success", f"Report written: {REPORT_FILENAME}"),
        ("success", "PDF written: report.pdf"),
    ]
    path = Path(experiment.path) / REPORT_DIRNAME / REPORT_FILENAME
    assert path.is_file()
    assert "Rough Milling" in path.read_text(encoding="utf-8")
    ((url),) = seen["opened"]
    assert Path(url.toLocalFile()) == path
    assert seen["printed"] == [path]


def test_with_no_pdf_the_page_still_is_and_says_how_to_make_one(window, seen, tmp_path):
    window.autolamella_ui.experiment = _experiment(tmp_path, "no-browser")
    seen["no_pdf"] = "no Edge, Chrome or Chromium was found to print the page with"

    window.action_generate_report_v2.trigger()

    assert _toasts(seen, 2)[1] == (
        "warning",
        "No PDF: no Edge, Chrome or Chromium was found to print the page with. "
        "The page's Print button makes one.",
    )
    assert len(seen["opened"]) == 1


def test_an_older_experiment_says_why_there_is_no_v2_report(window, seen, tmp_path):
    window.autolamella_ui.experiment = _experiment(tmp_path, "older", recorded=False)

    window.action_generate_report_v2.trigger()

    assert seen["opened"] == [] and seen["printed"] == []
    ((level, message),) = seen["toasts"]
    assert level == "warning" and "recorded before the event stream" in message


def test_with_no_experiment_open(window, seen):
    window.autolamella_ui.experiment = None

    window.action_generate_report_v2.trigger()

    assert seen["opened"] == [] and seen["printed"] == []
    assert seen["toasts"] == [("warning", "Open an experiment to report on.")]

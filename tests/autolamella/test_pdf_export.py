"""The report printed to a PDF by the browser already on the machine (FIB-1036).

A stand-in browser plays each way a real one can behave. It can write the PDF
and then not exit, as Chrome does on macOS, with a child process of its own. It
can exit without writing, never write, or write half a file. The last test
prints a real report with a real browser, and is skipped where there is none.
"""

import json
import os
import sys
import textwrap
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from fibsem.applications.autolamella.tools import pdf_export
from fibsem.applications.autolamella.tools.event_tables import event_tables
from fibsem.applications.autolamella.tools.pdf_export import (
    BROWSER_ENV,
    PdfExportError,
    find_browser,
    html_to_pdf,
)
from fibsem.applications.autolamella.tools.report_v2 import render_report

PDF = b"%PDF-1.4\n1 0 obj << >> endobj\ntrailer << >>\n%%EOF\n"

STAND_IN = textwrap.dedent(
    """
    import json, os, subprocess, sys, time
    mode, args = sys.argv[1], sys.argv[2:]
    out = next(a.split("=", 1)[1] for a in args if a.startswith("--print-to-pdf="))
    seen = {"args": args, "pid": os.getpid()}
    if mode == "write-and-stay":
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        seen["child"] = child.pid
    with open(os.environ["STAND_IN_SEEN"], "w") as f:
        json.dump(seen, f)
    if mode == "exit":
        sys.exit(3)
    if mode in ("write-and-stay", "write-and-exit"):
        with open(out, "wb") as f:
            f.write(%r)
        if mode == "write-and-exit":
            sys.exit(0)
    if mode == "half":
        with open(out, "wb") as f:
            f.write(b"%%PDF-1.4\\nhalf a page")
    time.sleep(60)
    """
    % PDF
)


@pytest.fixture
def page(tmp_path):
    path = tmp_path / "reporting" / "report.html"
    path.parent.mkdir()
    path.write_text("<!doctype html><title>t</title><p>page</p>", encoding="utf-8")
    return path


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """The stand-in browser, as a command for a mode; what it saw is read back
    with ``stand_in.seen()``."""
    script = tmp_path / "browser.py"
    script.write_text(STAND_IN, encoding="utf-8")
    seen = tmp_path / "seen.json"
    monkeypatch.setenv("STAND_IN_SEEN", str(seen))

    class StandIn:
        def __call__(self, mode):
            return [sys.executable, str(script), mode]

        def seen(self):
            return json.loads(seen.read_text())

    return StandIn()


def _gone(pid, within=5.0):
    """Whether process ``pid`` has stopped (POSIX; a zombie counts as gone)."""
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        try:
            if os.waitpid(pid, os.WNOHANG) != (0, 0):
                return True
        except ChildProcessError:
            pass  # not ours to reap: kill(0) says whether it runs
        time.sleep(0.1)
    return False


def test_the_pdf_is_taken_once_written_and_the_browser_stopped(page, stand_in):
    started = time.monotonic()

    pdf = html_to_pdf(page, browser=stand_in("write-and-stay"))

    assert time.monotonic() - started < 15, "waited on a browser that never exits"
    assert pdf == page.with_suffix(".pdf")
    assert pdf.read_bytes() == PDF
    assert not pdf.with_name(pdf.name + ".part").exists()
    seen = stand_in.seen()
    profile = next(a for a in seen["args"] if a.startswith("--user-data-dir="))
    assert not Path(profile.split("=", 1)[1]).exists(), "the profile was left behind"
    if os.name != "nt":
        # the browser, and what it started, are stopped
        assert _gone(seen["pid"]) and _gone(seen["child"])


def test_what_the_browser_is_asked(page, stand_in):
    html_to_pdf(page, browser=stand_in("write-and-exit"))

    args = stand_in.seen()["args"]
    assert args[-1] == page.resolve().as_uri()
    assert "--headless=new" in args and "--disable-background-networking" in args
    printed = next(a for a in args if a.startswith("--print-to-pdf="))
    assert printed.endswith("report.pdf.part")
    profile = next(a for a in args if a.startswith("--user-data-dir="))
    assert "fibsem-pdf-" in profile


def test_a_browser_that_exits_without_a_pdf(page, stand_in):
    with pytest.raises(PdfExportError, match="exited"):
        html_to_pdf(page, browser=stand_in("exit"))

    assert not page.with_suffix(".pdf").exists()


def test_a_browser_that_never_writes_is_stopped(page, stand_in):
    with pytest.raises(PdfExportError, match="within 1 s"):
        html_to_pdf(page, browser=stand_in("hang"), timeout=1)

    assert not page.with_suffix(".pdf").exists()
    if os.name != "nt":
        assert _gone(stand_in.seen()["pid"])


def test_half_a_pdf_is_not_taken(page, stand_in):
    with pytest.raises(PdfExportError):
        html_to_pdf(page, browser=stand_in("half"), timeout=1)

    pdf = page.with_suffix(".pdf")
    assert not pdf.exists() and not pdf.with_name(pdf.name + ".part").exists()


def test_a_named_browser_is_used_if_it_is_there(tmp_path, monkeypatch):
    browser = tmp_path / "facility-browser"
    browser.write_text("")
    monkeypatch.setenv(BROWSER_ENV, str(browser))
    assert find_browser() == str(browser)

    monkeypatch.setenv(BROWSER_ENV, str(tmp_path / "missing"))
    assert find_browser() is None


def test_with_no_browser_there_is_no_pdf(page, monkeypatch):
    monkeypatch.setattr(pdf_export, "find_browser", lambda: None)

    with pytest.raises(PdfExportError, match="no Edge, Chrome or Chromium"):
        html_to_pdf(page)


def test_a_real_browser_prints_the_report(tmp_path):
    browser = find_browser()
    if browser is None:
        pytest.skip("no Edge, Chrome or Chromium on this machine")
    t0 = datetime(2026, 9, 24, 9, 0)
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
        for kind, second in (("task_started", 0), ("task_completed", 900))
    ]
    html = tmp_path / "report.html"
    html.write_text(
        render_report(
            event_tables(records), "session", ["01-lamella"], ["Rough Milling"]
        ),
        encoding="utf-8",
    )

    pdf = html_to_pdf(html, timeout=120)

    data = pdf.read_bytes()
    assert data.startswith(b"%PDF") and b"%%EOF" in data[-1024:]

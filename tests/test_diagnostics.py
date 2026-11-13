"""Uncaught errors are logged, saved scrubbed, and offered on the next launch."""

import faulthandler
import json
import sys
import threading
from pathlib import Path

import pytest

from fibsem import diagnostics


@pytest.fixture
def fresh(tmp_path, monkeypatch):
    """diagnostics as a new process has it, saving under ``tmp_path``."""
    monkeypatch.setattr(diagnostics.cfg, "LOG_PATH", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_app", None)
    monkeypatch.setattr(diagnostics, "_on_error", None)
    monkeypatch.setattr(diagnostics, "_written", 0)
    monkeypatch.setattr(diagnostics, "_told", {})
    monkeypatch.setattr(diagnostics, "_fault_file", None)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    yield tmp_path
    faulthandler.disable()
    if diagnostics._fault_file is not None:
        diagnostics._fault_file.close()


def _raise_into_hook(error: Exception) -> None:
    try:
        raise error
    except Exception:
        sys.excepthook(*sys.exc_info())


def test_an_uncaught_error_is_saved_and_logged(fresh, caplog):
    diagnostics.install("test-app")

    _raise_into_hook(ValueError("the stage is elsewhere"))

    (record,) = diagnostics.records()
    assert record.app == "test-app"
    assert record.error == "ValueError"
    assert record.summary == "ValueError: the stage is elsewhere"
    assert "Traceback" in record.traceback
    assert not record.seen
    assert "Uncaught ValueError" in caplog.text


def test_the_saved_traceback_is_scrubbed(fresh):
    import os

    diagnostics.install("test-app")

    _raise_into_hook(FileNotFoundError(os.path.expanduser("~/data/lamella.tif")))

    (record,) = diagnostics.records()
    assert os.path.expanduser("~") not in record.traceback
    assert os.path.expanduser("~") not in Path(record.path).read_text()


def test_an_error_on_a_thread_is_saved_with_the_threads_name(fresh):
    # As outside pytest, whose own hook would report the thread's error.
    threading.excepthook = threading.__excepthook__
    diagnostics.install("test-app")

    def body():
        raise RuntimeError("off the main thread")

    thread = threading.Thread(target=body, name="acquire-worker")
    thread.start()
    thread.join()

    (record,) = diagnostics.records()
    assert record.thread == "acquire-worker"
    assert record.error == "RuntimeError"


def test_ctrl_c_is_not_an_error(fresh):
    seen = []
    sys.excepthook = lambda *args: seen.append(args[0])
    diagnostics.install("test-app")

    sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)

    assert diagnostics.records() == []
    assert seen == [KeyboardInterrupt]


def test_a_hook_that_was_already_there_still_hears(fresh):
    seen = []
    sys.excepthook = lambda exc_type, exc, tb: seen.append(exc_type)
    diagnostics.install("test-app")

    _raise_into_hook(ValueError("x"))

    assert seen == [ValueError]


def test_installing_twice_hooks_once_and_takes_the_later_listener(fresh):
    heard = []
    diagnostics.install("server")
    hook = sys.excepthook

    diagnostics.install("window", on_error=heard.append)

    assert sys.excepthook is hook
    _raise_into_hook(ValueError("x"))
    assert [record.app for record in heard] == ["window"]


def test_a_repeating_error_is_told_once_and_stops_being_saved(fresh):
    heard = []
    diagnostics.install("test-app", on_error=heard.append)

    def fail():
        raise ValueError("every tick")

    for _ in range(diagnostics.MAX_RECORDS_PER_PROCESS + 5):
        try:
            fail()
        except ValueError:
            sys.excepthook(*sys.exc_info())

    assert len(heard) == 1
    assert len(diagnostics.records()) == diagnostics.MAX_RECORDS_PER_PROCESS


def test_a_failing_listener_does_not_escape_the_hook(fresh):
    def listener(record):
        raise RuntimeError("the toast could not show")

    diagnostics.install("test-app", on_error=listener)

    _raise_into_hook(ValueError("x"))

    assert len(diagnostics.records()) == 1


def test_only_an_earlier_runs_unseen_records_are_offered(fresh):
    diagnostics.install("test-app")
    _raise_into_hook(ValueError("this run"))
    assert diagnostics.unseen_from_earlier_runs() == []

    (record,) = diagnostics.records()
    data = json.loads(Path(record.path).read_text())
    data["pid"] = record.pid + 1
    Path(record.path).write_text(json.dumps(data))

    (earlier,) = diagnostics.unseen_from_earlier_runs()
    diagnostics.mark_seen([earlier])
    assert diagnostics.unseen_from_earlier_runs() == []
    assert len(diagnostics.records()) == 1


def test_a_hard_crash_of_the_last_run_becomes_a_record(fresh):
    folder = diagnostics.crash_folder()
    folder.mkdir(parents=True)
    dump = "Fatal Python error: Segmentation fault\n\nCurrent thread 0x01:\n  File x"
    (folder / "faulthandler-test-app.log").write_text(dump)

    diagnostics.install("test-app")

    (record,) = diagnostics.unseen_from_earlier_runs()
    assert record.error == "fatal"
    assert "Segmentation fault" in record.traceback
    assert (folder / "faulthandler-test-app.log").read_text() == ""
    assert faulthandler.is_enabled()


def test_old_records_are_deleted_at_start(fresh, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_RECORDS_KEPT", 2)
    folder = diagnostics.crash_folder()
    folder.mkdir(parents=True)
    for n in range(4):
        (folder / f"2026-01-0{n + 1}-00-00-00-1-01.json").write_text("{}")

    diagnostics.install("test-app")

    assert len(list(folder.glob("*.json"))) == 2


def test_a_record_that_does_not_read_is_skipped(fresh):
    folder = diagnostics.crash_folder()
    folder.mkdir(parents=True)
    (folder / "broken.json").write_text("not json")

    assert diagnostics.records() == []

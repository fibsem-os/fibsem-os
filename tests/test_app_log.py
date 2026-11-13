"""The program's own log: written from start-up, kept when an experiment's
log is set up, and put in a bug report."""

import logging

import pytest

from fibsem import utils
from fibsem.util import app_log


@pytest.fixture
def log_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(app_log.cfg, "LOG_PATH", str(tmp_path))
    monkeypatch.setattr(app_log, "_handler", None)
    root = logging.getLogger()
    before, level = list(root.handlers), root.level
    yield tmp_path
    for handler in list(root.handlers):
        if handler not in before:
            root.removeHandler(handler)
            handler.close()
    root.handlers[:] = before
    root.setLevel(level)


def _text(path):
    for handler in logging.getLogger().handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


def test_it_is_written_before_any_experiment(log_folder):
    path = app_log.start("fibsem-test")
    logging.getLogger("fibsem.somewhere").info("connecting to the microscope")

    assert path == log_folder / "app" / "fibsem-test.log"
    assert "connecting to the microscope" in _text(path)


def test_starting_twice_keeps_one_handler(log_folder):
    first = app_log.start("fibsem-test")
    again = app_log.start("fibsem-other")

    assert again == first
    assert logging.getLogger().handlers.count(app_log.handler()) == 1


def test_it_goes_on_after_an_experiment_log_is_set_up(log_folder, tmp_path):
    path = app_log.start("fibsem-test")
    experiment = tmp_path / "experiment"
    experiment.mkdir()

    utils.configure_logging(path=experiment)
    logging.getLogger("fibsem.somewhere").info("after the experiment opened")

    assert "after the experiment opened" in _text(path)
    assert "after the experiment opened" in _text(experiment / "logfile.log")


def test_a_handler_already_there_hears_what_it_heard_before(log_folder):
    root = logging.getLogger()
    heard = []

    class Listener(logging.Handler):
        def emit(self, record):
            heard.append(record.getMessage())

    listener = Listener()
    root.addHandler(listener)
    root.setLevel(logging.WARNING)

    app_log.start("fibsem-test")
    logging.getLogger("fibsem.somewhere").info("quiet")
    logging.getLogger("fibsem.somewhere").warning("loud")

    assert heard == ["loud"]


def test_a_large_file_is_kept_as_the_older_one(log_folder, monkeypatch):
    monkeypatch.setattr(app_log, "MAX_BYTES", 10)
    path = app_log.path_for("fibsem-test")
    path.parent.mkdir(parents=True)
    path.write_text("the last run's lines\n", encoding="utf-8")

    app_log.start("fibsem-test")
    logging.getLogger("fibsem.somewhere").info("this run")

    assert "the last run" in (path.parent / "fibsem-test.1.log").read_text("utf-8")
    assert "the last run" not in _text(path)


def test_a_folder_that_cannot_be_made_does_not_stop_the_program(
    log_folder, monkeypatch
):
    blocker = log_folder / "app"
    blocker.write_text("a file where the folder should be")

    assert app_log.start("fibsem-test") is None
    assert app_log.handler() is None


def test_a_bug_report_includes_it(log_folder):
    from fibsem.tools import bug_report

    path = app_log.start("fibsem-test")
    logging.getLogger("fibsem.somewhere").info("before any experiment")
    _text(path)

    members = bug_report._files_from_this_pc(bug_report.BugReportContent())

    assert (str(path), "app-log/fibsem-test.log") in members

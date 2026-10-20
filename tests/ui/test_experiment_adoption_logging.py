"""Where the app points its logging when it takes on an experiment (FIB-421).

`Experiment.load` and `Experiment.create` no longer configure logging -- reading
an experiment must not reach into the calling process's global logging, and the
load dialog calls `load()` on every single click to preview a recent entry. The
app therefore has to do it at the one point it takes ownership, and this pins
that it still does. Without it the change is a silent regression: the workflow
would run with its log going wherever the app's log went at startup, and nobody
would notice until they went looking for an experiment's logfile.

Exercised through the session, and the window's unbound method against a stub,
so no microscope, no QApplication, and no widget tree are needed to check the
wiring.
"""

import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.structures import Experiment  # noqa: E402
from fibsem.applications.autolamella.ui.window import AutoLamellaWindow  # noqa: E402
from fibsem.workflows.session import EXPERIMENT, AppSession  # noqa: E402


class _StubUI:
    """The parts of the window that hearing a new experiment touches."""

    def __init__(self):
        self.calls = []

    def _setup_experiment_connections(self):
        self.calls.append("connections")

    def _on_experiment_update(self):
        self.calls.append("update")


@pytest.fixture
def quiet_root_logger():
    """Leave the root logger as we found it, whatever the test does to it."""
    root = logging.getLogger()
    saved = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if handler not in saved:
            handler.close()
    root.handlers = saved


def test_adopting_an_experiment_points_logging_at_its_logfile(
    tmp_path, quiet_root_logger
):
    experiment = Experiment.create(path=tmp_path, name="exp")
    AppSession().adopt(experiment)

    logfile = Path(experiment.path) / "logfile.log"
    assert logfile.is_file()
    assert str(logfile) in [
        getattr(h, "baseFilename", "") for h in logging.getLogger().handlers
    ]

    logging.info("after adoption")
    for handler in logging.getLogger().handlers:
        handler.flush()
    assert "after adoption" in logfile.read_text(encoding="utf-8")


def test_adopting_an_experiment_still_does_the_rest_of_the_wiring(
    tmp_path, quiet_root_logger
):
    """The logging call was added to an existing sequence, not in place of it:
    the session says so, and the window wires itself to it on hearing."""
    experiment = Experiment.create(path=tmp_path, name="exp")
    session = AppSession()
    heard = []
    session.changed.connect(heard.append)

    session.adopt(experiment)

    assert session.experiment is experiment
    assert heard == [EXPERIMENT]
    ui = _StubUI()
    AutoLamellaWindow._on_session_changed(ui, EXPERIMENT, None)
    assert ui.calls == ["connections", "update"]


def test_a_second_adoption_moves_logging_to_the_new_experiment(
    tmp_path, quiet_root_logger
):
    """Switching experiments mid-session has to follow, not accumulate."""
    first = Experiment.create(path=tmp_path / "a", name="exp")
    second = Experiment.create(path=tmp_path / "b", name="exp")
    session = AppSession()

    session.adopt(first)
    session.adopt(second)

    logging.info("belongs to the second experiment")
    for handler in logging.getLogger().handlers:
        handler.flush()

    assert "belongs to the second experiment" in (
        Path(second.path) / "logfile.log"
    ).read_text(encoding="utf-8")
    assert "belongs to the second experiment" not in (
        Path(first.path) / "logfile.log"
    ).read_text(encoding="utf-8")


def test_previewing_an_experiment_leaves_logging_alone(tmp_path, quiet_root_logger):
    """The load dialog calls Experiment.load on every single click to preview a
    recent entry. That must not repoint the app's logging at whatever the user
    happens to be browsing -- the bug this issue is about."""
    experiment = Experiment.create(path=tmp_path, name="exp")
    AppSession().adopt(experiment)

    other = Experiment.create(path=tmp_path / "elsewhere", name="other")
    before = [getattr(h, "baseFilename", "") for h in logging.getLogger().handlers]

    Experiment.load(os.path.join(other.path, "experiment.yaml"))

    assert [
        getattr(h, "baseFilename", "") for h in logging.getLogger().handlers
    ] == before

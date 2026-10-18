"""The running application without a window: microscope, experiment, events.

A window is one client of an ``AppSession``; these drive it with none.
"""

import os
from pathlib import Path

import pytest

from fibsem import utils
from fibsem.applications.autolamella.structures import Experiment
from fibsem.workflows.event_recording import EVENTS_FILENAME, read_events, recorder_for
from fibsem.workflows.session import AppSession


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    microscope.disconnect()


@pytest.fixture
def experiment(tmp_path):
    experiment = Experiment(path=tmp_path, name="session")
    os.makedirs(experiment.path, exist_ok=True)
    return experiment


def _recorded(session, experiment):
    """Mark the stream, close it, and read back what reached the experiment."""
    session.microscope.record_event("task_step", {"step": "MARK"})
    session.stop_recording()
    return [
        e
        for e in read_events(Path(experiment.path) / EVENTS_FILENAME)
        if e.get("payload", {}).get("step") == "MARK"
    ]


@pytest.fixture
def session():
    session = AppSession()
    yield session
    session.stop_recording()


def test_no_microscope_records_nothing(session):
    session.start_recording()

    assert session.recorder is None


def test_a_connection_is_recorded_to_the_open_experiment(
    session, microscope, experiment
):
    session.microscope = microscope
    session.adopt(experiment)

    session.start_recording()

    assert recorder_for(microscope) is session.recorder
    assert _recorded(session, experiment)


def test_asking_again_keeps_the_stream(session, microscope):
    session.microscope = microscope
    session.start_recording()
    first = session.recorder

    session.start_recording()

    assert session.recorder is first


def test_adopting_an_experiment_moves_the_stream(session, microscope, experiment):
    session.microscope = microscope
    session.start_recording()

    session.adopt(experiment)

    assert _recorded(session, experiment)


def test_stopping_closes_the_stream(session, microscope):
    session.microscope = microscope
    session.start_recording()

    session.stop_recording()

    assert session.recorder is None
    assert recorder_for(microscope) is None

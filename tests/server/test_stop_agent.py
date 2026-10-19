"""Stop agent stops what the agent is doing on the microscope (FIB-1267), and
nothing anyone else is doing."""

import os

import pytest

from fibsem import utils
from fibsem.acting import AGENT, SCRIPT, WORKFLOW
from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.hosting import AppServerHost

pytest.importorskip("fastapi")


@pytest.fixture
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    microscope.disconnect()


@pytest.fixture
def stops(microscope, monkeypatch):
    """What got stopped, in order, instead of stopping it."""
    seen = []
    monkeypatch.setattr(
        microscope, "stop_acquisition", lambda: seen.append("acquisition")
    )
    # The milling service's stop, which ends a milling task too (FIB-1398).
    for name in ("milling", "spot_burn", "imaging"):
        service = getattr(microscope, name, None)
        if service is not None:
            monkeypatch.setattr(service, "stop", lambda n=name: seen.append(n))
    return seen


@pytest.fixture
def host():
    from fibsem.server import AuthConfig

    host = AppServerHost(session=None, app_context=None)
    host.auth = AuthConfig.generate()
    host.auth.set_level(AgentLevel.DRIVE)
    return host


def test_the_agent_in_control_is_stopped(host, microscope, stops):
    with microscope.control.take(AGENT, "POST /run_milling"):
        stopped = host.stop_agent(microscope)
    assert stopped == "POST /run_milling"
    assert "milling" in stops and "acquisition" in stops
    assert host.auth.level is AgentLevel.WATCH and host.auth.stopped


def test_what_the_agent_started_and_left_running_is_stopped(host, microscope, stops):
    # start_milling returns, and lets go of control, while the mill runs on.
    with microscope.control.take(AGENT, "POST /start_milling"):
        pass
    assert host.stop_agent(microscope) == "POST /start_milling"
    assert "milling" in stops


@pytest.mark.parametrize("by", [WORKFLOW, SCRIPT])
def test_a_run_or_script_in_control_is_left_alone(host, microscope, stops, by):
    with microscope.control.take(AGENT, "POST /move_stage_relative"):
        pass
    with microscope.control.take(by, "a lamella run"):
        assert host.stop_agent(microscope) is None
    assert stops == []
    assert host.auth.level is AgentLevel.WATCH


def test_nothing_milling_is_no_failure(host, microscope):
    """The milling service refuses a stop with nothing milling (FIB-1398); Stop
    agent lets that be, and stops the rest."""
    with microscope.control.take(AGENT, "POST /run_milling"):
        assert host.stop_agent(microscope) == "POST /run_milling"


def test_the_agent_borrowing_during_a_question_is_stopped(host, microscope, stops):
    control = microscope.control
    with control.take(WORKFLOW, "a lamella run"), control.lend():
        with control.take(AGENT, "POST /run_milling"):
            assert host.stop_agent(microscope) == "POST /run_milling"
    assert "milling" in stops


def test_an_agent_that_did_nothing_stops_nothing(host, microscope, stops):
    assert host.stop_agent(microscope) is None
    assert stops == []
    assert host.auth.stopped


def test_one_stop_failing_does_not_keep_the_others_from_running(
    host, microscope, stops, monkeypatch
):
    def _fails():
        raise RuntimeError("the instrument said no")

    monkeypatch.setattr(microscope, "stop_milling", _fails)
    with microscope.control.take(AGENT, "POST /run_milling"):
        host.stop_agent(microscope)
    assert "acquisition" in stops

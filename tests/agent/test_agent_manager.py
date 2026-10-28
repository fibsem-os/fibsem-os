"""The agent manager against a stand-in ``fibsem-agent`` (``stand_in.py``):
real processes and real session folders, no model."""

import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from fibsem.ai.agent import manager
from fibsem.ai.agent.manager import (
    MANAGED,
    RECORD,
    AgentRunning,
    AgentSessionManager,
    SessionFolder,
    read_json,
)

ROOT = Path(__file__).resolve().parents[2]
STAND_IN = Path(__file__).with_name("stand_in.py")


class Clock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def make(tmp_path):
    """A manager over *tmp_path*'s sessions, whose agents follow *plan*."""
    made = []

    def make(plan="wait", reachable=lambda microscope: True, **kwargs):
        env = {
            "STAND_IN_ROOT": str(tmp_path),
            "STAND_IN_PLAN": plan,
            "PYTHONPATH": str(ROOT),
        }
        kwargs.setdefault("env_for", lambda microscope: env)
        made.append(
            AgentSessionManager(
                root=tmp_path,
                command=[sys.executable, str(STAND_IN)],
                reachable=reachable,
                end_wait_s=kwargs.pop("end_wait_s", 5.0),
                **kwargs,
            )
        )
        return made[-1], env

    yield make
    for each in made:
        for agent in list(each._agents.values()):
            if agent.process is not None and agent.process.poll() is None:
                agent.process.kill()
                agent.process.wait(10)


def until(test, manager_=None, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if manager_ is not None:
            manager_.poll()
        if test():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


def start(mgr, env, microscope="arctis-1", **kwargs):
    return mgr.start(
        microscope, str(uuid.uuid4()), ["--files", "/w"], env=env, **kwargs
    )


def runs(agent):
    return read_json(agent.folder / RECORD).get("runs", [])


def test_a_new_session_has_its_own_folder_with_its_record(make):
    mgr, env = make("0")
    agent = start(mgr, env, goal="mill 01")
    until(lambda: agent.state == "ended", mgr)

    assert agent.folder.name.endswith(agent.session_id)
    record = read_json(agent.folder / RECORD)
    assert record["session_id"] == agent.session_id
    assert record["goal"] == "mill 01"
    assert record["args"] == ["--files", "/w"]
    assert runs(agent)[0]["exit_code"] == 0
    managed = read_json(agent.folder / MANAGED)
    assert managed["microscope"] == "arctis-1"
    assert managed["args"] == ["--files", "/w"]
    assert "STAND_IN_PLAN" not in (agent.folder / MANAGED).read_text(), (
        "the environment (the token) is never written down"
    )


@pytest.mark.parametrize(
    "plan, why",
    [("0", "stopped"), ("2", "its budget is spent"), ("fatal", "can't answer")],
)
def test_an_agent_that_ends_for_good_is_not_restarted(make, plan, why):
    mgr, env = make(plan)
    agent = start(mgr, env)
    until(lambda: agent.state == "ended", mgr)
    assert why in agent.ended
    assert len(runs(agent)) == 1


def test_an_agent_whose_app_went_away_resumes_the_same_session_once_it_answers(make):
    answers = {"arctis-1": False}
    mgr, env = make("1,wait", reachable=lambda microscope: answers[microscope])
    agent = start(mgr, env, goal="mill 01")
    until(lambda: agent.state == "waiting", mgr)
    mgr.poll()
    assert agent.state == "waiting", "not while the app doesn't answer"

    answers["arctis-1"] = True
    until(lambda: len(runs(agent)) == 2, mgr)
    record = read_json(agent.folder / RECORD)
    assert record["args"] == ["--files", "/w"], "the same flags again"
    assert record["goal"] == "mill 01", "a resume keeps the first run's goal"
    assert manager.folders(agent.folder.parent) == [agent.folder], "one folder"
    assert agent.state == "running"


def test_restarts_after_the_app_went_away_are_three_an_hour(make):
    clock = Clock()
    mgr, env = make("1,1,1,1,wait", clock=clock)
    agent = start(mgr, env)
    until(lambda: agent.state == "ended", mgr)
    assert agent.ended == "restarts used up"
    assert len(runs(agent)) == 4


def test_a_restart_budget_frees_up_after_an_hour(tmp_path):
    clock = Clock()
    mgr = AgentSessionManager(root=tmp_path, clock=clock)
    agent = manager.Agent("arctis-1", "abc", tmp_path / "x")
    old = clock.now - manager.HOUR_S - 1
    manager.write_json(
        agent.folder / MANAGED,
        {
            "restarts": [{"at": old, "kind": "app"}] * 3
            + [{"at": clock.now, "kind": "crash"}]
        },
    )
    assert mgr._may_restart(agent, "app"), "three, but over an hour ago"
    assert not mgr._may_restart(agent, "crash")


def test_a_crash_is_resumed_once_an_hour(make):
    mgr, env = make("crash,crash,wait")
    agent = start(mgr, env)
    until(lambda: agent.state == "ended", mgr)
    assert agent.ended == "restarts used up"
    assert len(runs(agent)) == 2
    assert "exit_code" not in runs(agent)[1], "a crash records nothing"


def test_one_agent_per_microscope(make):
    mgr, env = make("wait")
    start(mgr, env)
    with pytest.raises(AgentRunning):
        start(mgr, env)
    other = start(mgr, env, microscope="arctis-2")
    assert other.state == "running"


def test_end_stops_the_agent_with_its_stop_file(make):
    mgr, env = make("wait")
    agent = start(mgr, env)
    until(lambda: len(runs(agent)) == 1)
    mgr.end("arctis-1")
    assert agent.state == "ended"
    assert agent.ended == "ended by the operator"
    assert runs(agent)[0]["exit_code"] == 0, "it stopped itself"
    assert start(mgr, env).state == "running", "the microscope is free again"


def test_an_agent_that_does_not_stop_is_killed(make):
    mgr, env = make("ignore", end_wait_s=0.5)
    agent = start(mgr, env)
    until(lambda: len(runs(agent)) == 1)
    mgr.end("arctis-1")
    assert agent.ended == "killed: it didn't stop"
    assert agent.process.poll() is not None


def test_agents_outlive_their_manager_and_a_new_one_takes_them_back(make):
    first, env = make("wait")
    agent = start(first, env)
    until(lambda: len(runs(agent)) == 1)

    second, _ = make("wait")
    taken = second.agent("arctis-1")
    assert taken is not None and taken.session_id == agent.session_id
    second.poll()
    assert taken.state == "running"
    # Here the first manager is this process: reap its child as the system
    # would once it has gone.
    threading.Thread(target=agent.process.wait, daemon=True).start()
    second.end("arctis-1")
    assert taken.ended == "ended by the operator"
    until(lambda: agent.process.poll() is not None)


def test_a_new_manager_resumes_an_agent_that_lost_its_app_while_none_ran(make):
    first, env = make("1,wait")
    agent = start(first, env)
    until(lambda: agent.process.poll() is not None)

    second, _ = make("1,wait")
    taken = second.agent("arctis-1")
    until(lambda: len(runs(taken)) == 2, second)
    assert taken.state == "running"


def test_an_ended_session_is_not_taken_back(make):
    first, env = make("0")
    agent = start(first, env)
    until(lambda: agent.state == "ended", first)
    second, _ = make()
    assert second.agent("arctis-1") is None


def test_sessions_lists_every_session_with_the_managers_part(make):
    mgr, env = make("0")
    one = start(mgr, env)
    until(lambda: one.state == "ended", mgr)
    two = start(mgr, env)
    until(lambda: two.state == "ended", mgr)
    rows = mgr.sessions()
    assert {row["session_id"] for row in rows} == {one.session_id, two.session_id}
    assert all(row["manager"]["microscope"] == "arctis-1" for row in rows)


def test_the_last_session_is_the_newest_run_in_the_same_folder(tmp_path):
    work, other = tmp_path / "work", tmp_path / "other"
    assert manager.last_session(work, tmp_path) is None
    for session_id, cwd, started in [
        ("first", work, 1000.0),
        ("elsewhere", other, 2000.0),
        ("second", work, 3000.0),
    ]:
        folder = tmp_path / manager.folder_name(session_id, started)
        manager.write_json(folder / RECORD, {"session_id": session_id, "cwd": str(cwd)})
    assert manager.last_session(work, tmp_path) == "second"
    assert manager.last_session(other, tmp_path) == "elsewhere"


def test_a_session_folder_records_each_run_and_how_it_ended(tmp_path):
    session = SessionFolder.open("abc", {"cwd": "/w", "goal": "mill 01"}, tmp_path)
    session.end(
        1, "the app could not be watched", spent_usd=0.5, turns=3, tokens={"output": 9}
    )
    (session.folder / manager.STOP).touch()
    again = SessionFolder.open("abc", {"cwd": "/w", "goal": None}, tmp_path)

    assert again.folder == session.folder
    assert not again.stop_requested(), "an old stop is not for this run"
    record = again.record()
    assert record["goal"] == "mill 01"
    first, second = record["runs"]
    assert first["exit_code"] == 1 and first["spent_usd"] == 0.5 and first["turns"] == 3
    assert first["tokens"] == {"output": 9}
    assert "ended" not in second


def test_a_bare_resume_finds_the_last_session_that_worked_in_its_own_folder(tmp_path):
    own = manager.session_folder("own", tmp_path)
    SessionFolder.open("own", {"cwd": str(own / manager.WORK)}, tmp_path, own)
    work = tmp_path / "w"
    later = SessionFolder.open("later", {"cwd": str(work)}, tmp_path)
    assert manager.session_folder("own", tmp_path) == own
    assert later.folder.parent == tmp_path
    assert manager.last_session(None, tmp_path) == "own"
    assert manager.last_session(work, tmp_path) == "later"

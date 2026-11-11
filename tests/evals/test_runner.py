"""The eval runner end to end, with stand-ins for the App Server and the agent."""

import json
import os
import sys
from pathlib import Path

import pytest
import yaml

from fibsem.ai.evals import report
from fibsem.ai.evals.__main__ import main
from fibsem.ai.evals.cases import load_case
from fibsem.ai.evals.runner import (
    RunConfig,
    RunnerError,
    link_inputs,
    run_case,
    run_cases,
    task_text,
)
from fibsem.server import discovery

FAKES = str(Path(__file__).with_name("fakes.py"))


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("FIBSEM_HOME", raising=False)
    monkeypatch.setattr(
        discovery, "DISCOVERY_FILE", home / ".fibsem" / "app-server.json"
    )
    monkeypatch.setattr(
        discovery, "LEGACY_DISCOVERY_FILE", home / ".fibsem" / "old.json"
    )
    return home


@pytest.fixture
def case(tmp_path):
    folder = tmp_path / "cases" / "polish"
    (folder / "experiment" / "iter").mkdir(parents=True)
    for name in ("p1", "p2", "p3"):
        (folder / "experiment" / "iter" / f"{name}.tif").write_bytes(b"x")
    (folder / "reference").mkdir()
    (folder / "reference" / "what-the-run-did.json").write_text("{}")
    data = {
        "id": "polish",
        "task": "polishing_stop",
        "format": 0,
        "brief": "Say carry on or stop after each pass.",
        "sample": "test cells",
        "passes": [
            {"name": n, "fib": f"experiment/iter/{n}.tif", "measured": {"median_nm": m}}
            for n, m in (("p1", 400), ("p2", 210), ("p3", 190))
        ],
        "key": {"stop_after": "p3", "reason": "secret reason"},
    }
    (folder / "case.yaml").write_text(yaml.safe_dump(data))
    return load_case(folder)


def config(tmp_path, **kwargs):
    return RunConfig(
        out=tmp_path / "out",
        repeats=1,
        server_command=[sys.executable, FAKES, "server", "--manufacturer", "Demo"],
        agent_command=[sys.executable, FAKES, "agent"],
        server_wait_s=20,
        **kwargs,
    )


STOP_AT_P3 = {
    "passes": [
        {"name": "p1", "decision": "carry_on"},
        {"name": "p2", "decision": "carry_on"},
        {"name": "p3", "decision": "stop"},
    ]
}


def test_run_scores_the_answer(home, case, tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_FAKE_ANSWER", json.dumps(STOP_AT_P3))
    result = run_case(case, 1, config(tmp_path, timeout_s=20))
    assert result["status"] == "answered"
    assert [s["result"] for s in result["scores"]] == ["hit"]
    assert result["spent_usd"] == 0.25

    folder = tmp_path / "out" / "polish" / "r1"
    task = (folder / "work" / "task.md").read_text()
    assert "iter/p2.tif" in task and "median_nm" in task and "passes" in task
    assert "secret reason" not in task and "stop_after" not in task
    experiment = folder / "home" / "experiments" / "polish"
    assert (experiment / "iter" / "p1.tif").read_bytes() == b"x"
    assert not (experiment / "what-the-run-did.json").exists()
    assert json.loads((folder / "result.json").read_text())["case"] == "polish"
    assert (folder / "session" / "session.json").is_file()
    # Each run starts from its own state root, which lets the agent connect.
    prefs = yaml.safe_load(
        (folder / "home" / "config" / "user-preferences.yaml").read_text()
    )
    assert prefs["install"]["app_server"]["allow_local_agents"] is True
    assert discovery.read_discovery_file() is None  # the server was ended

    (row,) = report.summarise(report.load_results(tmp_path / "out"))
    assert (row["runs"], row["hit"], row["all_repeats_right"]) == (1, 1, 1)
    assert "| polishing_stop | yes | 1 | 1 | 0 | 0 | 1 of 1 |" in report.table([row])


def test_run_stops_an_agent_that_never_answers(home, case, tmp_path, monkeypatch):
    monkeypatch.delenv("EVAL_FAKE_ANSWER", raising=False)
    result = run_case(case, 1, config(tmp_path, timeout_s=1))
    assert result["status"] == "timeout"
    assert result["agent_exit"] == 0  # stopped through its stop file
    assert [s["result"] for s in result["scores"]] == ["invalid"]


def test_refuses_while_another_app_server_runs(home, case, tmp_path):
    discovery.DISCOVERY_FILE.parent.mkdir(parents=True)
    discovery.DISCOVERY_FILE.write_text(
        json.dumps(
            {"url": "http://127.0.0.1:1", "connect_token": "x", "pid": os.getpid()}
        )
    )
    with pytest.raises(RunnerError, match="already running"):
        run_case(case, 1, config(tmp_path))


def test_repeats_and_skips_unkeyed(home, case, tmp_path, monkeypatch):
    monkeypatch.setenv("EVAL_FAKE_ANSWER", json.dumps(STOP_AT_P3))
    case.data["key"] = {"status": "TODO"}
    assert run_cases([case], config(tmp_path, timeout_s=20)) == []
    case.data["key"] = {"stop_after": "p2"}
    cfg = config(tmp_path, timeout_s=20)
    cfg.repeats = 2
    results = run_cases([case], cfg)
    assert [(r["repeat"], r["scores"][0]["result"]) for r in results] == [
        (1, "unsafe"),
        (2, "unsafe"),
    ]


def test_cli_run_and_report(home, case, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("EVAL_FAKE_ANSWER", json.dumps(STOP_AT_P3))
    monkeypatch.setattr(
        "fibsem.ai.evals.runner.server_command",
        lambda: [sys.executable, FAKES, "server"],
    )
    monkeypatch.setattr(
        "fibsem.ai.evals.runner.agent_command",
        lambda: [sys.executable, FAKES, "agent"],
    )
    out = tmp_path / "out"
    code = main(
        [
            "run",
            str(case.path.parent),
            "--out",
            str(out),
            "--repeats",
            "1",
        ]
    )
    assert code == 0
    assert "polish r1: answered, hit" in capsys.readouterr().out
    assert (out / "report.md").is_file()
    assert main(["report", str(out)]) == 0


def test_a_file_shared_from_another_case_is_linked_in(case, tmp_path):
    shared = case.path.parent / "fm" / "experiment" / "stack.ome.tiff"
    shared.parent.mkdir(parents=True)
    shared.write_bytes(b"stack")
    case.data["inputs"] = {"fm_stack": "../fm/experiment/stack.ome.tiff"}
    experiment = tmp_path / "exp"
    placed = link_inputs(case, experiment)
    assert "stack.ome.tiff" in placed and "iter/p1.tif" in placed
    assert (experiment / "stack.ome.tiff").read_bytes() == b"stack"
    assert "- stack.ome.tiff" in task_text(case, placed)

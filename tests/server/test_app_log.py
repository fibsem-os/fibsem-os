"""The session log over the App Server (FIB-905): windows, levels, caps, and the
route and tool that serve it."""

import asyncio
import json
import os
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from fibsem.applications.autolamella.server import log_tail

T0 = datetime(2026, 10, 12, 9, 0, 0)


def _line(time, level, message, name="root", where="run:10"):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S") + f",{time.microsecond // 1000:03d}"
    return f"{stamp} — {name} — {level} — {where} — {message}\n"


def _write_log(path, lines):
    path.write_bytes("".join(lines).encode("utf-8"))
    return path


def _a_long_log(path, count=20000):
    """One record a second from T0; every 1000th an ERROR with a traceback."""
    lines = []
    for i in range(count):
        time = T0 + timedelta(seconds=i)
        if i % 1000 == 999:
            lines.append(_line(time, "ERROR", f"failed {i}"))
            lines.append("Traceback (most recent call last):\n")
            lines.append(f'  File "x.py", line {i}, in run\n')
        else:
            lines.append(_line(time, "INFO", f"step {i} " + "x" * 80))
    return _write_log(path, lines)


def test_the_tail_is_the_latest_records(tmp_path):
    log = _a_long_log(tmp_path / "logfile.log")
    out = log_tail.read_log(log, lines=3)
    assert out["records"][-1]["time"] == (T0 + timedelta(seconds=19999)).isoformat()
    assert len(out["records"]) == 3
    assert out["truncated"] is True


def test_the_tail_reads_only_the_end_of_a_long_log(tmp_path, monkeypatch):
    monkeypatch.setattr(log_tail, "TAIL_BYTES", 50_000)
    log = _a_long_log(tmp_path / "logfile.log")
    out = log_tail.read_log(log, lines=2000)
    # about 500 records fit in 50 kB; the read says where it started
    assert 300 < len(out["records"]) < 700
    assert out["read_from"] == out["records"][0]["time"]
    assert out["truncated"] is False


def test_a_window_deep_in_the_file(tmp_path):
    log = _a_long_log(tmp_path / "logfile.log")
    since, until = T0 + timedelta(seconds=5000), T0 + timedelta(seconds=5009)
    out = log_tail.read_log(log, since=since, until=until, lines=2000)
    times = [r["time"] for r in out["records"]]
    assert times[0] == since.isoformat()
    assert times[-1] == until.isoformat()
    assert len(times) == 10


def test_level_keeps_errors_with_their_tracebacks(tmp_path):
    log = _a_long_log(tmp_path / "logfile.log")
    out = log_tail.read_log(
        log,
        since=T0 + timedelta(seconds=2990),
        until=T0 + timedelta(seconds=3010),
        level="WARNING",
    )
    assert len(out["records"]) == 1
    record = out["records"][0]
    assert record["level"] == "ERROR"
    assert record["text"].splitlines() == [
        _line(T0 + timedelta(seconds=2999), "ERROR", "failed 2999").rstrip("\n"),
        "Traceback (most recent call last):",
        '  File "x.py", line 2999, in run',
    ]


def test_a_cp1252_line_still_parses(tmp_path):
    log = tmp_path / "logfile.log"
    log.write_bytes(
        _line(T0, "INFO", "utf-8").encode("utf-8")
        + _line(T0 + timedelta(seconds=1), "WARNING", "old install").encode("cp1252")
    )
    out = log_tail.read_log(log, level="WARNING")
    assert [r["level"] for r in out["records"]] == ["WARNING"]


def test_the_answer_is_capped_in_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(log_tail, "MAX_TEXT_BYTES", 2000)
    monkeypatch.setattr(log_tail, "MAX_RECORD_CHARS", 500)
    log = _write_log(
        tmp_path / "logfile.log",
        [_line(T0 + timedelta(seconds=i), "INFO", "y" * 1000) for i in range(10)],
    )
    out = log_tail.read_log(log, lines=10)
    assert out["truncated"] is True
    assert sum(len(r["text"]) for r in out["records"]) <= 2000
    assert out["records"][-1]["text"].endswith("…[cut]")
    # the latest are kept
    assert out["records"][-1]["time"] == (T0 + timedelta(seconds=9)).isoformat()


def _write_events(folder, records):
    with open(folder / "events.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _event(kind, seconds, task_id="run-1"):
    time = T0 + timedelta(seconds=seconds)
    return {"t": time.isoformat() + "+10:00", "kind": kind, "task": {"id": task_id}}


def test_a_runs_window_comes_from_the_event_record(tmp_path):
    _write_events(
        tmp_path,
        [
            _event("task_started", 10),
            _event("task_started", 12, task_id="other"),
            _event("task_failed", 40),
        ],
    )
    start, end = log_tail.run_window(tmp_path, "run-1")
    # wall clock: the offset is dropped, not converted
    assert start == T0 + timedelta(seconds=10) - log_tail.RUN_MARGIN
    assert end == T0 + timedelta(seconds=40) + log_tail.RUN_MARGIN
    assert log_tail.run_window(tmp_path, "missing") is None


def test_a_run_still_going_has_an_open_end(tmp_path):
    _write_events(tmp_path, [_event("task_started", 10)])
    assert log_tail.run_window(tmp_path, "run-1")[1] is None


# --- over HTTP and MCP ---------------------------------------------------------

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


@pytest.fixture(scope="module")
def microscope():
    # The route tests need the [server] extra; the reader tests above do not.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fibsem import utils

    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def host(tmp_path):
    host = Host()
    # the log route reads only the experiment's folder
    host.experiment = SimpleNamespace(path=tmp_path)
    _a_long_log(tmp_path / "logfile.log", count=3000)
    _write_events(tmp_path, [_event("task_started", 2995), _event("task_failed", 2999)])
    return host


@pytest.fixture
def client(microscope, host):
    from fastapi.testclient import TestClient

    from fibsem.applications.autolamella.server import AutoLamellaAppContext
    from fibsem.server import AuthConfig, build_server

    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_a_failed_runs_errors_over_http(client):
    body = client.get(
        "/app/log", params={"task_id": "run-1", "level": "warning"}, headers=AUTH
    ).json()
    assert body["available"] is True
    assert [r["text"].splitlines()[0][-11:] for r in body["records"]] == ["failed 2999"]
    assert body["since"] == (T0 + timedelta(seconds=2990)).isoformat()


def test_a_runs_log_defaults_to_info(client, host):
    # FIB-1330: at DEBUG a run's window is mostly noise, and the cap keeps
    # its end. Given a level, the agent gets what it asked for.
    log = host.experiment.path / "logfile.log"
    with open(log, "a", encoding="utf-8") as f:
        f.write(_line(T0 + timedelta(seconds=2996), "DEBUG", "noise"))
    get = lambda **q: client.get("/app/log", params=q, headers=AUTH).json()  # noqa: E731

    body = get(task_id="run-1")
    assert body["level"] == "INFO"
    assert not [r for r in body["records"] if r["level"] == "DEBUG"]
    debug = get(task_id="run-1", level="debug")
    assert debug["level"] == "DEBUG"
    assert [r for r in debug["records"] if r["level"] == "DEBUG"]


def test_refusals_over_http(client):
    get = lambda **q: client.get("/app/log", params=q, headers=AUTH)  # noqa: E731
    assert get(task_id="missing").status_code == 404
    assert get(level="LOUD").json()["detail"]["levels"][0] == "DEBUG"
    assert get(since="yesterday").status_code == 422
    assert get(task_id="run-1", since=T0.isoformat()).status_code == 422
    assert client.get("/app/log").status_code == 401


def test_no_experiment_is_unavailable(client, host):
    host.experiment = None
    assert client.get("/app/log", headers=AUTH).json() == {
        "available": False,
        "records": [],
    }


def test_the_tool_passes_the_window(client):
    pytest.importorskip("mcp")
    from fibsem.mcp.sidecar import build_sidecar

    client.headers["Authorization"] = f"Bearer {TOKEN}"
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    since = (T0 + timedelta(seconds=100)).isoformat() + "+10:00"
    result = asyncio.run(sidecar.call_tool("get_log", {"since": since, "lines": 2}))
    if isinstance(result, tuple):
        result = result[0]
    text = "".join(c.text for c in getattr(result, "content", result))
    body = json.loads(text)
    assert body["since"] == (T0 + timedelta(seconds=100)).isoformat()
    assert len(body["records"]) == 2

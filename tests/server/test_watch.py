"""``fibsem-mcp watch`` against a real App Server on a real socket: the urllib
client, the sidecar's reconnecting client and the /app/events long-poll, with
nothing faked but the moment events arrive (FIB-1269, FIB-870)."""

import os
import subprocess
import sys
import threading
import time

import pytest

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.mcp import watch as watch_mod  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.discovery import write_discovery_file  # noqa: E402
from fibsem.server.levels import AgentLevel  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from fibsem.workflows.events import EventBuffer  # noqa: E402

TOKEN = "test-token"
POLL_S = 0.3


class Host:
    experiment = None
    microscope = None
    _task_manager = None
    is_workflow_running = False


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def discovery(tmp_path, monkeypatch):
    """Discovery files in tmp, and no server named by the environment."""
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_SERVER_TOKEN", raising=False)
    path = tmp_path / "app-server.json"
    monkeypatch.setattr("fibsem.server.discovery.DISCOVERY_FILE", path)
    monkeypatch.setattr(
        "fibsem.server.discovery.LEGACY_DISCOVERY_FILE", tmp_path / "legacy.json"
    )
    return path


class App:
    """A running App Server: its auth, its app context and its port."""

    def __init__(self, auth, context, port):
        self.auth = auth
        self.context = context
        self.port = port

    @property
    def buffer(self):
        return self.context._event_buffer

    def set_running(self, running):
        self.context._host.is_workflow_running = running


def _serve(microscope, tmp_path, event_buffer):
    host = Host()
    exp = Experiment(path=tmp_path / "exp", name="watch-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    auth = AuthConfig(token=TOKEN)
    context = AutoLamellaAppContext(host, event_buffer=event_buffer)
    app = build_server(microscope, app_context=context, auth=auth)
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        assert thread.is_alive(), "App Server failed to start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, thread, App(auth, context, port)


@pytest.fixture
def app(microscope, tmp_path, discovery):
    server, thread, app = _serve(microscope, tmp_path, EventBuffer())
    write_discovery_file("127.0.0.1", app.port, app.auth, path=discovery)
    yield app
    server.should_exit = True
    thread.join(timeout=5)


class _Steered(watch_mod._UrllibClient):
    """The real client, with a test's hand on what happens before poll N."""

    steps = {}
    requests = 0

    def request(self, method, path, json=None, timeout=10.0):
        type(self).requests += 1
        step = self.steps.get(type(self).requests)
        if step is not None:
            step()
        return super().request(method, path, json=json, timeout=timeout)


class _Seen(Exception):
    """The test has seen what it waited for; the watch itself never stops on it."""


def _ended(lines):
    return lines[-1].startswith(watch_mod.ENDED)


def _watch(steps, sleeps=None, since=None, until=_ended):
    """Watch the discovered server, running ``steps[n]`` before request n, until
    ``until(lines)`` (by default, a workflow's end).

    The watch runs for the whole session (FIB-1431), so the test stops it.
    ``code`` is None when it did, or what the watch returned when it gave up."""
    client_type = type("Steered", (_Steered,), {"steps": steps, "requests": 0})
    client = watch_mod.connect(client_factory=client_type)
    assert client is not None
    lines = []

    def out(line):
        lines.append(line)
        if until(lines):
            raise _Seen

    sleep_fn = sleeps.append if sleeps is not None else (lambda s: None)
    try:
        code = watch_mod.watch(
            client, out=out, sleep_fn=sleep_fn, poll_s=POLL_S, since=since
        )
    except _Seen:
        code = None
    return code, lines


def test_seeds_at_the_tail(app):
    # History from before the watch: replaying it would raise a stale PROMPT
    # and report the earlier workflow's end as news.
    app.buffer.append("prompt_raised", {"question_id": 1, "type": "Confirm"})
    app.buffer.append("workflow_completed", {})

    def run():
        app.buffer.append("task_started", {"item_name": "lamella-01"})
        app.buffer.append("prompt_raised", {"question_id": 2, "type": "Confirm"})
        app.buffer.append("prompt_answered", {"question_id": 2}, actor="operator")
        app.buffer.append("milling_progress", {})  # not worth a wake-up
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert code is None
    assert lines == [
        "EVENT task_started lamella-01",
        "PROMPT question=2 type=Confirm",
        "ANSWERED question=2 by=operator response=None",
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_since_resumes_where_the_last_watch_left_off(app):
    # A bounded watch returned at seq 1; what landed before the next one
    # started is still printed, and nothing from before seq 1 is.
    app.buffer.append("prompt_raised", {"question_id": 1, "type": "Confirm"})
    app.buffer.append("request_decided", {"request_id": "ab12", "outcome": "approved"})
    app.buffer.append("workflow_completed", {})

    code, lines = _watch({}, since=1)

    assert code is None
    assert lines == ["REQUEST ab12 approved", "WORKFLOW ENDED (workflow_completed)"]


def test_the_operator_s_message_is_a_line_that_says_how_to_answer(app):
    # FIB-1423: the operator wrote from the card.
    def run():
        app.buffer.append("operator_message", {"id": "ab12", "text": "skip\n03"})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert lines[0] == (
        "MESSAGE id=ab12 from the operator: skip 03 -- answer with "
        "experiment_add_note (reply_to=ab12). It is not an approval: your level "
        "still decides what you may do"
    )


def test_an_answer_says_when_it_took_the_agent_s_suggestion(app):
    def run():
        app.buffer.append(
            "prompt_answered",
            {"question_id": 3, "response": True, "suggested_by": "agent"},
            actor="operator",
        )
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert lines[0] == (
        "ANSWERED question=3 by=operator response=True suggested_by=agent"
    )


def test_a_dismissed_suggestion_says_why(app):
    # FIB-1424: the operator declined the agent's suggestion on the card.
    def run():
        app.buffer.append(
            "prompt_suggestion_dismissed",
            {"question_id": 3, "response": True, "reason": "still curtaining"},
        )
        app.buffer.append("prompt_suggestion_dismissed", {"question_id": 4})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert lines[:2] == [
        "SUGGESTION DISMISSED question=3 reason=still curtaining",
        "SUGGESTION DISMISSED question=4",
    ]


def test_the_agent_s_request_is_followed_to_its_end(app):
    def run():
        app.buffer.append(
            "request_decided",
            {"request_id": "ab12", "outcome": "approved", "decided_by": "operator"},
        )
        app.buffer.append("request_applied", {"request_id": "ab12", "index": 0})
        app.buffer.append(
            "request_finished",
            {"request_id": "ab12", "status": "failed", "applied": 1, "of": 2},
        )
        app.buffer.append(
            "request_finished",
            {
                "request_id": "gh78",
                "status": "applied",
                "applied": 2,
                "of": 2,
                "cancelled": 1,
            },
        )
        app.buffer.append(
            "request_decided",
            {"request_id": "cd34", "outcome": "rejected", "reason": "not now"},
        )
        app.buffer.append(
            "request_cancelled",
            {"request_id": "ef56", "why": "the operator stopped the agent"},
        )
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert lines[:5] == [
        "REQUEST ab12 approved",
        "REQUEST ab12 failed (1 of 2 applied)",
        # A mill in it was stopped part way (FIB-1398).
        "REQUEST gh78 applied (2 of 2; run cancelled)",
        "REQUEST cd34 rejected: not now",
        "REQUEST ef56 cancelled: the operator stopped the agent",
    ]


def test_a_long_command_s_end_is_a_line(app):
    def run():
        app.buffer.append(
            "command_ended",
            {"run_id": "r1", "service": "milling", "command": "run", "state": "done"},
        )
        app.buffer.append(
            "command_ended",
            {
                "run_id": "r2",
                "service": "spot_burn",
                "command": "run",
                "state": "failed",
                "error": "the beam would not blank",
            },
        )
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert lines[:2] == [
        "COMMAND ENDED milling.run run=r1 state=done",
        "COMMAND ENDED spot_burn.run run=r2 state=failed (the beam would not blank)",
    ]


def test_task_lines_name_the_task(app):
    def run():
        app.buffer.append(
            "task_started", {"item_name": "01-alive-dory", "task_name": "Rough Milling"}
        )
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert code is None
    assert lines[0] == "EVENT task_started 01-alive-dory Rough Milling"


def test_a_skip_line_says_why_and_which_requirement(app):
    def run():
        app.buffer.append(
            "task_skipped",
            {
                "item_name": "01-alive-dory",
                "task_name": "Rough Milling",
                "skip_reason": "missing_prereqs",
                "missing_prereqs": ["Mill Fiducial"],
            },
        )
        app.buffer.append(
            "task_skipped",
            {"item_name": "02", "task_name": "Rough Milling", "skip_reason": "failure"},
        )
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert code is None
    assert lines[:2] == [
        "EVENT task_skipped 01-alive-dory Rough Milling"
        " (missing_prereqs: Mill Fiducial)",
        "EVENT task_skipped 02 Rough Milling (failure)",
    ]


def test_watch_is_found_after_the_shared_options():
    from fibsem.mcp.sidecar import _watch_argv

    assert _watch_argv(["watch", "--help"]) == ["--help"]
    assert _watch_argv(["--url", "http://x", "watch", "--help"]) == [
        "--url",
        "http://x",
        "--help",
    ]
    assert _watch_argv(["--token=t", "watch"]) == ["--token=t"]
    assert _watch_argv(["--check"]) is None
    assert _watch_argv(["--wait", "5", "watch"]) is None
    assert _watch_argv([]) is None


def test_quiet_polls_keep_watching(app):
    code, lines = _watch({4: lambda: app.buffer.append("workflow_cancelled", {})})
    assert code is None
    assert lines == ["WORKFLOW ENDED (workflow_cancelled)"]


@pytest.mark.parametrize(
    "kind, payload, line",
    [
        ("workflow_stalled", {}, "WORKFLOW ENDED (workflow_stalled)"),
        (
            "workflow_failed",
            {"error": "cannot represent an object"},
            "WORKFLOW ENDED (workflow_failed: cannot represent an object)",
        ),
        (
            "workflow_completed",
            {"task_counts": {"completed": 0, "failed": 0, "skipped": 1}},
            "WORKFLOW ENDED (workflow_completed: nothing ran, 1 skipped)",
        ),
        (
            "workflow_completed",
            {"task_counts": {"completed": 3, "failed": 1, "skipped": 0}},
            "WORKFLOW ENDED (workflow_completed: 3 completed, 1 failed)",
        ),
    ],
)
def test_a_workflow_s_end_says_how_it_ended(app, kind, payload, line):
    code, lines = _watch({3: lambda: app.buffer.append(kind, payload)})
    assert code is None
    assert lines == [line]


def test_a_workflow_that_stops_without_an_end_event_is_reported(app):
    # FIB-1331: the run died outside a task, so no end event was recorded;
    # only /app/workflow/status knows it is over.
    app.set_running(True)

    def crash():
        app.buffer.append("task_completed", {"item_name": "03-stable-owl"})
        app.set_running(False)

    code, lines = _watch({3: crash})

    assert code is None
    assert lines == [
        "EVENT task_completed 03-stable-owl",
        "WORKFLOW ENDED (stopped without an end event)",
    ]


def test_a_watch_started_before_the_workflow_waits_for_it(app):
    # Nothing running at the start is not an end: the agent starts the
    # watch, then the workflow.
    def start():
        app.set_running(True)
        app.buffer.append("workflow_started", {})

    def crash():
        app.set_running(False)

    code, lines = _watch({4: start, 7: crash})

    assert code is None
    assert lines == [
        "WORKFLOW STARTED",
        "WORKFLOW ENDED (stopped without an end event)",
    ]


def test_a_workflow_s_end_does_not_end_the_watch(app):
    # FIB-1431: the agent starts the next workflow itself, and between the two
    # it still has to hear the app.
    def first():
        app.buffer.append("workflow_started", {})
        app.buffer.append("workflow_completed", {})

    def between():
        app.buffer.append("prompt_raised", {"question_id": 4, "type": "Confirm"})

    def second():
        app.buffer.append("workflow_started", {})
        app.buffer.append("workflow_cancelled", {})

    code, lines = _watch(
        {2: first, 4: between, 6: second},
        until=lambda lines: len(lines) == 5,
    )

    assert code is None
    assert lines == [
        "WORKFLOW STARTED",
        "WORKFLOW ENDED (workflow_completed)",
        "PROMPT question=4 type=Confirm",
        "WORKFLOW STARTED",
        "WORKFLOW ENDED (workflow_cancelled)",
    ]


def test_after_a_workflow_stops_without_an_end_event_the_watch_carries_on(app):
    # The fallback is reported once, and the next workflow is followed as the
    # first was.
    app.set_running(True)

    def crash():
        app.set_running(False)

    def next_one():
        app.set_running(True)
        app.buffer.append("workflow_started", {})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({3: crash, 8: next_one}, until=lambda lines: len(lines) == 3)

    assert code is None
    assert lines == [
        "WORKFLOW ENDED (stopped without an end event)",
        "WORKFLOW STARTED",
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_an_end_event_beats_the_fallback(app):
    # The status already says stopped when the end event lands: the watch
    # reports the event, not the fallback.
    app.set_running(True)

    def finish():
        app.set_running(False)
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({4: finish})
    assert code is None
    assert lines == ["WORKFLOW ENDED (workflow_completed)"]


def test_evicted_events_print_a_gap_then_carry_on(microscope, tmp_path, discovery):
    server, thread, app = _serve(microscope, tmp_path, EventBuffer(maxlen=3))
    write_discovery_file("127.0.0.1", app.port, app.auth, path=discovery)

    def burst():
        for i in range(5):
            app.buffer.append("tick", {"i": i})
        app.buffer.append("question_asked", {"item_name": "l1", "task_name": "Mill"})
        app.buffer.append("workflow_completed", {})

    try:
        code, lines = _watch({2: burst})
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    assert code is None
    assert lines[0].startswith("GAP:")
    assert lines[1].startswith("QUESTION item=l1 task=Mill")
    assert lines[2] == "WORKFLOW ENDED (workflow_completed)"


def test_a_new_stream_says_the_app_restarted_and_reads_it_from_the_start(app):
    """A restarted app's seqs start again: nothing was lost, so the line says it
    restarted rather than GAP (FIB-1421)."""
    for _ in range(5):
        app.buffer.append("tick", {})

    def restart():
        # The app restarted with a fresh buffer: seqs begin again below the
        # cursor, and every event in it is news.
        app.context._event_buffer = EventBuffer()
        app.buffer.append("task_started", {"item_name": "l2"})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: restart})

    assert code is None
    assert lines == [
        watch_mod.RESTARTED_LINE,
        "EVENT task_started l2",
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_a_new_stream_past_the_cursor_is_still_a_restart(app):
    """A new session that has already run past the cursor is told by its stream,
    not its seqs: read from its start, not from the old cursor."""
    app.buffer.append("tick", {})

    def restart():
        app.context._event_buffer = EventBuffer()
        app.buffer.append("task_started", {"item_name": "l2"})
        app.buffer.append("tick", {})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: restart})

    assert code is None
    assert lines == [
        watch_mod.RESTARTED_LINE,
        "EVENT task_started l2",
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_resuming_with_a_seq_from_before_a_restart_says_so(app):
    """``--since`` from a watch of the app before it restarted."""
    app.buffer.append("task_started", {"item_name": "l3"})
    app.buffer.append("workflow_completed", {})

    code, lines = _watch({}, since=40)

    assert code is None
    assert lines == [
        watch_mod.RESTARTED_LINE,
        "EVENT task_started l3",
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_a_401_rereads_discovery_and_carries_on(app, discovery):
    def rotate():
        # A restart that kept the address but changed the token.
        app.auth.token = "rotated"
        write_discovery_file("127.0.0.1", app.port, app.auth, path=discovery)
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: rotate})

    assert code is None
    assert lines == ["WORKFLOW ENDED (workflow_completed)"]


def test_a_server_that_stays_gone_ends_the_watch(app):
    def gone():
        raise ConnectionRefusedError("refused")

    sleeps = []
    code, lines = _watch({n: gone for n in range(2, 20)}, sleeps=sleeps)

    assert code == 1
    assert lines[-1].startswith("SERVER GONE")
    assert len(sleeps) == watch_mod.FAILURES_BEFORE_GIVING_UP - 1


def test_agent_off_is_not_the_app_stopping(app):
    """The operator set the level to Off: the app still runs, and refuses the
    agent until the level comes back (FIB-1429). Events wait in the buffer."""

    def off():
        app.auth.set_level(AgentLevel.OFF)
        app.buffer.append("task_started", {"item_name": "lamella-01"})

    def back():
        app.auth.set_level(AgentLevel.WATCH)
        app.buffer.append("workflow_completed", {})

    sleeps = []
    # request 2 is the status check, then four refused polls
    code, lines = _watch({2: off, 7: back}, sleeps=sleeps)

    assert code is None
    assert lines == [
        watch_mod.AGENT_OFF_LINE,
        watch_mod.AGENT_BACK_LINE,
        "EVENT task_started lamella-01",
        "WORKFLOW ENDED (workflow_completed)",
    ]
    # as many refusals as end a watch on a gone server: these never count
    assert len(sleeps) == watch_mod.FAILURES_BEFORE_GIVING_UP


def test_a_watch_started_while_the_agent_is_off_waits_for_it(app):
    app.auth.set_level(AgentLevel.OFF)

    def back():
        app.auth.set_level(AgentLevel.WATCH)

    def finish():
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({3: back, 4: finish})

    assert code is None
    assert lines == [
        watch_mod.AGENT_OFF_LINE,
        watch_mod.AGENT_BACK_LINE,
        "WORKFLOW ENDED (workflow_completed)",
    ]


def test_a_refused_token_ends_the_watch_at_once(app, discovery, monkeypatch):
    monkeypatch.setenv("FIBSEM_SERVER_TOKEN", "wrong")
    monkeypatch.setenv("FIBSEM_SERVER_URL", f"http://127.0.0.1:{app.port}")
    code, lines = _watch({})
    assert code == 1
    assert lines == [
        "SERVER UNAVAILABLE: /app/events gave HTTP 401. Is the app running?"
    ]


def test_without_an_event_stream_the_poll_parks_and_the_watch_ends(
    microscope, tmp_path, discovery
):
    server, thread, app = _serve(microscope, tmp_path, None)
    write_discovery_file("127.0.0.1", app.port, app.auth, path=discovery)
    try:
        client = watch_mod.connect()
        # FIB-870: no buffer still honours the timeout, so a client that
        # misses `available` waits instead of spinning.
        start = time.monotonic()
        payload, failure = watch_mod._poll(client, since=0, timeout=0.5)
        assert failure is None
        assert payload["available"] is False
        assert time.monotonic() - start >= 0.5

        lines = []
        assert watch_mod.watch(client, out=lines.append) == 1
        assert lines[0].startswith("NO EVENTS")
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_the_subcommand_runs_without_the_mcp_extra(tmp_path):
    # Watching needs only HTTP: with neither the [mcp] nor the [server] extra
    # importable, the entry point still runs it. Port 9 (discard) refuses.
    script = (
        "import sys\n"
        "for name in ('mcp', 'httpx', 'fastapi', 'uvicorn', 'starlette'):\n"
        "    sys.modules[name] = None\n"
        "from fibsem.mcp.sidecar import main\n"
        "main(['watch', '--url', 'http://127.0.0.1:9', '--token', 't'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert result.returncode == 1, result.stderr
    assert result.stdout.startswith("SERVER UNAVAILABLE"), result.stdout


def test_every_mill_is_followed_by_its_stage_and_end_lines(app, microscope, tmp_path):
    """A mill run as a workflow task runs it, through the service and not the
    App Server: its recorded events print a line per stage and one at the end."""
    from fibsem import config as fcfg
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig, run_milling_task
    from fibsem.workflows.events import attach_microscope_taps

    stages = [FibsemMillingStage(name=name) for name in ("Rough", "Polish")]
    stages[1].enabled = False
    config = FibsemMillingTaskConfig.from_stages(stages, name="Trench")
    config.alignment.enabled = False
    config.acquisition.acquire_sem = config.acquisition.acquire_fib = False
    disposers = attach_microscope_taps(app.buffer, microscope)
    data_path = fcfg.DATA_CC_PATH
    fcfg.DATA_CC_PATH = str(tmp_path / "data")
    try:
        task = run_milling_task(microscope, config)
    finally:
        fcfg.DATA_CC_PATH = data_path
        for dispose in disposers:
            dispose()
    rough = stages[0]
    run = task.task_id

    code, lines = _watch(
        {},
        since=0,
        until=lambda lines: lines[-1].startswith("COMMAND ENDED"),
    )

    mill = [line for line in lines if line.startswith("MILL")]
    assert mill == [
        f"MILL STAGE STARTED 1/1 'Rough' task='Trench' "
        f"current={rough.milling.milling_current:g}A "
        f"depth={rough.pattern.depth:g}m run={run}",
        f"MILL ENDED 'Trench' status=completed run={run} stages: "
        "Rough=completed, Polish=skipped (turned off in the protocol)",
    ]
    # The command's own end stays its line.
    assert f"COMMAND ENDED milling.run_task run={run} state=done" in lines


def test_a_mill_line_names_the_item_and_a_failure():
    lines = []
    watch_mod._print_events(
        [
            {
                "seq": 1,
                "kind": "milling",
                "item": {"id": "a", "name": "01-alive-dory"},
                "payload": {
                    "milling_run_id": "r1",
                    "milling_task_name": "Mill Rough",
                    "status": "failed",
                    "error": "RuntimeError: the beam would not blank",
                    "stages": [{"name": "Rough", "status": "failed"}],
                },
            }
        ],
        0,
        lines.append,
    )
    assert lines == [
        "MILL ENDED 'Mill Rough' item=01-alive-dory status=failed run=r1 "
        "stages: Rough=failed (RuntimeError: the beam would not blank)"
    ]

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
from fibsem.applications.autolamella.server.events import EventBuffer  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.mcp import watch as watch_mod  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.discovery import write_discovery_file  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402

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


def _watch(steps, sleeps=None):
    """Watch the discovered server, running ``steps[n]`` before request n."""
    client_type = type("Steered", (_Steered,), {"steps": steps, "requests": 0})
    client = watch_mod.connect(client_factory=client_type)
    assert client is not None
    lines = []
    sleep_fn = sleeps.append if sleeps is not None else (lambda s: None)
    code = watch_mod.watch(client, out=lines.append, sleep_fn=sleep_fn, poll_s=POLL_S)
    return code, lines


def test_seeds_at_the_tail_and_ends_with_the_run(app):
    # History from before the watch: replaying it would raise a stale PROMPT
    # and end the watch on the earlier run's completion.
    app.buffer.append("prompt_raised", {"question_id": 1, "type": "Confirm"})
    app.buffer.append("workflow_completed", {})

    def run():
        app.buffer.append("task_started", {"item_name": "lamella-01"})
        app.buffer.append("prompt_raised", {"question_id": 2, "type": "Confirm"})
        app.buffer.append(
            "prompt_answered", {"question_id": 2, "answered_by": "operator"}
        )
        app.buffer.append("milling_progress", {})  # not worth a wake-up
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: run})

    assert code == 0
    assert lines == [
        "EVENT task_started lamella-01",
        "PROMPT question=2 type=Confirm",
        "ANSWERED question=2 by=operator response=None",
        "RUN ENDED (workflow_completed)",
    ]


def test_quiet_polls_keep_watching(app):
    code, lines = _watch({4: lambda: app.buffer.append("workflow_cancelled", {})})
    assert code == 0
    assert lines == ["RUN ENDED (workflow_cancelled)"]


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

    assert code == 0
    assert lines[0].startswith("GAP:")
    assert lines[1].startswith("QUESTION item=l1 task=Mill")
    assert lines[2] == "RUN ENDED (workflow_completed)"


def test_a_new_stream_prints_a_gap_and_reads_it_from_the_start(app):
    for _ in range(5):
        app.buffer.append("tick", {})

    def restart():
        # The app restarted with a fresh buffer: seqs begin again below the
        # cursor, and every event in it is news.
        app.context._event_buffer = EventBuffer()
        app.buffer.append("task_started", {"item_name": "l2"})
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: restart})

    assert code == 0
    assert lines[0].startswith("GAP:")
    assert lines[1:] == ["EVENT task_started l2", "RUN ENDED (workflow_completed)"]


def test_a_401_rereads_discovery_and_carries_on(app, discovery):
    def rotate():
        # A restart that kept the address but changed the token.
        app.auth.token = "rotated"
        write_discovery_file("127.0.0.1", app.port, app.auth, path=discovery)
        app.buffer.append("workflow_completed", {})

    code, lines = _watch({2: rotate})

    assert code == 0
    assert lines == ["RUN ENDED (workflow_completed)"]


def test_a_server_that_stays_gone_ends_the_watch(app):
    def gone():
        raise ConnectionRefusedError("refused")

    sleeps = []
    code, lines = _watch({n: gone for n in range(2, 20)}, sleeps=sleeps)

    assert code == 1
    assert lines[-1].startswith("SERVER GONE")
    assert len(sleeps) == watch_mod.FAILURES_BEFORE_GIVING_UP - 1


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

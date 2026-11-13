"""The agent on another computer, through an SSH tunnel.

Two processes with homes and state roots of their own stand in for the two
computers: ``fibsem-autolamella-server`` on the microscope PC, and the harness
with a stand-in model (``remote_agent.py``) on the agent computer. A port
forward on this machine stands in for ``ssh -L``. The agent pairs with a code,
watches through the tunnel, says what to check when the tunnel drops, and
watches again once it is back.
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from fibsem.client import transport

ROOT = Path(__file__).resolve().parents[2]
REMOTE_AGENT = Path(__file__).with_name("remote_agent.py")
WAIT_S = 120.0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Tunnel:
    """``ssh -L port:127.0.0.1:target`` stood in: ``close`` drops it and every
    connection through it, ``open`` brings it back on the same port."""

    def __init__(self, target: int) -> None:
        self.target = target
        self.port = _free_port()
        self._listener = None
        self._live = []
        self._lock = threading.Lock()

    def open(self) -> None:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", self.port))
        listener.listen()
        self._listener = listener
        threading.Thread(target=self._accept, args=(listener,), daemon=True).start()

    def close(self) -> None:
        # A shutdown wakes the accept, so the port stops answering at once.
        try:
            self._listener.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._listener.close()
        with self._lock:
            live, self._live = self._live, []
        for sock in live:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def _accept(self, listener) -> None:
        while True:
            try:
                near, _ = listener.accept()
            except OSError:
                return
            try:
                far = socket.create_connection(("127.0.0.1", self.target))
            except OSError:
                near.close()
                continue
            with self._lock:
                self._live += [near, far]
            for a, b in ((near, far), (far, near)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    @staticmethod
    def _pump(source, sink) -> None:
        try:
            while data := source.recv(65536):
                sink.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, sink):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def _computer(folder: Path) -> dict:
    """The environment of a computer whose home is *folder*: its own
    ``~/.fibsem`` and state root, and nothing pointing at another server."""
    state = folder / "state"
    (state / "config").mkdir(parents=True)
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("FIBSEM_SERVER_") and k != "FIBSEM_DISCOVERY_FILE"
    }
    env.update(
        HOME=str(folder),
        USERPROFILE=str(folder),
        FIBSEM_HOME=str(state),
        FIBSEM_UNRELEASED="1",
        FIBSEM_SIM_NO_DELAY="1",
        PYTHONPATH=str(ROOT),
        QT_QPA_PLATFORM="offscreen",
    )
    return env


def _until(predicate, what: str, timeout: float = WAIT_S, log=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = predicate()
        if found:
            return found
        time.sleep(0.1)
    said = "\n".join(log.lines[-40:]) if log is not None else ""
    raise AssertionError(f"timed out waiting for {what}\n{said}")


class Lines:
    """A process's output, read on a thread so it never blocks."""

    def __init__(self, stream) -> None:
        self.lines = []
        threading.Thread(target=self._read, args=(stream,), daemon=True).start()

    def _read(self, stream) -> None:
        for line in stream:
            self.lines.append(line.rstrip("\n"))

    def prompts(self):
        return [
            json.loads(line)["prompt"] for line in self.lines if line.startswith("{")
        ]


def _end(process) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)


@pytest.fixture
def microscope_pc(tmp_path):
    """The App Server on the microscope PC, serving a new experiment on the
    Demo microscope: its home, its url and its operator's client."""
    home = tmp_path / "microscope-pc"
    env = _computer(home)
    port = _free_port()
    log = open(tmp_path / "microscope-pc.log", "w")
    server = subprocess.Popen(
        [sys.executable, "-m", "fibsem.applications.autolamella.launch"]
        + [str(home / "grid-1"), "--create", "--port", str(port)],
        env=env,
        cwd=ROOT,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    discovery = home / ".fibsem" / "app-server.json"
    try:
        _until(
            lambda: discovery.is_file() or server.poll() is not None,
            "the App Server",
        )
        assert server.poll() is None, (tmp_path / "microscope-pc.log").read_text()
        token = json.loads((home / ".fibsem" / "operator-token.json").read_text())
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"Authorization": f"Bearer {token['token']}"},
            timeout=30,
        ) as operator:
            yield home, port, operator
    finally:
        _end(server)
        log.close()


def test_an_agent_on_another_computer_works_through_the_tunnel_and_says_when_it_drops(
    microscope_pc, tmp_path
):
    pc_home, port, operator = microscope_pc
    tunnel = Tunnel(port)
    tunnel.open()
    url = f"http://localhost:{tunnel.port}"
    laptop_home = tmp_path / "agent-laptop"

    opened = operator.post("/pairing", json={"role": "agent"})
    assert opened.is_success, opened.text
    agent = subprocess.Popen(
        [sys.executable, str(REMOTE_AGENT), url, opened.json()["code"]],
        env=_computer(laptop_home),
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out, log = Lines(agent.stdout), Lines(agent.stderr)
    try:
        # Paired by the code, through the tunnel: the agent computer keeps the
        # token, and the microscope PC lists it.
        _until(lambda: out.prompts(), "the kickoff", log=log)
        kept = laptop_home / ".fibsem" / "agent" / f"localhost_{tunnel.port}.json"
        assert kept.is_file()
        computers = operator.get("/pairing").json()["computers"]
        assert [c["name"] for c in computers if c["role"] == "agent"] == [
            "agent-laptop"
        ]
        heard = operator.get("/app/agent").json()
        assert heard["last_heard_s"] is not None and heard["last_heard_s"] < 30
        assert "agent-laptop" in heard["client"]

        # Separate state roots: each side wrote only its own.
        assert not (laptop_home / ".fibsem" / "app-server.json").exists()
        assert not (laptop_home / ".fibsem" / "operator-token.json").exists()
        assert not (pc_home / ".fibsem" / "agent" / kept.name).exists()
        assert (pc_home / "grid-1" / "experiment.yaml").is_file()
        assert not (laptop_home / "grid-1").exists()

        # The tunnel drops: the agent says to check it, and waits.
        tunnel.close()
        _until(
            lambda: any("SSH tunnel" in line for line in log.lines),
            "the agent to say the tunnel is down",
            log=log,
        )
        assert agent.poll() is None, "\n".join(log.lines)

        # Back up: the agent watches again and is told what it missed.
        turns = len(out.prompts())
        tunnel.open()
        back = _until(
            lambda: out.prompts()[turns:],
            "the agent to watch again",
            log=log,
        )
        assert "SERVER GONE" in back[0] and "SSH tunnel" in back[0]
        assert "could not be watched" in back[0]
    finally:
        _end(agent)
        agent.stdout.close()
        agent.stderr.close()
        tunnel.close()


@pytest.mark.parametrize(
    "url, local_app, says",
    [
        ("http://localhost:8001", None, "SSH tunnel to the microscope PC is down"),
        ("http://localhost:8001", 8002, "SSH tunnel to the microscope PC is down"),
        ("http://127.0.0.1:8001", 8001, "the app on this computer isn't serving"),
        ("http://scope-pc:8001", None, "check that scope-pc is on"),
    ],
)
def test_what_to_check_when_nothing_answers(monkeypatch, url, local_app, says):
    found = {"url": f"http://127.0.0.1:{local_app}"} if local_app else None
    monkeypatch.setattr(transport, "read_discovery_file", lambda: found)
    assert says in transport.unreachable_why(url)

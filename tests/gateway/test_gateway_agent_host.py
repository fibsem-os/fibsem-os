"""The Agent Server the gateway runs: started while none runs on this PC, not
again straight after one ended, never over a host started by hand, and
stopped by its stop file when the gateway closes."""

import subprocess
import sys

from fibsem.gateway.agent_host import RETRY_S, TARGET, AgentHostProcess


class Proc:
    def __init__(self, command, **kwargs):
        self.command = command
        self.pid = 4321
        self.returncode = None
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("host", timeout)
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _made(tmp_path, found=None):
    clock = [1000.0]
    started = []

    def popen(command, **kwargs):
        started.append(Proc(command, **kwargs))
        return started[-1]

    host = AgentHostProcess(
        sys.executable,
        home=tmp_path,
        found=found or (lambda: None),
        popen=popen,
        clock=lambda: clock[0],
    )
    return host, started, clock


def test_the_gateway_starts_a_host_once_and_again_only_after_a_wait(tmp_path):
    host, started, clock = _made(tmp_path)
    host.ensure()
    assert len(started) == 1
    command = started[0].command
    assert command[1:3] == ["-m", "fibsem.gateway.child"] and command[4] == TARGET
    host.ensure()  # still starting
    started[0].returncode = 1  # it ended (its port taken, say)
    host.ensure()
    assert len(started) == 1
    clock[0] += RETRY_S
    host.ensure()
    assert len(started) == 2


def test_a_host_started_by_hand_is_used_and_never_stopped(tmp_path):
    found = {"url": "http://127.0.0.1:8030", "pid": 77}
    host, started, _ = _made(tmp_path, lambda: found)
    host.ensure()
    host.close()
    assert started == []
    assert host.url() == "http://127.0.0.1:8030"


def test_closing_stops_the_gateways_host_by_its_stop_file(tmp_path):
    host, started, _ = _made(tmp_path)
    host.ensure()
    stop_file = tmp_path / started[0].command[3].rsplit("/", 1)[-1]
    seen = []
    proc = started[0]

    def wait(timeout=None):
        seen.append(stop_file.exists())
        proc.returncode = 0
        return 0

    proc.wait = wait
    host.close()
    assert seen == [True]
    assert not proc.killed
    assert not stop_file.exists()


def test_a_host_that_wont_stop_is_killed(tmp_path):
    host, started, _ = _made(tmp_path)
    host.ensure()
    host.close(wait_s=0.0)
    assert started[0].killed

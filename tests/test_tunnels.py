"""SSH tunnels kept open by name: forwards both ways, for any caller."""

from types import SimpleNamespace

import pytest

from fibsem.util.tunnels import TunnelDown, Tunnels, command


class FakeSsh:
    def __init__(self, command, fails=None):
        self.command = command
        self.code = None if fails is None else 255
        self.stderr = SimpleNamespace(read=lambda: fails or "")

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def wait(self, timeout=None):
        return self.code

    def kill(self):
        self.code = -9


class Ssh:
    def __init__(self):
        self.started = []
        self.fails = None
        self.now = 0.0
        ports = iter(range(50_000, 60_000))
        self.tunnels = Tunnels(
            spawn=self.spawn,
            port=lambda: next(ports),
            accepts=self.accepts,
            which=lambda name: "/usr/bin/ssh",
            clock=lambda: self.now,
            sleep=self.sleep,
        )

    def sleep(self, seconds):
        self.now += seconds

    def spawn(self, command):
        self.started.append(FakeSsh(command, self.fails))
        return self.started[-1]

    def accepts(self, port):
        return any(
            process.poll() is None and f"127.0.0.1:{port}:127.0.0.1" in forward
            for process in self.started
            for forward, option in zip(process.command[1:], process.command)
            if option == "-L"
        )


@pytest.fixture
def ssh():
    return Ssh()


def test_a_tunnel_forwards_one_port_here_and_lends_ports_there(ssh):
    tunnel = ssh.tunnels.ensure("scope", "user@scope-1", 8010, reverse=[(8031, 40123)])
    said = command(tunnel)
    assert said[-2:] == ["--", "user@scope-1"]
    assert said[said.index("-L") + 1] == "127.0.0.1:50000:127.0.0.1:8010"
    assert said[said.index("-R") + 1] == "127.0.0.1:8031:127.0.0.1:40123"
    assert tunnel.open and tunnel.url == "http://127.0.0.1:50000"


def test_a_tunnel_with_only_reverse_forwards_is_open_once_ssh_has_stayed_up(ssh):
    tunnel = ssh.tunnels.ensure("chat", "user@scope-1", reverse=[(8031, 40123)])
    assert tunnel.open and "-L" not in command(tunnel)
    assert ssh.now >= 1.5


def test_a_reverse_forward_the_other_pc_refuses_says_why(ssh):
    ssh.fails = "Error: remote port forwarding failed for listen port 8031"
    with pytest.raises(TunnelDown, match="remote port forwarding failed"):
        ssh.tunnels.ensure("chat", "user@scope-1", reverse=[(8031, 40123)])


def test_asking_for_other_forwards_reopens_the_tunnel_on_its_port(ssh):
    first = ssh.tunnels.ensure("scope", "user@scope-1", 8010)
    again = ssh.tunnels.ensure("scope", "user@scope-1", 8010)
    assert again is first and len(ssh.started) == 1
    second = ssh.tunnels.ensure("scope", "user@scope-1", 8010, reverse=[(8031, 40123)])
    assert len(ssh.started) == 2 and ssh.started[0].poll() is not None
    assert second.open and second.port == first.port
    back = ssh.tunnels.ensure("scope", "user@scope-1", 8010)
    assert len(ssh.started) == 3 and back.reverse == () and back.port == first.port


def test_a_tunnel_given_nothing_to_forward_starts_no_ssh(ssh):
    with pytest.raises(TunnelDown, match="nothing to forward"):
        ssh.tunnels.ensure("scope", "user@scope-1")
    assert ssh.started == []

"""``fibsem-gateway stop`` and ``restart``: the running gateway is asked to
quit (and start again); one that doesn't answer is ended by its pid."""

from urllib.error import URLError

import pytest

from fibsem.gateway import main as gateway_main


class Here:
    """The gateway file, the quit route, and the clock, as the command sees them."""

    def __init__(self, monkeypatch, running=True, answers=True, comes_back=True):
        self.file = {"url": "http://127.0.0.1:8010", "pid": 100} if running else None
        self.asked, self.started, self.ended = [], [], []
        self.time = 0.0

        def call(method, route, body=None, timeout=3.0):
            self.asked.append((method, route, body))
            if not answers:
                raise URLError("refused")
            restart = body["restart"]
            self.file = {"url": self.file["url"], "pid": 101} if restart else None
            return {"stopping": True, "restart": restart}

        def start_gateway(flags=()):
            self.started.append(list(flags))
            if comes_back:
                self.file = {"url": "http://127.0.0.1:8010", "pid": 102}

        def end(pid):
            self.ended.append(pid)
            self.file = None

        def sleep(seconds):
            self.time += seconds

        monkeypatch.setattr(gateway_main, "read_gateway_file", lambda: self.file)
        monkeypatch.setattr(gateway_main, "call", call)
        monkeypatch.setattr(gateway_main, "start_gateway", start_gateway)
        monkeypatch.setattr(gateway_main, "_end", end)
        self.stop = lambda restart=False: gateway_main.stop(
            restart, clock=lambda: self.time, sleep=sleep
        )


def test_stop_asks_the_gateway_to_quit(monkeypatch, capsys):
    here = Here(monkeypatch)
    assert here.stop() == 0
    assert here.asked == [("POST", "/gateway/quit", {"restart": False})]
    assert here.ended == [] and here.started == []
    assert "stopped" in capsys.readouterr().out


def test_stop_with_none_running_says_so(monkeypatch, capsys):
    here = Here(monkeypatch, running=False)
    assert here.stop() == 0
    assert here.asked == [] and "No gateway runs here" in capsys.readouterr().out


def test_a_gateway_that_doesnt_answer_is_ended_by_its_pid(monkeypatch):
    here = Here(monkeypatch, answers=False)
    assert here.stop() == 0
    assert here.ended == [100]


def test_restart_waits_for_the_new_gateway(monkeypatch, capsys):
    here = Here(monkeypatch)
    assert here.stop(restart=True) == 0
    assert here.asked == [("POST", "/gateway/quit", {"restart": True})]
    assert "pid 101" in capsys.readouterr().out


@pytest.mark.parametrize("answers, running", [(False, True), (True, False)])
def test_restart_starts_one_itself_when_it_must(monkeypatch, answers, running):
    """After ending one that didn't answer, or when none ran."""
    here = Here(monkeypatch, answers=answers, running=running)
    assert here.stop(restart=True) == 0
    assert here.started == [[]]


def test_a_restart_that_doesnt_come_back_says_so(monkeypatch, capsys):
    here = Here(monkeypatch, running=False, comes_back=False)
    assert here.stop(restart=True) == 1
    assert "didn't come back" in capsys.readouterr().out


def test_a_restarted_gateway_keeps_its_flags(monkeypatch):
    started = []
    monkeypatch.setattr(
        gateway_main.subprocess, "Popen", lambda command, **kw: started.append(command)
    )
    gateway_main.start_gateway(["--no-tray", "--manufacturer", "Demo"])
    assert started[0][1:] == [
        "-m",
        "fibsem.gateway",
        "run",
        "--no-tray",
        "--manufacturer",
        "Demo",
    ]

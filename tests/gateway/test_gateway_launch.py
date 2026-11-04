"""Opening the gateway's page (`fibsem.gateway.launch`): the gateway started
when none runs, then its page signed in as the operator."""

import pytest

from fibsem.gateway import main as gateway_main
from fibsem.gateway.launch import OpenFailed, gateway_page

URL = "http://127.0.0.1:8010"


class FakeGateway:
    """The gateway as `fibsem.gateway.main.call` reaches it."""

    def __init__(self, up=True):
        self.up = up
        self.started = 0

    def ask(self, method, route, body=None, timeout=3.0):
        return {"mode": "off"} if self.up else None

    def start(self):
        self.started += 1


def _page(gateway, start=None, clock=None):
    return gateway_page(
        ask=gateway.ask,
        start=start or gateway.start,
        found=lambda: {"url": URL} if gateway.up else None,
        token=lambda: "operator",
        clock=clock or (lambda: 0.0),
        sleep=lambda seconds: None,
    )


def test_a_running_gateway_opens_signed_in():
    gateway = FakeGateway()
    assert _page(gateway) == f"{URL}/ui/gateway#token=operator"
    assert gateway.started == 0


def test_no_gateway_starts_one_then_opens():
    gateway = FakeGateway(up=False)

    def start():
        gateway.started += 1
        gateway.up = True

    assert _page(gateway, start=start) == f"{URL}/ui/gateway#token=operator"
    assert gateway.started == 1


def test_a_gateway_that_never_answers_says_how_to_see_why():
    times = iter(range(0, 1000, 30))
    with pytest.raises(OpenFailed, match="Run `fibsem-gateway`"):
        _page(FakeGateway(up=False), clock=lambda: float(next(times)))


def test_the_shortcut_opens_the_gateway(tmp_path, monkeypatch):
    from fibsem.tools import desktop_shortcut

    monkeypatch.setattr(desktop_shortcut, "get_desktop_directory", lambda: tmp_path)
    monkeypatch.setattr(
        desktop_shortcut, "find_entry_point", lambda name: tmp_path / name
    )
    written = []
    monkeypatch.setattr(
        desktop_shortcut, "_write_windows_lnk", lambda *a: written.append(a)
    )
    assert gateway_main.gateway_shortcut() == 0
    shortcut = next(tmp_path.iterdir()) if not written else written[0][0]
    assert "Agent Gateway" in shortcut.name
    if not written:
        assert "fibsem-gateway-open" in shortcut.read_text()

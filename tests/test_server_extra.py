"""The base install, without the [server] extra: the host says why it isn't
serving, the server commands say what to install, and a client of a Device
Server still works."""

import subprocess
import sys
from types import SimpleNamespace

from fibsem import server
from fibsem.applications.autolamella import launch as autolamella_launch
from fibsem.applications.volume import launch as volume_launch
from fibsem.gateway import main as gateway_main
from fibsem.server import headless, hosting

MISSING = "the [server] extra (uvicorn not installed): pip install fibsem[server]"


def test_the_extra_is_missing_when_either_package_is(monkeypatch):
    monkeypatch.setattr(
        server, "find_spec", lambda name: None if name == "uvicorn" else object()
    )
    assert server.missing_server_extra() == (
        "the [server] extra (uvicorn not installed): pip install fibsem[server]"
    )
    monkeypatch.setattr(server, "find_spec", lambda name: object())
    assert server.missing_server_extra() is None


def test_the_host_keeps_why_it_is_not_serving(monkeypatch):
    monkeypatch.setattr(hosting, "missing_server_extra", lambda: MISSING)
    host = hosting.AppServerHost(SimpleNamespace(), lambda: None)

    assert host.start() is False
    assert not host.running
    assert host.start_error == MISSING


def test_the_server_commands_say_what_to_install(monkeypatch, capsys):
    monkeypatch.setattr(headless, "missing_server_extra", lambda: MISSING)
    monkeypatch.setattr(gateway_main, "missing_server_extra", lambda: MISSING)

    assert autolamella_launch.main([]) == 1
    assert volume_launch.main(["somewhere"]) == 1
    assert gateway_main.main(["run"]) == 1

    out = capsys.readouterr().out.splitlines()
    assert out == [
        f"error: fibsem-autolamella-server needs {MISSING}",
        f"error: fibsem-volume needs {MISSING}",
        f"error: fibsem-gateway needs {MISSING}",
    ]


def test_a_device_server_client_needs_no_server_extra(tmp_path):
    # fastapi and uvicorn made unimportable in a fresh interpreter
    code = (
        "import sys\n"
        "sys.modules['fastapi'] = sys.modules['uvicorn'] = None\n"
        "import fibsem.server.operator_token as t\n"
        f"t.DEVICE_TOKEN_FILE = t.Path({str(tmp_path / 'none.json')!r})\n"
        "import fibsem.drivers.remote.devices\n"
        "from fibsem.client.pairing import device_server_token\n"
        "print(device_server_token('http://127.0.0.1:9'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "None"

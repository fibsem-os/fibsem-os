"""``fibsem-cli`` through an App Server on the Demo, as the operator: each
command takes the route the agent's tools take, and the server's checks.

Skipped when the [server] extra is not installed.
"""

import json
import math
import os

import pytest
import yaml

pytest.importorskip("fastapi")
uvicorn = pytest.importorskip("uvicorn")

from fibsem import cli, utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.app_context import AppContext  # noqa: E402
from fibsem.structures import FibsemImage  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402
from tests.fixtures.app_server import serving  # noqa: E402

OPERATOR = "operator-token"
# A stage offset as the codec reads one: every field named.
DELTA = {"x": 1e-6, "y": 0.0, "z": 0.0, "r": 0.0, "t": 0.0, "coordinate_system": "RAW"}


@pytest.fixture(scope="module")
def microscope():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    yield microscope
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


@pytest.fixture(scope="module")
def session(microscope):
    session = AppSession()
    session.microscope = microscope
    return session


@pytest.fixture(scope="module")
def url(microscope, session):
    # The agent at Watch: the operator token acts at any level.
    auth = AuthConfig.generate(level="watch", token="agent-token")
    auth.operator_token = OPERATOR
    with serving(
        build_server(microscope, app_context=AppContext(session), auth=auth)
    ) as url:
        yield url


@pytest.fixture
def run(url, monkeypatch, capsys):
    """Run fibsem-cli with *argv* against the server: (exit code, stdout, stderr)."""
    monkeypatch.setenv("FIBSEM_SERVER_URL", url)
    monkeypatch.setenv("FIBSEM_OPERATOR_TOKEN", OPERATOR)

    def run(*argv):
        with pytest.raises(SystemExit) as exit:
            cli.main(list(argv))
        out, err = capsys.readouterr()
        return exit.value.code, out, err

    return run


def test_the_microscope_says_who_it_is_and_its_state(run):
    code, out, _ = run("microscope", "get-info")
    assert code == 0 and "DemoMicroscope" in out
    code, out, _ = run("microscope", "get-state")
    assert code == 0 and "Stage Position" in out
    assert "ELECTRON" in out and "blanked:" in out


def test_a_relative_move_moves_by_the_offset_in_um(run, microscope):
    before = microscope.get_stage_position()
    code, _, _ = run("stage", "move-relative", "--x", "10")
    assert code == 0
    assert math.isclose(microscope.get_stage_position().x - before.x, 10e-6)


def test_an_absolute_move_is_the_safe_move_and_keeps_axes_left_out(
    run, microscope, monkeypatch
):
    """The route the agent's stage_move_absolute takes, not the raw stage."""
    from functools import wraps

    calls = []
    real = microscope.stage_movement.safe_absolute_stage_movement

    @wraps(real)
    def spy(stage_position):
        calls.append(stage_position)
        return real(stage_position)

    monkeypatch.setattr(microscope.stage_movement, "safe_absolute_stage_movement", spy)
    before = microscope.get_stage_position()
    code, out, _ = run("stage", "move-absolute", "--x", "25")
    assert code == 0, out
    assert len(calls) == 1
    after = microscope.get_stage_position()
    assert math.isclose(after.x, 25e-6)
    assert math.isclose(after.t, before.t) and math.isclose(after.y, before.y)


def test_a_move_needs_an_axis(run):
    code, _, err = run("stage", "move-relative")
    assert code == 2 and "at least one of" in err


def test_set_beam_writes_with_the_servers_checks(run, microscope):
    code, out, _ = run("beam", "set-parameters", "--hfw", "100")
    assert code == 0
    assert math.isclose(microscope.devices["electron"].hfw.get_value(), 100e-6)
    # Out of limits: clipped, and said so.
    code, out, _ = run("beam", "set-parameters", "--hfw", "1e9")
    assert code == 0 and "clipped" in out
    # Not one of the choices: refused with the beam's own message.
    code, _, err = run("beam", "set-parameters", "--voltage", "-5")
    assert code == 1 and "voltage" in err and "not one of" in err


def test_beam_blanks_and_unblanks(run, microscope):
    assert run("beam", "blank")[0] == 0
    assert microscope.devices["electron"].blanked.get_value() is True
    assert run("beam", "unblank")[0] == 0
    assert microscope.devices["electron"].blanked.get_value() is False


def test_acquire_saves_here_when_asked(run, tmp_path):
    code, out, _ = run(
        "beam",
        "acquire-image",
        "--beam-type",
        "both",
        "--resolution",
        "64x64",
        "--hfw",
        "80",
        "--out",
        str(tmp_path),
    )
    assert code == 0, out
    assert "ELECTRON" in out and "ION" in out
    saved = FibsemImage.load(str(tmp_path / "image_eb.tif"))
    assert math.isclose(saved.metadata.image_settings.hfw, 80e-6)
    assert (tmp_path / "image_ib.tif").is_file()


def test_autofocus_and_autocontrast(run):
    assert run("beam", "auto-focus")[0] == 0
    code, out, _ = run("beam", "auto-contrast", "--beam-type", "both")
    assert code == 0 and out.count("contrast:") == 2


def test_mill_angle(run):
    code, out, _ = run("stage", "move-to-milling-angle", "15")
    assert code == 0 and "Stage Position" in out


def test_device_and_service_by_name(run, microscope):
    code, out, _ = run("device", "list")
    assert code == 0 and "electron" in out and "stage" in out
    code, out, _ = run("device", "describe", "electron")
    assert code == 0 and "working_distance" in out and "commands:" in out
    assert run("device", "set-parameter", "ion", "hfw", "2e-4")[0] == 0
    assert math.isclose(microscope.devices["ion"].hfw.get_value(), 2e-4)
    code, out, _ = run("device", "get-parameter", "ion", "hfw")
    assert code == 0 and json.loads(out) == pytest.approx(2e-4)
    code, out, _ = run(
        "service",
        "call-command",
        "stage_movement",
        "move_relative",
        "--kwargs",
        json.dumps({"delta": DELTA}),
    )
    assert code == 0 and "Stage Position" in out


def test_kwargs_from_a_file(run, tmp_path, microscope):
    path = tmp_path / "move.yaml"
    path.write_text(yaml.safe_dump({"delta": {**DELTA, "x": 0.0, "y": 2e-6}}))
    before = microscope.get_stage_position()
    code, _, _ = run(
        "service",
        "call-command",
        "stage_movement",
        "move_relative",
        "--kwargs",
        f"@{path}",
    )
    assert code == 0
    assert math.isclose(microscope.get_stage_position().y - before.y, 2e-6)


def test_beam_turns_off_and_on(run, microscope):
    code, out, _ = run("beam", "turn-off", "--beam-type", "ion")
    assert code == 0 and "ION: off" in out
    assert microscope.devices["ion"].on.get_value() is False
    code, out, _ = run("beam", "turn-on", "--beam-type", "ion")
    assert code == 0 and "ION: on" in out
    assert microscope.devices["ion"].on.get_value() is True


def test_what_the_agent_may_never_do_the_cli_may_not_either(run):
    """The blocklist holds for the operator token (#2121)."""
    code, _, err = run("device", "set-parameter", "electron", "on", "false")
    assert code == 1 and "not_allowed" in err
    code, _, err = run("device", "call-command", "chamber", "vent")
    assert code == 1 and "not_allowed" in err


def _experiment(root, name, serial):
    """An experiment.yaml as the app writes one, run on instrument *serial*."""
    import time

    import yaml

    directory = root / name
    directory.mkdir(parents=True)
    record = {
        "name": directory.name,
        "path": str(directory),
        "created_at": time.time(),
        "positions": [{}],
        "session": {
            "recorded_at": time.time(),
            "system": {"serial_number": serial, "model": "Arctis"},
            "user": {"name": "pat"},
            "plugins": {},
        },
    }
    (directory / "experiment.yaml").write_text(yaml.safe_dump(record))


def test_experiments_lists_the_apps_experiments_folder(
    run, session, tmp_path, monkeypatch
):
    monkeypatch.setattr(session, "experiments_folder", str(tmp_path))
    _experiment(tmp_path, "lab/first", "SN-1")
    _experiment(tmp_path, "second", "SN-2")
    code, out, _ = run("experiment", "list-all")
    assert code == 0, out
    assert "lab/first" in out and "Arctis/SN-1" in out and "pat" in out
    assert f"2 experiment(s) in {tmp_path}" in out
    code, out, _ = run("experiment", "list-all", "--instrument", "SN-2")
    assert "second" in out and "lab/first" not in out
    code, out, _ = run("experiment", "list-all", "--since", "2999-01-01")
    assert code == 0 and "No experiments found" in out
    code, out, _ = run("experiment", "list-all", "--group")
    assert code == 0 and "Arctis (SN-1)" in out and "Arctis (SN-2)" in out


def test_no_server_says_what_to_start(monkeypatch, capsys):
    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_OPERATOR_TOKEN", raising=False)
    monkeypatch.setattr("fibsem.client.transport.read_discovery_file", lambda: None)
    with pytest.raises(SystemExit) as exit:
        cli.main(["stage", "get-position"])
    assert exit.value.code == 1
    assert "fibsem-autolamella-server" in capsys.readouterr().err


def test_the_operator_token_never_the_agents(monkeypatch, tmp_path):
    """With no token given, the operator token file; the discovery file's token
    is the agent's and is never used."""
    from fibsem.client.transport import resolve_operator

    monkeypatch.delenv("FIBSEM_SERVER_URL", raising=False)
    monkeypatch.delenv("FIBSEM_OPERATOR_TOKEN", raising=False)
    monkeypatch.setattr(
        "fibsem.client.transport.read_discovery_file",
        lambda: {"url": "http://pc:8001", "token": "agent"},
    )
    monkeypatch.setattr("fibsem.client.transport.read_operator_token", lambda: None)
    assert resolve_operator() is None
    monkeypatch.setattr("fibsem.client.transport.read_operator_token", lambda: "op")
    assert resolve_operator() == ("http://pc:8001", "op")
    assert resolve_operator("http://other", "given") == ("http://other", "given")


def test_reading_the_operator_token_never_makes_one(tmp_path):
    from fibsem.server.operator_token import load_operator_token, read_operator_token

    path = tmp_path / "operator-token.json"
    assert read_operator_token(path) is None
    assert not path.exists()
    made = load_operator_token(path)
    assert read_operator_token(path) == made


def test_a_server_not_serving_the_microscope_says_where_to_turn_it_on(
    microscope, monkeypatch, capsys
):
    from fastapi.testclient import TestClient

    auth = AuthConfig.generate(level="watch", token="agent-token")
    auth.operator_token = OPERATOR
    app = build_server(microscope, auth=auth, serve_microscope=False)
    with TestClient(app) as client:
        detail = client.get(
            "/microscope/devices", headers={"Authorization": f"Bearer {OPERATOR}"}
        ).json()
    from fibsem.client.transport import error_for

    said = cli._said(error_for("GET /microscope/devices", 404, detail["detail"]))
    assert "Tools > App Server" in said


def test_lengths_are_um_on_the_command_line():
    args = cli._build_parser().parse_args(["beam", "acquire-image"])
    assert args.hfw == 150.0 and args.dwell_time == 1.0

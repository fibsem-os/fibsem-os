"""The fluorescence microscope on the imaging service (map 8d): an ``fm`` role
next to the beams, ``fm_focus_sweep`` as a long command, and one ``stop`` that
ends it. The FM's own ``cancel`` is the imaging stop's for the agent.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

import fibsem.config as fconfig  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.autofunctions.autofocus import (  # noqa: E402
    AutoFocusSettings,
    FocusSweepPass,
)
from fibsem.fm.microscope import AcquisitionRefused  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402

TOKEN = "test-token"
IMAGING = "/microscope/services/imaging"
# One short pass, so a sweep on the sim takes a few frames.
SHORT = AutoFocusSettings(passes=[FocusSweepPass(search_range=2e-6, step_size=1e-6)])


@pytest.fixture(scope="module")
def microscope(tmp_path_factory):
    """A Demo microscope with an FM, the sample scene off (noise frames)."""
    with open(os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path_factory.mktemp("fm-config") / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    assert microscope.fm is not None
    microscope.move_to_orientation("FM")  # where the FM can see the sample
    return microscope


@pytest.fixture
def experiment(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    return folder


@pytest.fixture
def client(microscope, experiment):
    context = SimpleNamespace(experiment_path=str(experiment))
    app = build_server(
        microscope,
        app_context=context,
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


def _sweep(client, kwargs):
    resp = client.post(f"{IMAGING}/commands/fm_focus_sweep", json={"kwargs": kwargs})
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    return client.get(
        f"{IMAGING}/run", params={"run_id": run["run_id"], "wait": 60}
    ).json()["run"]


def test_the_fm_joins_the_imaging_service(microscope, client):
    assert microscope.imaging.fm is microscope.fm_devices["fm"]
    command = client.get("/microscope/services").json()["imaging"]["commands"][
        "fm_focus_sweep"
    ]
    assert command["long"] is True and command["available"] is True


def test_fm_focus_sweep_runs_on_and_saves_inside_the_experiment(client, experiment):
    run = _sweep(
        client, {"settings": SHORT.to_dict(), "path": str(experiment / "focus")}
    )
    assert run["state"] == "done", run
    assert run["result"]["status"] == "completed"
    assert "working_distance" in run["result"]
    assert os.path.isdir(experiment / "focus" / "autofocus")


def test_a_path_outside_the_experiment_is_refused(client, tmp_path):
    resp = client.post(
        f"{IMAGING}/commands/fm_focus_sweep",
        json={"kwargs": {"path": str(tmp_path / "elsewhere")}},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_stop_ends_the_sweep_cancelled(microscope):
    imaging = microscope.imaging
    group = microscope.fm_devices["fm"]
    acquire = group.acquire_frame
    frames = threading.Event()

    def stopping_frame(*args, **kwargs):
        frames.set()
        imaging.stop()
        return acquire(*args, **kwargs)

    group.acquire_frame = stopping_frame
    try:
        result = imaging.fm_focus_sweep(settings=SHORT)
    finally:
        del group.acquire_frame
    assert frames.is_set()
    assert result["status"] == "cancelled"
    assert not microscope.fm.is_acquiring


def test_the_sweep_is_refused_while_the_fm_is_in_use(microscope):
    microscope.fm.set_acquiring(True, "overview acquisition")
    try:
        with pytest.raises(AcquisitionRefused, match="overview acquisition"):
            microscope.imaging.fm_focus_sweep(settings=SHORT)
    finally:
        microscope.fm.set_acquiring(False)


def test_the_sweep_is_refused_while_the_fm_device_is_live(microscope):
    """Live view started on the device itself, as an agent starts it, is not
    the FM tab's stream, which is all `acquiring` sees."""
    device = microscope.fm_devices["fm"]
    device.start_live()
    try:
        assert not microscope.fm.is_acquiring
        with pytest.raises(AcquisitionRefused, match="live view"):
            microscope.imaging.fm_focus_sweep(settings=SHORT)
    finally:
        device.stop_live()


def test_the_sweep_is_refused_away_from_the_fm_pose(microscope):
    """Acquiring from a beam pose is allowed (NEEDS_REPOSE), but a focus sweep
    there sweeps the objective where it cannot see the sample."""
    microscope.move_to_orientation("SEM")
    try:
        with pytest.raises(AcquisitionRefused, match="Cannot start auto-focus"):
            microscope.imaging.fm_focus_sweep(settings=SHORT)
    finally:
        microscope.move_to_orientation("FM")


def test_the_agent_stops_the_fm_through_imaging(client):
    resp = client.post(
        "/microscope/devices/fm/commands/cancel",
        json={"kwargs": {}, "reason": "testing"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"

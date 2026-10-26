"""The fluorescence microscope on the imaging service (map 8d): an ``fm`` role
next to the beams, ``fm_auto_focus`` and ``fm_acquire_image`` as long commands,
and one ``stop`` that ends them. The FM's own ``cancel`` is the imaging stop's for the agent.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import yaml
from fastapi.testclient import TestClient

import fibsem.config as fconfig
from fibsem import utils
from fibsem.autofunctions.autofocus import (
    AutoFocusSettings,
    FocusSweepPass,
)
from fibsem.fm.microscope import AcquisitionRefused
from fibsem.fm.structures import ChannelSettings, ZParameters
from fibsem.server import AuthConfig, build_server

TOKEN = "test-token"
IMAGING = "/microscope/services/imaging"
# One short pass, so a sweep on the sim takes a few frames.
SHORT = AutoFocusSettings(passes=[FocusSweepPass(search_range=2e-6, step_size=1e-6)])
CHANNELS = [ChannelSettings(name="GFP", exposure_time=0.01)]
THREE_PLANES = ZParameters(zmin=-1e-6, zmax=1e-6, zstep=1e-6)


@pytest.fixture(scope="module")
def microscope(tmp_path_factory):
    """A Demo microscope with an FM, the sample scene off (noise frames)."""
    with open(
        os.path.join(fconfig.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml")
    ) as f:
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
    resp = client.post(f"{IMAGING}/commands/fm_auto_focus", json={"kwargs": kwargs})
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    return client.get(
        f"{IMAGING}/run", params={"run_id": run["run_id"], "wait": 60}
    ).json()["run"]


def test_the_fm_joins_the_imaging_service(microscope, client):
    assert microscope.imaging.fm is microscope.fm_devices["fm"]
    command = client.get("/microscope/services").json()["imaging"]["commands"][
        "fm_auto_focus"
    ]
    assert command["long"] is True and command["available"] is True


def test_fm_focus_sweep_runs_on_and_saves_inside_the_experiment(
    client, microscope, experiment
):
    start = microscope.fm.objective.position
    run = _sweep(
        client, {"settings": SHORT.to_dict(), "path": str(experiment / "focus")}
    )
    assert run["state"] == "done", run
    # Noise frames score flat: no focus found, and the objective goes back
    # rather than to the noisiest frame (FIB-1519).
    assert run["result"]["status"] == "no_focus"
    assert run["result"]["found"] is False
    assert run["result"]["working_distance"] == pytest.approx(start)
    assert microscope.fm.objective.position == pytest.approx(start)
    assert os.path.isdir(experiment / "focus" / "autofocus")


def test_a_path_outside_the_experiment_is_refused(client, tmp_path):
    resp = client.post(
        f"{IMAGING}/commands/fm_auto_focus",
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
        result = imaging.fm_auto_focus(settings=SHORT)
    finally:
        del group.acquire_frame
    assert frames.is_set()
    assert result["status"] == "cancelled"
    assert not microscope.fm.is_acquiring


def test_the_sweep_is_refused_while_the_fm_is_in_use(microscope):
    microscope.fm.set_acquiring(True, "overview acquisition")
    try:
        with pytest.raises(AcquisitionRefused, match="overview acquisition"):
            microscope.imaging.fm_auto_focus(settings=SHORT)
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
            microscope.imaging.fm_auto_focus(settings=SHORT)
    finally:
        device.stop_live()


def test_the_sweep_is_refused_away_from_the_fm_pose(microscope):
    """Acquiring from a beam pose is allowed (NEEDS_REPOSE), but a focus sweep
    there sweeps the objective where it cannot see the sample."""
    microscope.move_to_orientation("SEM")
    try:
        with pytest.raises(AcquisitionRefused, match="Cannot start auto-focus"):
            microscope.imaging.fm_auto_focus(settings=SHORT)
    finally:
        microscope.move_to_orientation("FM")


def test_the_agent_stops_the_fm_through_imaging(client):
    resp = client.post(
        "/microscope/devices/fm/commands/cancel",
        json={"kwargs": {}, "reason": "testing"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"


def _acquire(client, kwargs):
    resp = client.post(f"{IMAGING}/commands/fm_acquire_image", json={"kwargs": kwargs})
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    return client.get(
        f"{IMAGING}/run", params={"run_id": run["run_id"], "wait": 60}
    ).json()["run"]


def test_an_fm_image_runs_on_and_saves_inside_the_experiment(client, experiment):
    folder = experiment / "fm"
    run = _acquire(
        client,
        {
            "channels": [c.to_dict() for c in CHANNELS],
            "path": str(folder),
            "name": "gfp",
        },
    )
    assert run["state"] == "done", run
    result = run["result"]
    assert result["status"] == "completed"
    image = result["image"]
    assert os.path.basename(image["path"]).startswith("gfp-")
    assert image["path"].startswith(str(folder)) and os.path.isfile(image["path"])
    assert image["summary"]["channels"] == ["GFP"]
    assert "substituted" not in result  # 550 nm is one of the Demo's bands


def test_an_fm_image_says_when_a_wavelength_was_substituted(client, experiment):
    """FIB-1520: GFP asked at 488 nm was taken at the Demo's nearest, 450 nm,
    and the answer said nothing."""
    gfp = ChannelSettings(name="GFP", excitation_wavelength=488, exposure_time=0.01)
    run = _acquire(
        client, {"channels": [gfp.to_dict()], "path": str(experiment / "fm")}
    )
    assert run["state"] == "done", run
    (note,) = run["result"]["substituted"]
    assert note.startswith("GFP: 488 nm") and "450 nm" in note


def test_an_fm_z_stack_takes_its_planes_and_puts_the_objective_back(
    client, microscope, experiment
):
    objective = microscope.fm.objective.position
    run = _acquire(
        client,
        {
            "channels": [c.to_dict() for c in CHANNELS],
            "path": str(experiment / "fm"),
            "zparams": THREE_PLANES.to_dict(),
        },
    )
    assert run["state"] == "done", run
    shape = run["result"]["image"]["summary"]["shape"]
    assert 3 in shape[:-2]  # the planes, before the frame's height and width
    assert microscope.fm.objective.position == pytest.approx(objective)
    assert not microscope.fm.is_acquiring


def test_an_fm_image_outside_the_experiment_is_refused(client, tmp_path):
    resp = client.post(
        f"{IMAGING}/commands/fm_acquire_image",
        json={
            "kwargs": {
                "channels": [c.to_dict() for c in CHANNELS],
                "path": str(tmp_path / "elsewhere"),
            }
        },
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_stop_ends_an_fm_z_stack_cancelled(microscope, tmp_path):
    imaging = microscope.imaging
    group = microscope.fm_devices["fm"]
    acquire = group.acquire_frame

    def stopping_frame(*args, **kwargs):
        imaging.stop()
        return acquire(*args, **kwargs)

    group.acquire_frame = stopping_frame
    try:
        result = imaging.fm_acquire_image(CHANNELS, str(tmp_path), zparams=THREE_PLANES)
    finally:
        del group.acquire_frame
    assert result == {"status": "cancelled", "image": None}
    assert not microscope.fm.is_acquiring
    assert not any(name.endswith(".ome.tiff") for name in os.listdir(tmp_path))


def test_an_fm_image_is_refused_while_the_fm_is_in_use(microscope, tmp_path):
    microscope.fm.set_acquiring(True, "overview acquisition")
    try:
        with pytest.raises(AcquisitionRefused, match="overview acquisition"):
            microscope.imaging.fm_acquire_image(CHANNELS, str(tmp_path))
    finally:
        microscope.fm.set_acquiring(False)


def _frame(microscope, color: str):
    channel = ChannelSettings(name="GFP", exposure_time=0.01, color=color)
    microscope.fm.set_channel(channel)
    frame = microscope.fm.acquire_image(channel_settings=channel)
    frame.metadata.channels[0].color = color
    return frame


def test_the_session_keeps_the_fm_s_latest_frame_until_unfollowed(microscope):
    from fibsem.workflows.session import AppSession

    session = AppSession()
    session.microscope = microscope
    frame = _frame(microscope, "green")
    microscope.fm.acquisition_signal.emit(frame)
    assert session.images.get_fm() is frame

    session.microscope = None
    microscope.fm.acquisition_signal.emit(_frame(microscope, "green"))
    assert session.images.get_fm() is frame


def test_the_display_images_answer_the_fm_frame_in_its_colour(microscope):
    import base64
    import io

    import numpy as np
    from PIL import Image

    from fibsem.server.app_context import AppContext
    from fibsem.workflows.session import AppSession

    session = AppSession()
    session.microscope = microscope
    context = AppContext(session)
    assert context.display_images().fm_available
    assert context.display_images().fm is None

    frame = _frame(microscope, "green")
    microscope.fm.acquisition_signal.emit(frame)
    fm = context.display_images().fm
    assert fm.full_width == frame.data.shape[-1]
    assert fm.pixelsize.x == frame.metadata.pixel_size_x
    assert fm.hfw == pytest.approx(frame.metadata.pixel_size_x * fm.full_width)
    assert fm.acquired_at is not None
    assert fm.info["kind"] == "FM"
    assert fm.info["scalebar"]["fraction"] > 0
    jpeg = Image.open(io.BytesIO(base64.b64decode(fm.image_b64_jpeg)))
    red, green, blue = np.asarray(jpeg.convert("RGB"), float).mean(axis=(0, 1))
    assert green > 4 * max(red, blue)
    session.microscope = None


def test_fm_frames_reach_the_events_one_a_second_at_most(microscope):
    from fibsem.workflows.event_recording import LIVE_KINDS
    from fibsem.workflows.events import EventBuffer, attach_microscope_taps

    buffer = EventBuffer()
    disposers = attach_microscope_taps(buffer, microscope)
    try:
        frame = _frame(microscope, "green")
        for _ in range(5):
            microscope.fm.acquisition_signal.emit(frame)
        frames = [
            e for e in buffer.events_since(0)["events"] if e["kind"] == "fm_acquisition"
        ]
        assert len(frames) == 1
        assert frames[0]["payload"]["channels"] == ["GFP"]
        assert "fm_acquisition" in LIVE_KINDS
    finally:
        for dispose in disposers:
            dispose()

"""Tiled overviews and focus stacks through the App Server (map 8e): the imaging
service's ``acquire_overview`` and ``acquire_focus_stack`` run on as long commands and save
inside the open experiment, ``stop`` ends them cancelled, and
``estimate_overview`` answers the tile count and scan time.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    ImageSettings,
    OverviewAcquisitionSettings,
)

TOKEN = "test-token"
IMAGING = "/microscope/services/imaging"


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


def _image_settings() -> ImageSettings:
    return ImageSettings(
        beam_type=BeamType.ELECTRON,
        resolution=[64, 64],
        hfw=50e-6,
        dwell_time=1e-9,
        filename="stack",
    )


def _overview() -> OverviewAcquisitionSettings:
    return OverviewAcquisitionSettings(
        image_settings=_image_settings(), nrows=2, ncols=2, overlap=0.0
    )


def _wait(client, run_id):
    return client.get(f"{IMAGING}/run", params={"run_id": run_id, "wait": 30}).json()[
        "run"
    ]


def test_an_overview_runs_on_and_saves_inside_the_experiment(
    client, microscope, experiment
):
    start = microscope.get_stage_position()
    folder = experiment / "overviews"
    resp = client.post(
        f"{IMAGING}/commands/acquire_overview",
        json={
            "kwargs": {
                "settings": _overview().to_dict(),
                "path": str(folder),
                "name": "ov",
            }
        },
    )
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    ended = _wait(client, run["run_id"])
    assert ended["state"] == "done", ended
    result = ended["result"]
    assert result["status"] == "completed"
    image = result["image"]
    assert image["path"].startswith(str(folder))
    assert os.path.basename(image["path"]).startswith("ov-")
    assert os.path.isfile(image["path"])
    assert image["preview"] and image["summary"]["kind"] == "image"
    assert len(result["tile_paths"]) == 4
    assert all(os.path.isfile(path) for path in result["tile_paths"])
    # Not asked to register: the tiles stay as planned, and the record says so.
    overview = result["overview"]
    assert overview["registration"]["requested"] is False
    assert overview["registration"]["applied"] is False
    assert overview["tiles_acquired"] == overview["tiles_total"]
    assert os.path.isfile(overview["paths"]["record"])
    # The stage comes back to where it started.
    end = microscope.get_stage_position()
    assert abs(end.x - start.x) < 1e-9 and abs(end.y - start.y) < 1e-9


def test_an_overview_says_what_tile_registration_did(client, experiment):
    settings = _overview()
    settings.register_tiles = True
    resp = client.post(
        f"{IMAGING}/commands/acquire_overview",
        json={
            "kwargs": {
                "settings": settings.to_dict(),
                "path": str(experiment / "overviews"),
                "name": "reg",
            }
        },
    )
    assert resp.status_code == 200, resp.text
    ended = _wait(client, resp.json()["result"]["run_id"])
    assert ended["state"] == "done", ended
    registration = ended["result"]["overview"]["registration"]
    assert registration["requested"] is True
    assert isinstance(registration["applied"], bool)
    assert registration["note"]


def test_a_path_outside_the_experiment_is_refused(client, tmp_path):
    for command, kwargs in (
        ("acquire_overview", {"settings": _overview().to_dict()}),
        ("acquire_focus_stack", {"image_settings": _image_settings().to_dict()}),
    ):
        resp = client.post(
            f"{IMAGING}/commands/{command}",
            json={"kwargs": {**kwargs, "path": str(tmp_path / "elsewhere")}},
        )
        assert resp.status_code == 403, command
        assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_stop_ends_an_overview_cancelled(microscope, tmp_path):
    imaging = microscope.imaging
    acquire = imaging.acquire_image
    tiles = []

    def stopping_acquire(*args, **kwargs):
        tiles.append(1)
        imaging.stop()
        return acquire(*args, **kwargs)

    imaging.acquire_image = stopping_acquire
    try:
        result = imaging.acquire_overview(_overview(), str(tmp_path))
    finally:
        del imaging.acquire_image
    assert result["status"] == "cancelled"
    assert result["image"] is None
    assert len(tiles) == 1 and len(result["tile_paths"]) == 1


def test_the_estimate_counts_tiles_and_scan_time(client):
    settings = _overview()
    settings.tile_mask = [[True, False], [True, True]]
    resp = client.post(
        f"{IMAGING}/queries/estimate_overview",
        json={"kwargs": {"settings": settings.to_dict()}},
    )
    assert resp.status_code == 200, resp.text
    estimate = resp.json()["result"]
    assert estimate["tiles"] == 3
    assert estimate["scan_time"] == pytest.approx(settings.scan_time)
    assert estimate["fov"][0] == pytest.approx(100e-6)


def test_the_estimate_says_the_resolution_scanned_and_the_tiles_out_of_reach(
    client, microscope, monkeypatch
):
    """A size the beam doesn't scan comes back as the one it does, and a grid
    wider than the stage's travel names the tiles it can't reach, before anything
    moves."""
    beam = microscope.beams[BeamType.ELECTRON]
    monkeypatch.setattr(beam, "scan_resolution", lambda resolution: (64, 48))
    settings = _overview()
    settings.nrows, settings.ncols = 1, 3
    # The demo stage travels 100 mm each way: 150 mm tiles put the outer two past it.
    settings.image_settings.hfw = 0.15
    start = microscope.get_stage_position()
    resp = client.post(
        f"{IMAGING}/queries/estimate_overview",
        json={"kwargs": {"settings": settings.to_dict()}},
    )
    assert resp.status_code == 200, resp.text
    estimate = resp.json()["result"]
    assert estimate["resolution"] == [64, 48]
    assert estimate["requested_resolution"] == [64, 64]
    assert estimate["unreachable"] == [[0, 0], [0, 2]]
    end = microscope.get_stage_position()
    assert abs(end.x - start.x) < 1e-12 and abs(end.y - start.y) < 1e-12


def test_a_focus_stack_saves_inside_the_experiment(client, experiment):
    resp = client.post(
        f"{IMAGING}/commands/acquire_focus_stack",
        json={
            "kwargs": {
                "image_settings": _image_settings().to_dict(),
                "path": str(experiment),
                "n_steps": 2,
                "auto_focus": False,
            }
        },
    )
    assert resp.status_code == 200, resp.text
    ended = _wait(client, resp.json()["result"]["run_id"])
    assert ended["state"] == "done", ended
    image = ended["result"]["image"]
    assert ended["result"]["status"] == "completed"
    assert image["path"].startswith(str(experiment / "stack_eb"))
    assert os.path.isfile(image["path"])


def test_stop_ends_a_focus_stack_cancelled(microscope, tmp_path):
    imaging = microscope.imaging
    acquire = imaging.acquire_image

    def stopping_acquire(*args, **kwargs):
        imaging.stop()
        return acquire(*args, **kwargs)

    imaging.acquire_image = stopping_acquire
    try:
        result = imaging.acquire_focus_stack(
            _image_settings(), str(tmp_path), n_steps=3, auto_focus=False
        )
    finally:
        del imaging.acquire_image
    assert result == {"status": "cancelled", "image": None}
    assert not list(tmp_path.iterdir())


def test_a_stop_survives_the_focus_sweep_inside_a_focus_stack(microscope, tmp_path):
    # A stop that lands just as a strip's sweep starts: the sweep starts a
    # routine of its own, and must not clear a stop meant for the whole stack.
    from fibsem.autofunctions.autofocus import AutoFocusSettings, FocusSweepPass

    imaging = microscope.imaging
    one_step = AutoFocusSettings(
        passes=[FocusSweepPass(search_range=1e-6, step_size=1e-6)]
    )

    def stop_then_sweep(beam_type, reduced_area=None):
        imaging.stop()
        return imaging.focus_sweep(beam_type=beam_type, settings=one_step)

    imaging.auto_focus = stop_then_sweep
    try:
        result = imaging.acquire_focus_stack(
            _image_settings(), str(tmp_path), n_steps=3, auto_focus=True
        )
    finally:
        del imaging.auto_focus
    assert result == {"status": "cancelled", "image": None}

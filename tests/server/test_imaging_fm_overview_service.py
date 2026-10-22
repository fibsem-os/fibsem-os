"""Fluorescence overviews through the App Server (map 8f): the imaging service's
``fm_acquire_overview`` runs on as a long command and saves inside the open
experiment, ``stop`` ends it cancelled, it is refused while the FM is in use,
and ``fm_estimate_overview`` answers the tiles and the run's length.
``fm_plan_sparse_overview`` answers the plan the FM overview's sparse selection
makes from the same regions of a saved beam overview.
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
from fibsem.fm.microscope import AcquisitionRefused  # noqa: E402
from fibsem.fm.planning import plan_sparse_fm_overview  # noqa: E402
from fibsem.fm.structures import (  # noqa: E402
    AutoFocusMode,
    ChannelSettings,
    ObjectiveStartPosition,
    OverviewParameters,
)
from fibsem.fm.timing import estimate_fm_overview_time  # noqa: E402
from fibsem.geometry.projection import (  # noqa: E402
    BeamStageProjection,
    FMStageProjection,
)
from fibsem.imaging.tiling.geometry import PlaneRegion  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    ImageSettings,
    InsertableDeviceState,
)

TOKEN = "test-token"
IMAGING = "/microscope/services/imaging"
CHANNELS = [ChannelSettings(name="GFP", exposure_time=0.01)]
ONE_BY_TWO = OverviewParameters(rows=1, cols=2, overlap=0.1)


@pytest.fixture(scope="module")
def microscope(tmp_path_factory):
    """A Demo microscope with an FM, the sample scene off (noise frames), at the
    FM pose with the objective in."""
    with open(os.path.join(fconfig.CONFIG_PATH, "sim-arctis-configuration.yaml")) as f:
        config = yaml.safe_load(f)
    config.setdefault("sim", {}).setdefault("sample", {})["enabled"] = False
    path = tmp_path_factory.mktemp("fm-config") / "sim-arctis-configuration.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    microscope, _ = utils.setup_session(manufacturer="Demo", config_path=str(path))
    assert microscope.fm is not None
    microscope.move_to_orientation("FM")
    if microscope.fm.objective.state is not InsertableDeviceState.INSERTED:
        microscope.fm.objective.insert()
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


def _kwargs(**extra):
    return {
        "channels": [c.to_dict() for c in CHANNELS],
        "parameters": ONE_BY_TWO.to_dict(),
        **extra,
    }


def test_the_command_is_listed_as_long(client):
    command = client.get("/microscope/services").json()["imaging"]["commands"][
        "fm_acquire_overview"
    ]
    assert command["long"] is True and command["available"] is True


def test_an_overview_runs_on_and_saves_inside_the_experiment(
    client, microscope, experiment
):
    start = microscope.get_stage_position()
    objective = microscope.fm.objective.position
    folder = experiment / "overviews"
    resp = client.post(
        f"{IMAGING}/commands/fm_acquire_overview",
        json={"kwargs": _kwargs(path=str(folder), name="fm")},
    )
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    ended = client.get(
        f"{IMAGING}/run", params={"run_id": run["run_id"], "wait": 60}
    ).json()["run"]
    assert ended["state"] == "done", ended
    result = ended["result"]
    assert result["status"] == "completed"
    image = result["image"]
    assert image["path"].startswith(str(folder))
    assert os.path.basename(image["path"]).startswith("fm-")
    assert os.path.isfile(image["path"])
    assert image["summary"]["kind"] == "fluorescence_image"
    assert image["summary"]["channels"] == ["GFP"]
    assert len(result["tile_paths"]) == 2
    assert all(
        os.path.isfile(p) and p.startswith(str(folder)) for p in result["tile_paths"]
    )
    # The stage and objective come back to where they started, and the FM is free.
    end = microscope.get_stage_position()
    assert abs(end.x - start.x) < 1e-9 and abs(end.y - start.y) < 1e-9
    assert microscope.fm.objective.position == pytest.approx(objective)
    assert not microscope.fm.is_acquiring
    # Not asked to register: the tiles stay as planned, and the record says so.
    overview = result["overview"]
    assert overview["registration"]["requested"] is False
    assert overview["registration"]["applied"] is False
    assert overview["tiles_acquired"] == overview["tiles_total"]
    assert os.path.isfile(overview["paths"]["record"])


def test_an_overview_says_what_tile_registration_did(client, experiment):
    parameters = ONE_BY_TWO.to_dict()
    parameters["register_tiles"] = True
    resp = client.post(
        f"{IMAGING}/commands/fm_acquire_overview",
        json={
            "kwargs": _kwargs(
                parameters=parameters, path=str(experiment / "overviews"), name="reg"
            )
        },
    )
    assert resp.status_code == 200, resp.text
    ended = client.get(
        f"{IMAGING}/run", params={"run_id": resp.json()["result"]["run_id"], "wait": 60}
    ).json()["run"]
    assert ended["state"] == "done", ended
    registration = ended["result"]["overview"]["registration"]
    assert registration["requested"] is True
    assert isinstance(registration["applied"], bool)
    assert registration["note"]


def test_a_path_outside_the_experiment_is_refused(client, tmp_path):
    resp = client.post(
        f"{IMAGING}/commands/fm_acquire_overview",
        json={"kwargs": _kwargs(path=str(tmp_path / "elsewhere"))},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_stop_ends_an_overview_cancelled(microscope, tmp_path):
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
        result = imaging.fm_acquire_overview(CHANNELS, ONE_BY_TWO, str(tmp_path))
    finally:
        del group.acquire_frame
    assert frames.is_set()
    assert result["status"] == "cancelled"
    assert result["image"] is None
    assert len(result["tile_paths"]) <= 1
    assert not microscope.fm.is_acquiring


def test_an_overview_is_refused_while_the_fm_is_in_use(microscope, tmp_path):
    microscope.fm.set_acquiring(True, "z-stack")
    try:
        with pytest.raises(AcquisitionRefused, match="z-stack"):
            microscope.imaging.fm_acquire_overview(CHANNELS, ONE_BY_TWO, str(tmp_path))
    finally:
        microscope.fm.set_acquiring(False)
    assert not list(tmp_path.iterdir())


def test_the_estimate_is_the_runs_length(client):
    parameters = OverviewParameters(
        rows=2,
        cols=2,
        tile_mask=[[True, False], [True, True]],
        autofocus_mode=AutoFocusMode.ONCE,
        objective_start=ObjectiveStartPosition.CURRENT,
    )
    resp = client.post(
        f"{IMAGING}/queries/fm_estimate_overview",
        json={
            "kwargs": {
                "channels": [c.to_dict() for c in CHANNELS],
                "parameters": parameters.to_dict(),
            }
        },
    )
    assert resp.status_code == 200, resp.text
    estimate = resp.json()["result"]
    assert estimate["tiles"] == 3 and estimate["images"] == 3
    from fibsem.autofunctions.autofocus import AutoFocusSettings

    expected = estimate_fm_overview_time(
        CHANNELS, parameters, None, AutoFocusSettings()
    )
    assert estimate["duration"] == pytest.approx(expected)
    # The focus sweep counts: more than the same grid without it.
    parameters.autofocus_mode = AutoFocusMode.NONE
    assert estimate["duration"] > estimate_fm_overview_time(CHANNELS, parameters)


def test_the_estimate_names_the_tiles_out_of_reach(client, microscope):
    """The Arctis stage travels well under a grid's width from its centre, so a
    long row of camera fields runs off both ends; the middle stays in reach."""
    start = microscope.get_stage_position()
    resp = client.post(
        f"{IMAGING}/queries/fm_estimate_overview",
        json={
            "kwargs": {
                "channels": [c.to_dict() for c in CHANNELS],
                "parameters": OverviewParameters(rows=1, cols=51).to_dict(),
            }
        },
    )
    assert resp.status_code == 200, resp.text
    unreachable = resp.json()["result"]["unreachable"]
    assert [0, 0] in unreachable and [0, 50] in unreachable
    assert [0, 25] not in unreachable
    end = microscope.get_stage_position()
    assert abs(end.x - start.x) < 1e-12 and abs(end.y - start.y) < 1e-12
    # A small grid is all in reach.
    resp = client.post(
        f"{IMAGING}/queries/fm_estimate_overview",
        json={
            "kwargs": {
                "channels": [c.to_dict() for c in CHANNELS],
                "parameters": ONE_BY_TWO.to_dict(),
            }
        },
    )
    assert resp.json()["result"]["unreachable"] == []


def test_an_overview_is_centred_where_asked_and_announces_its_mosaic(
    microscope, tmp_path, monkeypatch
):
    """The tab's path too: it centres on a dragged grid and shows the mosaic as it
    lands, before the run reports finished."""
    from fibsem.fm import acquisition

    centres, seen = [], []
    runner = acquisition.FMTiledAcquisitionRunner

    def recording(**kwargs):
        centres.append(kwargs["centre_position"])
        return runner(**kwargs)

    monkeypatch.setattr(acquisition, "FMTiledAcquisitionRunner", recording)
    centre = microscope.get_stage_position()

    def record(*args):
        seen.append(args)

    microscope.imaging.overview_acquired.connect(record)
    try:
        result = microscope.imaging.fm_acquire_overview(
            CHANNELS, ONE_BY_TWO, str(tmp_path), centre_position=centre
        )
    finally:
        microscope.imaging.overview_acquired.disconnect(record)
    assert centres == [centre]
    ((mosaic, path),) = seen
    assert path == result["image"]["path"]


# ── a sparse overview, planned on a beam overview ────────────────────────

SPARSE_PLAN = "/microscope/services/imaging/queries/fm_plan_sparse_overview"
WIDTH, HEIGHT, HFW = 768, 512, 600e-6


@pytest.fixture
def overview(microscope, experiment):
    """A FIB overview taken at the milling pose; the stage back at the FM after."""
    microscope.move_to_orientation("MILLING")
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ION, resolution=[WIDTH, HEIGHT], hfw=HFW)
    )
    microscope.move_to_orientation("FM")
    path = str(experiment / "overview.tif")
    image.save(path)
    return path


def _plan(client, path, regions, **extra):
    return client.post(
        SPARSE_PLAN,
        json={"kwargs": {"image_path": path, "regions": regions, **extra}},
    )


def test_the_plan_is_the_sparse_selection_s_for_the_same_ground(
    client, microscope, overview
):
    """A box of pixels plans what the selection plans for the same box in
    metres from the overview's centre."""
    box = [WIDTH / 2 + 50, HEIGHT / 2 - 100, WIDTH / 2 + 250, HEIGHT / 2 + 20]
    resp = _plan(client, overview, [box], overlap=0.1)
    assert resp.status_code == 200, resp.text
    planned = resp.json()["result"]

    image = FibsemImage.load(overview)
    pixel = image.metadata.pixel_size.x
    region = PlaneRegion.from_points(
        [
            ((box[0] - WIDTH / 2) * pixel, (box[1] - HEIGHT / 2) * pixel),
            ((box[2] - WIDTH / 2) * pixel, (box[3] - HEIGHT / 2) * pixel),
        ]
    )
    width, height = microscope.fm.camera.resolution
    px, py = microscope.fm.camera.pixel_size
    expected = plan_sparse_fm_overview(
        [region],
        image.metadata.microscope_state.stage_position,
        BeamStageProjection.from_image(image),
        FMStageProjection.from_microscope(microscope),
        microscope.get_orientation("FM"),
        fov_x=width * px,
        fov_y=height * py,
        overlap=0.1,
        fm_is_pose=microscope._fm_is_a_pose(),
    )
    assert (planned["rows"], planned["cols"]) == (expected.rows, expected.cols)
    assert planned["tile_mask"] == [list(row) for row in expected.mask]
    assert planned["tiles"] == sum(map(sum, expected.mask))
    centre = planned["centre_position"]
    assert centre["x"] == pytest.approx(expected.centre_position.x)
    assert centre["y"] == pytest.approx(expected.centre_position.y)


def test_two_apart_regions_skip_the_ground_between(client, overview):
    left = [10, HEIGHT / 2 - 20, 60, HEIGHT / 2 + 20]
    right = [WIDTH - 60, HEIGHT / 2 - 20, WIDTH - 10, HEIGHT / 2 + 20]
    planned = _plan(client, overview, [left, right]).json()["result"]
    assert planned["tiles"] < planned["rows"] * planned["cols"]


def test_a_box_that_is_not_four_numbers_is_refused(client, overview):
    resp = _plan(client, overview, [[1, 2, 3]])
    assert resp.status_code == 422
    assert "(x0, y0, x1, y1)" in resp.text

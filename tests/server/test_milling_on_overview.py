"""Items' milling patterns drawn on a grid overview (grid_draw_milling): the
outlines land where the existing stage geometry puts them, the gap between
neighbours is measured on the sample surface, and what can't be drawn is
refused. The Demo microscope's SEM overview is face on to the sample at the
SEM pose; its FIB image is at a 15 degree milling angle."""

import copy
import math
import os
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from psygnal.containers import EventedDict

import fibsem.applications.autolamella as autolamella
from fibsem.ai.agent.catalog import CATALOG, route_call
from fibsem.applications.autolamella.server import AutoLamellaAppContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.geometry.projection import image_point_to_stage, stage_to_pixel
from fibsem.geometry.saved_images import stage_projector
from fibsem.milling.patterning import footprints as fp
from fibsem.server import AuthConfig, build_server
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType, ImageSettings, MicroscopeState
from fibsem.workflows.grids import GridRecord
from fibsem.workflows.status import TaskState, TaskStatus
from tests.run_state import stub_session

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TOOL = next(spec for spec in CATALOG if spec.name == "grid_draw_milling")
MILLING_ANGLE = 15.0
# The second item's FIB image, this far along stage x from the first's.
APART = 60e-6


@pytest.fixture
def views(microscope, monkeypatch):
    """A SEM overview at the SEM pose and a FIB image at the milling angle,
    both at the same stage x, y, z, as the Demo takes them."""
    monkeypatch.setenv("FIBSEM_SIM_NO_DELAY", "1")
    microscope.move_to_orientation("SEM")
    sem = microscope.acquire_image(
        ImageSettings(resolution=(512, 512), hfw=900e-6, beam_type=BeamType.ELECTRON)
    )
    microscope.move_to_milling_angle(milling_angle=MILLING_ANGLE)
    fib = microscope.acquire_image(
        ImageSettings(resolution=(384, 256), hfw=80e-6, beam_type=BeamType.ION)
    )
    return sem, fib


def _shipped_milling(item):
    protocol = AutoLamellaTaskProtocol.load(
        os.path.join(
            os.path.dirname(autolamella.__file__), "protocol", "task-protocol.yaml"
        )
    )
    for name, config in protocol.task_config.items():
        item.task_config[name] = copy.deepcopy(config)


def _shifted(fib, dx):
    moved = copy.deepcopy(fib)
    moved.metadata.microscope_state.stage_position.x += dx
    return moved


@pytest.fixture
def host(tmp_path, microscope, views):
    """Grid aspen with its SEM overview; two items on it with the shipped
    milling tasks, each with a FIB image, APART along x; a third with no
    milling on grid birch."""
    sem, fib = views
    host = stub_session()
    experiment = Experiment(path=tmp_path / "exp", name="overview-exp")
    experiment.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True)
    aspen = experiment.add_grid(GridRecord(name="grid-aspen"))
    birch = experiment.add_grid(GridRecord(name="grid-birch"))
    folder = experiment.grid_path(aspen) / "overview_sem"
    folder.mkdir(parents=True)
    sem.save(str(folder / "overview.tif"))
    aspen.task_history.append(
        TaskState(
            name="overview_sem",
            status=TaskStatus.Completed,
            outputs={"overview_sem": ["overview_sem/overview.tif"]},
        )
    )
    for index, dx in enumerate((0.0, APART)):
        experiment.add_new_lamella(MicroscopeState(), EventedDict())
        item = experiment.positions[index]
        item.grid_id = aspen.id
        image = _shifted(fib, dx)
        item.milling_pose = copy.deepcopy(image.metadata.microscope_state)
        _shipped_milling(item)
        os.makedirs(str(item.path), exist_ok=True)
        image.save(os.path.join(str(item.path), "ref_ion_ib.tif"))
    experiment.add_new_lamella(MicroscopeState(), EventedDict())
    experiment.positions[2].grid_id = birch.id
    host.experiment = experiment
    host.microscope = microscope
    return host


@pytest.fixture
def client(microscope, host):
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig(token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def _draw(client, **arguments):
    method, path, body = route_call(TOOL, arguments)
    return client.request(method, path, json=body)


def test_a_fib_frame_point_lands_where_the_stage_geometry_puts_it(views):
    """A pattern corner 10 µm right and 5 µm up in the FIB frame: x is kept,
    y is the 15 degree view's foreshortening undone, on an overview that sees
    the sample face on. And the overview pixel resolves back (through the
    existing inverse) to the FIB pixel it came from."""
    sem, fib = views
    frame = fp.FibFrame.from_image(fib, "fib")
    assert frame is not None
    ((x, y),) = frame.to_overview(np.array([[10e-6, 5e-6]]), sem)
    size = sem.metadata.pixel_size.x
    surface = 5e-6 / math.sin(math.radians(MILLING_ANGLE))
    assert x == pytest.approx(256 + 10e-6 / size, abs=1e-6)
    assert y == pytest.approx(256 - surface / size, abs=1e-6)

    back = stage_to_pixel(fib, image_point_to_stage(sem, x, y))
    fib_size = fib.metadata.pixel_size.x
    assert back == pytest.approx((192 + 10e-6 / fib_size, 128 - 5e-6 / fib_size))


def test_the_fib_centre_lands_on_the_item_s_marker(views, tmp_path):
    """The frame's centre goes where grid_get_markers puts the pose."""
    sem, fib = views
    path = str(tmp_path / "sem.tif")
    sem.save(path)
    _, project = stage_projector(path)
    marker = project(fib.metadata.microscope_state.stage_position)
    frame = fp.FibFrame.from_image(fib, "fib")
    ((x, y),) = frame.to_overview(np.zeros((1, 2)), FibsemImage.load(path))
    assert (x, y) == pytest.approx((marker.x, marker.y))


def test_gaps_between_outlines():
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    distance, a, b = fp.gap(square, square + [3, 0])
    assert distance == pytest.approx(2) and a[0] == 1 and b[0] == 3
    assert fp.gap(square, square + [0.5, 0.5])[0] == 0
    assert fp.gap(square, square * 3 - 1)[0] == 0  # one inside the other
    line = np.array([[0.5, 4.0], [0.5, 9.0]])
    assert fp.gap(square, line)[0] == pytest.approx(3)
    corner = fp.gap(square, square + [4, 5])[0]
    assert corner == pytest.approx(5)


def test_two_items_apart_on_the_overview(client, host, views):
    """The two items' nearest cuts are measured on the sample surface: the
    same gap as their FIB-frame outlines give with the 15 degree view's
    foreshortening undone, the second APART along x."""
    sem, fib = views
    first, second = host.experiment.positions[:2]
    resp = _draw(client, item_names=[first.name, second.name])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["image"].endswith("overview_sem/overview.tif")
    assert body["path"].endswith("overview_sem/drawn/overview-milling.png")
    saved = os.path.join(str(host.experiment.path), body["path"])
    assert os.path.getsize(saved) > 0
    assert body["grid_name"] == "grid-aspen"
    assert body["task_names"] == ["Mill Fiducial", "Rough Milling", "Polishing"]
    assert body["unplaced"] == []
    assert body["caveats"]
    assert "plot" in body

    stretch = 1 / math.sin(math.radians(MILLING_ANGLE))
    outlines = [
        outline * [1, stretch]
        for name in body["task_names"]
        for milling in first.task_config[name].milling.values()
        for stage in milling.enabled_stages
        for outline in fp.stage_outlines(stage)
    ]
    expected = min(fp.gap(a, b + [APART, 0])[0] for a in outlines for b in outlines)
    (one, two) = body["items"]
    assert one["frame"].endswith("ref_ion_ib.tif")
    assert {"trench", "stress relief", "fiducial", "polishing"} <= set(one["kinds"])
    assert one["nearest"]["item_name"] == second.name
    assert one["nearest"]["distance_um"] == pytest.approx(expected * 1e6, abs=0.1)
    assert body["closest"]["items"] == [first.name, second.name]
    assert body["closest"]["distance_um"] == one["nearest"]["distance_um"]
    # Identical patterns: the same footprint, APART along x.
    assert one["size_um"] == pytest.approx(two["size_um"], abs=0.1)
    assert two["box_m"][0] - one["box_m"][0] == pytest.approx(APART, abs=1e-7)


def test_an_item_with_no_fib_image_is_placed_from_its_pose(client, host):
    first, second = host.experiment.positions[:2]
    drawn = _draw(client, grid_name="grid-aspen").json()
    os.remove(os.path.join(str(second.path), "ref_ion_ib.tif"))
    posed = _draw(client, grid_name="grid-aspen").json()
    by_image = {i["item_name"]: i for i in drawn["items"]}
    by_pose = {i["item_name"]: i for i in posed["items"]}
    assert by_pose[second.name]["frame"] == "milling pose"
    assert by_pose[second.name]["box_m"] == pytest.approx(
        by_image[second.name]["box_m"], abs=1e-9
    )
    assert any("milling pose" in c for c in posed["caveats"])


def test_refusals(client, host, tmp_path):
    first, second, other = host.experiment.positions
    elsewhere = _draw(
        client, item_names=[first.name, other.name], grid_name="grid-aspen"
    )
    assert elsewhere.status_code == 422
    assert elsewhere.json()["detail"]["item_names"] == [other.name]
    spread = _draw(client, item_names=[first.name, other.name])
    assert spread.status_code == 422
    assert spread.json()["detail"]["grid_names"] == ["grid-aspen", "grid-birch"]

    bare = os.path.join(str(host.experiment.path), "bare.tif")
    FibsemImage(np.zeros((64, 64), dtype=np.uint8), None).save(bare)
    no_stage = _draw(client, item_names=[first.name], path="bare.tif")
    assert no_stage.status_code == 422
    assert "stage geometry" in no_stage.json()["detail"]["message"]

    nothing = _draw(client, item_names=[other.name], grid_name="grid-birch")
    assert nothing.status_code == 404
    assert "SEM overview" in nothing.json()["detail"]["message"]
    mine = os.path.join(str(host.experiment.path), "mine.tif")
    shutil.copy(
        host.experiment.grid_path(host.experiment.grids[0])
        / "overview_sem"
        / "overview.tif",
        mine,
    )
    no_patterns = _draw(client, item_names=[other.name], path="mine.tif")
    assert no_patterns.status_code == 404
    assert no_patterns.json()["detail"]["unplaced"][0]["item_name"] == other.name
    no_task = _draw(client, item_names=[first.name], task_names=["Nope"])
    assert no_task.status_code == 404

    assert _draw(client).status_code == 422
    outside = _draw(client, item_names=[first.name], box=[5000, 5000, 10, 10])
    assert outside.status_code == 422

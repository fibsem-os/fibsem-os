"""A position's slices over the App Server: read as one stack in iteration
order, with the FIB image its milling kept, and its History like any item's."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem.applications.volume.manager import VolumeTaskManager  # noqa: E402
from fibsem.applications.volume.server import VolumeAppContext  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.workflows.events import EventBuffer  # noqa: E402
from fibsem.workflows.session import AppSession  # noqa: E402

from .conftest import set_up_position  # noqa: E402

TOKEN = "volume-token"
MILL = "Volume FIB Milling"
IMAGE = "Volume SEM Acquisition"


@pytest.fixture
def sliced(microscope, experiment):
    """A position cut into three slices, its milling keeping a FIB image."""
    position = set_up_position(experiment, microscope, "aspen", iterations=3)
    milling = experiment.task_protocol.task_config[MILL]
    milling.milling.acquisition.acquire_fib = True
    VolumeTaskManager(microscope=microscope, experiment=experiment).execute(
        [MILL, IMAGE]
    )
    return position, milling.slice_thickness


@pytest.fixture
def agent(microscope, experiment):
    session = AppSession()
    session.microscope = microscope
    session.experiment = experiment
    app = build_server(
        microscope,
        app_context=VolumeAppContext(session, event_buffer=EventBuffer()),
        auth=AuthConfig.generate(level="watch", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client


def _ok(answer):
    assert answer.status_code == 200, answer.text
    return answer.json()


def test_the_slices_are_one_stack_in_iteration_order(agent, sliced):
    _, thickness = sliced
    stack = _ok(
        agent.get("/app/experiment/items/aspen/stack", params={"role": "slice"})
    )

    assert stack["available"] is True
    assert stack["role"] == "slice"
    slices = stack["slices"]
    assert [s["iteration"] for s in slices] == [0, 1, 2]
    assert [s["filename"] for s in slices] == [
        "slice_0000_eb.tif",
        "slice_0001_eb.tif",
        "slice_0002_eb.tif",
    ]
    assert [s["z"] for s in slices] == pytest.approx([0, thickness, 2 * thickness])
    for entry in slices:
        assert entry["pixel_size"] > 0
        assert entry["hfw"] > 0
        assert entry["acquired_at"]


def test_the_milling_keeps_a_fib_image_per_slice(agent, sliced):
    position, _ = sliced
    stack = _ok(
        agent.get("/app/experiment/items/aspen/stack", params={"role": "slice_fib"})
    )

    assert [s["iteration"] for s in stack["slices"]] == [0, 1, 2]
    milled = [run for run in position.task_history if run.name == MILL]
    assert all(run.outputs.get("slice_fib") for run in milled)


def test_a_slice_is_served_at_the_width_asked_for(agent, sliced):
    small = agent.get(
        "/app/experiment/items/aspen/outputs/slice_0001_eb.tif", params={"width": 64}
    )
    full = agent.get("/app/experiment/items/aspen/outputs/slice_0001_eb.tif")

    assert small.status_code == full.status_code == 200
    assert small.headers["content-type"] == "image/jpeg"
    assert len(small.content) < len(full.content)


def test_an_unknown_position_or_role_reads_as_nothing_there(agent, sliced):
    missing = _ok(
        agent.get("/app/experiment/items/oak/stack", params={"role": "slice"})
    )
    assert missing["available"] is False
    assert missing["item_names"] == ["aspen"]

    other = _ok(
        agent.get("/app/experiment/items/aspen/stack", params={"role": "other"})
    )
    assert other["available"] is True and other["slices"] == []


def test_a_position_has_a_history_like_any_item(agent, sliced):
    history = _ok(agent.get("/app/experiment/items/aspen/history"))

    assert history["available"] is True
    assert history["total"] == 6
    assert history["tasks"] == [MILL, IMAGE]
    imaged = [run for run in history["runs"] if run["task_name"] == IMAGE]
    assert all(run["images"] for run in imaged)

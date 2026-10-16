"""Milling a protocol's stages through the App Server (map 8b): ``run_stage`` and
``run_task`` mill as the app does and run on as long commands, saving only
inside the open experiment; ``preview_patterns`` draws stages over a saved image
the agent can look at.
"""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import config as fcfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.runs import ended_as  # noqa: E402
from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.tasks import FibsemMillingTaskConfig  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import BeamType, ImageSettings  # noqa: E402

TOKEN = "test-token"
MILLING = "/microscope/services/milling"


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
def client(microscope, experiment, tmp_path, monkeypatch):
    # Where a task with no path saves: the app's data folder, here a temporary one.
    monkeypatch.setattr(fcfg, "DATA_CC_PATH", str(tmp_path / "data"))
    context = SimpleNamespace(experiment_path=str(experiment))
    app = build_server(
        microscope,
        app_context=context,
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


def _stage(name="Rough") -> FibsemMillingStage:
    stage = FibsemMillingStage(name=name)
    stage.alignment.enabled = False
    return stage


def _run(client, command, kwargs):
    resp = client.post(f"{MILLING}/commands/{command}", json={"kwargs": kwargs})
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    ended = client.get(f"{MILLING}/run", params={"run_id": run["run_id"], "wait": 30})
    assert ended.status_code == 200, ended.text
    return ended.json()["run"]


def test_run_stage_mills_the_stage_as_a_task_of_its_own(client, tmp_path):
    run = _run(client, "run_stage", {"stage": _stage().to_dict()})
    assert run["state"] == "done", run
    result = run["result"]
    assert result["status"] == "completed"
    assert [stage["name"] for stage in result["stages"]] == ["Rough"]
    assert result["stages"][0]["status"] == "completed"
    # The stage acquires no images, so nothing is saved, and no folder is named.
    assert result["path"] is None
    assert not os.path.exists(tmp_path / "data" / "Milling" / "Rough")


def test_run_task_saves_inside_the_experiment(client, experiment):
    config = FibsemMillingTaskConfig.from_stages(
        [_stage("Rough"), _stage("Polish")], name="Trench"
    )
    config.acquisition.imaging.path = "/somewhere/else"
    config.acquisition.acquire_fib = True  # so the task saves something
    run = _run(
        client,
        "run_task",
        {"config": config.to_dict(), "path": str(experiment / "runs")},
    )
    assert run["state"] == "done", run
    assert run["result"]["path"].startswith(str(experiment))
    assert os.listdir(run["result"]["path"])
    assert [s["status"] for s in run["result"]["stages"]] == ["completed"] * 2


def test_a_task_that_saved_nothing_still_answers_its_folder(client, experiment):
    """A task stopped before its first image answered ``path: null`` (FIB-1430):
    the task's folder is answered however it ended, empty when nothing was
    saved. Here nothing is: no alignment, no acquisition."""
    config = FibsemMillingTaskConfig.from_stages([_stage()], name="Quiet")
    config.alignment.enabled = False
    config.acquisition.acquire_sem = config.acquisition.acquire_fib = False

    run = _run(
        client,
        "run_task",
        {"config": config.to_dict(), "path": str(experiment / "runs")},
    )

    folder = run["result"]["path"]
    assert folder == str(experiment / "runs" / "Milling" / "Quiet")
    assert os.path.isdir(folder) and not os.listdir(folder)


def test_a_path_outside_the_experiment_is_refused(client, tmp_path):
    resp = client.post(
        f"{MILLING}/commands/run_stage",
        json={"kwargs": {"stage": _stage().to_dict(), "path": str(tmp_path)}},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_a_stopped_task_ends_cancelled():
    assert ended_as({"status": "cancelled", "stages": []}) == "cancelled"
    assert ended_as({"status": "completed"}) == "done"


def test_the_listing_marks_the_stage_and_task_runs_long(client):
    commands = client.get("/microscope/services").json()["milling"]["commands"]
    assert commands["run_stage"]["long"] is True
    assert commands["run_task"]["long"] is True


def _saved_ion_image(microscope, folder) -> str:
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ION, resolution=[768, 512], hfw=150e-6)
    )
    path = str(folder / "ion.tif")
    image.save(path)
    return path


def test_pattern_preview_draws_the_stages_over_a_saved_image(
    client, microscope, experiment
):
    path = _saved_ion_image(microscope, experiment)
    resp = client.post(
        f"{MILLING}/queries/preview_patterns",
        json={"kwargs": {"image_path": path, "stages": [_stage().to_dict()]}},
    )
    assert resp.status_code == 200, resp.text
    answer = resp.json()["result"]
    assert answer["summary"]["metadata"]["stages"] == ["Rough"]
    assert answer["summary"]["metadata"]["full_width"] == 768
    # The saved image's shape, so a pixel there is the same place here.
    assert answer["summary"]["shape"][:2] == [512, 768]
    assert answer["preview"]["image_b64_jpeg"]


def test_pattern_preview_reads_only_inside_the_experiment(client, microscope, tmp_path):
    path = _saved_ion_image(microscope, tmp_path)
    resp = client.post(
        f"{MILLING}/queries/preview_patterns", json={"kwargs": {"image_path": path}}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"

"""Aligning to a saved reference image through the App Server (map 8c): the
alignment service's ``align_to_reference`` runs on as a long command and saves beside the
reference; ``measure_shift`` measures between two saved images and moves
nothing. Both read only inside the open experiment.
"""

import os
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from fibsem.server import AuthConfig, build_server
from fibsem.structures.settings import BeamType, ImageSettings

TOKEN = "test-token"
ALIGNMENT = "/microscope/services/alignment"


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


def _saved(microscope, folder, name="reference") -> str:
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ION, resolution=[384, 256], hfw=150e-6)
    )
    path = str(folder / f"{name}.tif")
    image.save(path)
    return path


def test_measure_shift_between_an_image_and_itself_is_none(
    client, microscope, experiment
):
    path = _saved(microscope, experiment)
    resp = client.post(
        f"{ALIGNMENT}/queries/measure_shift",
        json={"kwargs": {"reference_path": path, "image_path": path}},
    )
    assert resp.status_code == 200, resp.text
    shift = resp.json()["result"]
    assert abs(shift["shift_px"]["x"]) < 1 and abs(shift["shift_px"]["y"]) < 1
    assert shift["method"] == "cross-correlation"


def test_align_runs_on_and_saves_beside_the_reference(client, microscope, experiment):
    path = _saved(microscope, experiment)
    resp = client.post(
        f"{ALIGNMENT}/commands/align_to_reference",
        json={"kwargs": {"reference_path": path, "steps": 1}},
    )
    assert resp.status_code == 200, resp.text
    run = resp.json()["result"]
    assert run["state"] == "running"
    ended = client.get(
        f"{ALIGNMENT}/run", params={"run_id": run["run_id"], "wait": 30}
    ).json()["run"]
    assert ended["state"] == "done", ended
    result = ended["result"]
    assert result["status"] == "completed"
    assert len(result["results"]) == 1
    assert result["path"].startswith(str(experiment / "Alignment"))
    assert os.path.isdir(result["path"])


def test_a_reference_outside_the_experiment_is_refused(client, microscope, tmp_path):
    path = _saved(microscope, tmp_path, "elsewhere")
    for route, kwargs in (
        ("commands/align_to_reference", {"reference_path": path}),
        ("queries/measure_shift", {"reference_path": path, "image_path": path}),
    ):
        resp = client.post(f"{ALIGNMENT}/{route}", json={"kwargs": kwargs})
        assert resp.status_code == 403, route
        assert resp.json()["detail"]["error_type"] == "not_allowed"


def test_the_listing_has_the_alignment_service(client):
    alignment = client.get("/microscope/services").json()["alignment"]
    assert alignment["commands"]["align_to_reference"]["long"] is True
    assert "measure_shift" in alignment["queries"]


def test_check_coincidence_at_target_answers_an_overlay_and_saves_the_pair(
    client, experiment
):
    """The experimental target-patch measure: it takes its own pair, answers
    the overlay with the measurement (dx beside dy), and the saved pair
    replays through the query to the same answer."""
    resp = client.post(
        f"{ALIGNMENT}/commands/check_coincidence",
        json={
            "kwargs": {"hfw": 150e-6, "path": str(experiment / "coincidence")},
            "answer": "preview",
        },
    )
    assert resp.status_code == 200, resp.text
    answer = resp.json()["result"]
    assert answer["preview"]
    measured = answer["summary"]["metadata"]
    assert measured["method"] == "target-patch"
    for key in ("dx", "dy", "dz", "score", "warnings", "agrees"):
        assert key in measured
    for key in ("sem_path", "fib_path", "overlay_path"):
        assert os.path.isfile(measured[key]), key

    replay = client.post(
        f"{ALIGNMENT}/queries/measure_coincidence",
        json={
            "kwargs": {
                "sem_path": measured["sem_path"],
                "fib_path": measured["fib_path"],
            }
        },
    )
    assert replay.status_code == 200, replay.text
    replayed = replay.json()["result"]["summary"]["metadata"]
    assert replayed["dy"] == pytest.approx(measured["dy"])
    assert replayed["dx"] == pytest.approx(measured["dx"])

    from_fib = client.post(
        f"{ALIGNMENT}/queries/measure_coincidence",
        json={
            "kwargs": {
                "sem_path": measured["sem_path"],
                "fib_path": measured["fib_path"],
                "beam": "ION",
            }
        },
    )
    assert from_fib.status_code == 200, from_fib.text
    assert from_fib.json()["result"]["summary"]["metadata"]["reference"] == "ION"


def test_a_coincidence_pair_outside_the_experiment_is_refused(
    client, microscope, tmp_path
):
    path = _saved(microscope, tmp_path, "elsewhere")
    resp = client.post(
        f"{ALIGNMENT}/queries/measure_coincidence",
        json={"kwargs": {"sem_path": path, "fib_path": path}},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"

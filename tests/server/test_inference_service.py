"""Running a registered model on a saved image through the App Server: the
inference service's ``segment`` reads inside the open experiment, saves its
mask and overlay beside the image, and works at Watch."""

import os
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.ai.models import ModelEntry, ModelManager, ModelRegistry  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import BeamType, ImageSettings  # noqa: E402

TOKEN = "test-token"
INFERENCE = "/microscope/services/inference"


@pytest.fixture(scope="module")
def microscope():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    # A registry of its own: the developer's models.yaml stays out of the test.
    hidden = ModelEntry("hidden", "builtin://fake/segmentation", agent=False)
    microscope.inference._manager = ModelManager(
        ModelRegistry([*ModelRegistry().entries(), hidden])
    )
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
        auth=AuthConfig.generate(level="watch", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


def _saved(microscope, folder, name="overview") -> str:
    image = microscope.acquire_image(
        ImageSettings(beam_type=BeamType.ELECTRON, resolution=[384, 256], hfw=150e-6)
    )
    path = str(folder / f"{name}.tif")
    image.save(path)
    return path


def test_list_models_names_only_the_agents(client):
    resp = client.post(f"{INFERENCE}/queries/list_models", json={})
    assert resp.status_code == 200, resp.text
    names = [model["name"] for model in resp.json()["result"]]
    assert "fake-segmentation" in names and "hidden" not in names


def test_segment_at_watch_saves_beside_the_image(client, microscope, experiment):
    path = _saved(microscope, experiment)
    resp = client.post(
        f"{INFERENCE}/queries/segment", json={"kwargs": {"image_path": path}}
    )
    assert resp.status_code == 200, resp.text
    result = resp.json()["result"]
    assert result["model"]["model"] == "fake-segmentation"
    assert result["model"]["beam"] == "ELECTRON"
    assert result["classes"]["Grid Bar"]["fraction"] > 0
    assert result["classes"]["Grid Bar"]["area_um2"] > 0
    for key in ("mask_path", "overlay_path", "result_path"):
        assert result[key].startswith(str(experiment / "inference"))
        assert os.path.isfile(result[key])


def test_segment_refuses_a_model_not_registered_or_kept_from_the_agent(
    client, microscope, experiment
):
    path = _saved(microscope, experiment)
    for model in ("no-such-model", "hidden"):
        resp = client.post(
            f"{INFERENCE}/queries/segment",
            json={"kwargs": {"image_path": path, "model": model}},
        )
        assert resp.status_code == 404, resp.text
        assert resp.json()["detail"]["error_type"] == "ModelNotFound"


def test_an_image_outside_the_experiment_is_refused(client, microscope, tmp_path):
    path = _saved(microscope, tmp_path, "elsewhere")
    resp = client.post(
        f"{INFERENCE}/queries/segment", json={"kwargs": {"image_path": path}}
    )
    assert resp.status_code == 403
    assert resp.json()["detail"]["error_type"] == "not_allowed"

"""The export routes: the Qt export dialogs' files, over HTTP."""

import io
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402
from psygnal.containers import EventedDict  # noqa: E402

from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.codec import decode  # noqa: E402
from fibsem.imaging.export import (  # noqa: E402
    default_options,
    load_export_image,
    render_export,
)
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.responses import AnimationFrames, ImageExportForm  # noqa: E402
from fibsem.structures import BeamType, ImageSettings, MicroscopeState  # noqa: E402
from fibsem.workflows.status import TaskState, TaskStatus  # noqa: E402
from tests.run_state import stub_session  # noqa: E402

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
DASHBOARD = {"Authorization": "Bearer dashboard-token"}


@pytest.fixture
def experiment(tmp_path, microscope):
    """One lamella whose two tasks each saved a final SEM and FIB image."""
    exp = Experiment(path=tmp_path / "exp", name="export-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    lamella = exp.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    for task in ("rough_milling", "polishing"):
        outputs = {}
        for beam, key in (
            (BeamType.ELECTRON, "final_sem"),
            (BeamType.ION, "final_fib"),
        ):
            image = microscope.acquire_image(
                ImageSettings(resolution=(96, 64), hfw=80e-6, beam_type=beam)
            )
            name = (
                f"ref_{task}_final_highres_{'eb' if key == 'final_sem' else 'ib'}.tif"
            )
            image.save(os.path.join(lamella.path, name))
            outputs[key] = [name]
        lamella.task_history.append(
            TaskState(name=task, status=TaskStatus.Completed, outputs=outputs)
        )
    return exp


@pytest.fixture
def client(microscope, experiment):
    host = stub_session()
    host.experiment = experiment
    host.microscope = microscope
    app = build_server(
        microscope,
        app_context=AutoLamellaAppContext(host),
        auth=AuthConfig(token=TOKEN, dashboard_token="dashboard-token"),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def _image_path(experiment) -> str:
    lamella = experiment.positions[0]
    relative = os.path.relpath(lamella.path, experiment.path)
    return f"{relative}/ref_polishing_final_highres_ib.tif".replace(os.sep, "/")


def test_the_image_form_starts_from_the_dialog_s_defaults(client, experiment):
    path = _image_path(experiment)
    form = client.get(
        "/app/experiment/export/image/options", params={"path": path}, headers=AUTH
    )
    assert form.status_code == 200, form.text
    body = form.json()
    decode(ImageExportForm, body, strict=True)
    image = load_export_image(os.path.join(experiment.path, path))
    assert body["kind"] == "FIB"
    assert body["options"]["fields"] == list(default_options(image).fields)
    assert [f["key"] for f in body["fields"]] == image.field_keys()
    assert (body["width"], body["height"]) == (96, 64)
    assert body["adjustable"] is True
    # The dashboard looks: the form is a GET.
    watched = client.get(
        "/app/experiment/export/image/options",
        params={"path": path},
        headers=DASHBOARD,
    )
    assert watched.status_code == 200


def test_an_image_exports_as_the_dialog_renders_it(client, experiment):
    path = _image_path(experiment)
    options = {"scale": 2, "provenance": ["experiment"]}
    response = client.post(
        "/app/experiment/export/image",
        json={"path": path, "options": options},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["content-disposition"] == (
        "attachment; filename*=UTF-8''ref_polishing_final_highres_ib_export.png"
    )
    image = load_export_image(os.path.join(experiment.path, path))
    chosen = default_options(image)
    chosen.scale, chosen.provenance = 2, ["experiment"]
    expected = render_export(image, chosen)
    with Image.open(io.BytesIO(response.content)) as png:
        assert png.format == "PNG"
        assert (png.height, png.width) == expected.shape[:2]
    assert expected.shape[1] == 192


def test_an_export_refuses_as_the_files_routes_do(client, experiment):
    missing = client.post(
        "/app/experiment/export/image", json={"path": "nope.tif"}, headers=AUTH
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["error_type"] == "not_found"
    outside = client.post(
        "/app/experiment/export/image", json={"path": "../x.tif"}, headers=AUTH
    )
    assert outside.status_code == 403
    path = _image_path(experiment)
    for options in ({"scale": 3}, {"bogus": True}, {"scalebar_location": "top"}):
        bad = client.post(
            "/app/experiment/export/image",
            json={"path": path, "options": options},
            headers=AUTH,
        )
        assert bad.status_code == 422, options
    # The dashboard only looks: an export is a POST.
    watched = client.post(
        "/app/experiment/export/image", json={"path": path}, headers=DASHBOARD
    )
    assert watched.status_code == 403


def test_a_lamella_s_two_tasks_animate_as_two_frames(client, experiment):
    name = experiment.positions[0].name
    frames = client.get(
        "/app/experiment/export/animation/frames",
        params={"item_name": name},
        headers=AUTH,
    )
    assert frames.status_code == 200, frames.text
    body = frames.json()
    decode(AnimationFrames, body, strict=True)
    assert [f["title"] for f in body["frames"]] == ["rough_milling", "polishing"]
    assert body["frames"][0]["beams"] == ["SEM", "FIB"]
    assert "gif" in body["formats"]

    response = client.post(
        "/app/experiment/export/animation",
        json={"item_name": name, "options": {"beam": "SEM", "width": 128}},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/gif"
    assert f"{name}_workflow.gif" in response.headers["content-disposition"]
    with Image.open(io.BytesIO(response.content)) as gif:
        assert gif.n_frames == 2
        assert gif.width == 128

    skipped = client.post(
        "/app/experiment/export/animation",
        json={"item_name": name, "options": {"skipped": [0, 1]}},
        headers=AUTH,
    )
    assert skipped.status_code == 404


def test_an_unknown_lamella_has_no_animation(client):
    response = client.get(
        "/app/experiment/export/animation/frames",
        params={"item_name": "nobody"},
        headers=AUTH,
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error_type"] == "not_found"

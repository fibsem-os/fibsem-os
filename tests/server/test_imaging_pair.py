"""beam_acquire_pair_preview through the App Server on the Demo microscope: the imaging
service's ``acquire_pair_preview`` saves an SEM and a FIB image inside the open
experiment, answers compact facts and one side-by-side picture whose stated
origin and scale take a point read off it back to each image's pixels, and
refuses while someone else is in control or a beam is off."""

import base64
import io
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fibsem.acting import WORKFLOW
from fibsem.ai.agent.catalog import CATALOG, route_call
from fibsem.server import AuthConfig, build_server
from fibsem.services.imaging import pair_picture
from fibsem.structures.image import FibsemImage
from fibsem.structures.settings import BeamType
from fibsem.util import wording
from tests.fixtures.app_contexts import ExperimentContext

TOKEN = "test-token"
SPEC = next(t for t in CATALOG if t.name == "beam_acquire_pair_preview")


@pytest.fixture
def experiment(tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    return folder


@pytest.fixture
def client(microscope, experiment):
    app = build_server(
        microscope,
        app_context=ExperimentContext(str(experiment)),
        auth=AuthConfig.generate(level="drive", token=TOKEN),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update({"Authorization": f"Bearer {TOKEN}"})
        yield client


def _call(client, **arguments):
    method, path, body = route_call(SPEC, arguments)
    return client.request(method, path, json=body)


def _fast(microscope):
    """Small frames, so the Demo's scan sleeps stay short."""
    for beam_type in (BeamType.ELECTRON, BeamType.ION):
        beam = microscope.imaging.beam(beam_type)
        beam.resolution.set_value((768, 512))
        beam.dwell_time.set_value(1e-9)


def test_a_site_check_saves_both_images_and_answers_compactly(
    client, microscope, experiment
):
    _fast(microscope)
    resp = _call(client, hfw=80e-6, path="02-lamella/site-check", filename="s1")
    assert resp.status_code == 200, resp.text
    answer = resp.json()["result"]

    folder = experiment / "02-lamella" / "site-check"
    assert answer["sem"]["path"] == "02-lamella/site-check/s1_eb.tif"
    assert answer["fib"]["path"] == "02-lamella/site-check/s1_ib.tif"
    assert (folder / "s1_eb.tif").is_file() and (folder / "s1_ib.tif").is_file()
    assert answer["picture"]["path"] == "02-lamella/site-check/drawn/s1_eb-s1_ib.png"
    assert (folder / "drawn" / "s1_eb-s1_ib.png").is_file()

    # Each saved image is the one described, at the field of view asked.
    for key, beam_type in (("sem", BeamType.ELECTRON), ("fib", BeamType.ION)):
        image = FibsemImage.load(str(experiment / answer[key]["path"]))
        assert image.metadata.image_settings.beam_type is beam_type
        assert image.metadata.image_settings.hfw == pytest.approx(80e-6)
        assert answer[key]["resolution"] == [768, 512]
        assert answer[key]["pixel_size"] == pytest.approx(80e-6 / 768, rel=1e-3)
        assert answer[key]["autocontrast"] is False
        quality = answer[key]["quality"]
        assert set(quality) == {
            "clipped_black",
            "clipped_white",
            "centre",
            "spread",
            "focus",
        }
        # The field of view stays on the beam.
        assert microscope.imaging.beam(beam_type).hfw.get_value() == pytest.approx(
            80e-6
        )

    assert answer["hfw"] == pytest.approx(80e-6)
    assert set(answer["stage_position"]) == {"x", "y", "z", "r_deg", "t_deg"}
    picture = answer["picture"]
    assert set(picture) == {"path", "size", "sem", "fib", "image_b64_jpeg"}
    # Without the preview, the facts are a short text.
    facts = dict(
        answer, picture={k: v for k, v in picture.items() if k != "image_b64_jpeg"}
    )
    assert len(json.dumps(facts)) < 1200


def test_autocontrast_runs_only_on_the_beams_named(client, microscope):
    _fast(microscope)
    resp = _call(client, hfw=50e-6, path="checks", autocontrast=["ion"])
    assert resp.status_code == 200, resp.text
    answer = resp.json()["result"]
    assert answer["fib"]["autocontrast"] is True
    assert answer["sem"]["autocontrast"] is False
    assert os.path.basename(answer["sem"]["path"]).startswith("pair-")
    with pytest.raises(ValueError):
        route_call(SPEC, {"hfw": 50e-6, "path": "checks", "autocontrast": ["FM"]})


def _marked(microscope, folder, name, beam_type, box):
    """A saved image of the Demo's, dark but for a bright square in *box*
    (x, y, width, height in its pixels)."""
    image = microscope.imaging.acquire(beam_type, {"hfw": 100e-6})
    data = np.full_like(image.data, 40)
    x, y, w, h = box
    data[y : y + h, x : x + w] = 250
    marked = FibsemImage(data, image.metadata)
    return marked.save(os.path.join(folder, name)), data.shape


def _centroid(gray, origin, size):
    """The bright square's centre in one half, in the picture's pixels: its
    top 80% only, clear of the scale bar drawn at the bottom right."""
    x0, y0 = int(round(origin[0])), int(round(origin[1]))
    region = gray[y0 : y0 + int(size[1] * 0.8), x0 : x0 + int(size[0])]
    ys, xs = np.nonzero(region > 180)
    return x0 + xs.mean() + 0.5, y0 + ys.mean() + 0.5


@pytest.mark.parametrize(
    "sem_resolution, fib_resolution, sem_box",
    [
        ((768, 512), (768, 512), (100, 80, 40, 40)),
        ((1536, 1024), (768, 512), (900, 150, 80, 60)),
        ((1536, 1024), (1536, 1024), (900, 150, 80, 60)),
    ],
    ids=["small", "sem-twice-the-pixels", "both-full-size"],
)
def test_a_point_on_the_picture_maps_back_to_each_image(
    microscope, tmp_path, sem_resolution, fib_resolution, sem_box
):
    """Find each half's bright square in the preview the agent sees and take
    its centre back through the stated origin and scale: it lands on the
    square's centre in that image, within a couple of full-size pixels,
    whatever each image's size and however much the preview is shrunk."""
    from PIL import Image

    _fast(microscope)
    microscope.imaging.beam(BeamType.ELECTRON).resolution.set_value(sem_resolution)
    microscope.imaging.beam(BeamType.ION).resolution.set_value(fib_resolution)
    fib_box = (500, 300, 60, 30)
    sem, sem_shape = _marked(microscope, tmp_path, "m_eb", BeamType.ELECTRON, sem_box)
    fib, fib_shape = _marked(microscope, tmp_path, "m_ib", BeamType.ION, fib_box)

    picture = pair_picture(sem, fib, (sem_shape, fib_shape))
    jpeg = base64.b64decode(picture["image_b64_jpeg"])
    gray = np.asarray(Image.open(io.BytesIO(jpeg)).convert("L"))
    assert [gray.shape[1], gray.shape[0]] == picture["size"]

    for half, box in (("sem", sem_box), ("fib", fib_box)):
        mapping = picture[half]
        px, py = _centroid(gray, mapping["origin"], mapping["size"])
        sx, sy = mapping["scale"]
        x = (px - mapping["origin"][0]) * sx
        y = (py - mapping["origin"][1]) * sy
        assert x == pytest.approx(box[0] + box[2] / 2, abs=2.5), half
        assert y == pytest.approx(box[1] + box[3] / 2, abs=2.5), half
    # The FIB half starts right of the SEM half, past the gap.
    assert picture["fib"]["origin"][0] > picture["sem"]["size"][0]


def test_refused_while_someone_else_is_in_control(client, microscope, experiment):
    with microscope.control.take(WORKFLOW, "a run"):
        resp = _call(client, hfw=50e-6, path="busy")
    assert resp.status_code == 409, resp.text
    assert not (experiment / "busy").exists()


def test_refused_when_a_beam_is_off(client, microscope, experiment):
    ion = microscope.imaging.beam(BeamType.ION)
    ion.on.set_value(False)
    try:
        resp = _call(client, hfw=50e-6, path="off")
    finally:
        ion.on.set_value(True)
    assert resp.status_code == 422, resp.text
    assert "ion beam is off" in resp.json()["detail"]["message"]
    # Refused before anything was imaged: not even the SEM image is saved.
    assert not (experiment / "off").exists()


def test_refused_outside_the_experiment(client, experiment):
    resp = _call(client, hfw=50e-6, path=str(experiment.parent))
    assert resp.status_code == 403, resp.text


def test_the_operator_reads_what_it_will_do():
    line = SPEC.line(
        {"hfw": 50e-6, "path": "02-lamella/site-check", "autocontrast": ["ION"]},
        wording.ActionContext(),
    )
    assert line["what"] == "SEM and FIB"
    assert "50 µm wide" in line["change"]
    assert "autocontrast first on FIB" in line["change"]
    assert "02-lamella/site-check" in line["change"]


def test_the_agent_gets_the_facts_and_one_picture(client, microscope):
    """Through the MCP sidecar: one image content, the side-by-side picture,
    and the facts as text with the picture's place among the images."""
    import asyncio

    pytest.importorskip("mcp")
    from fibsem.ai.agent.mcp.sidecar import build_sidecar

    _fast(microscope)
    sidecar = build_sidecar(client, client.get("/capabilities").json())
    result = asyncio.run(
        sidecar.call_tool("beam_acquire_pair_preview", {"hfw": 50e-6, "path": "site"})
    )
    if isinstance(result, tuple):
        result = result[0]
    contents = list(getattr(result, "content", result))
    images = [c for c in contents if getattr(c, "type", None) == "image"]
    assert len(images) == 1
    text = "".join(getattr(c, "text", "") for c in contents)
    assert "image_b64_jpeg" not in text
    assert json.loads(text)["result"]["picture"]["image_index"] == 0

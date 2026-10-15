"""Tests for the preview-rendition endpoints and the downscale helper."""

import base64
import io
import json
import os

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
PILImage = pytest.importorskip("PIL.Image")

from fastapi.testclient import TestClient  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.server.images import preview_jpeg_bytes  # noqa: E402

PREVIEW = "/microscope/devices/ion/commands/acquire"
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def armed_client():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(microscope, auth=AuthConfig.generate(level="drive", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    if previous is None:
        os.environ.pop("FIBSEM_SIM_NO_DELAY", None)
    else:
        os.environ["FIBSEM_SIM_NO_DELAY"] = previous


def test_preview_downscales_and_normalizes():
    data = (np.random.rand(1024, 1536) * 65535).astype(np.uint16)
    jpeg, width, height = preview_jpeg_bytes(data, max_width=768)
    assert width == 768
    assert height == 512
    decoded = PILImage.open(io.BytesIO(jpeg))
    assert decoded.format == "JPEG"
    assert decoded.size == (768, 512)


def test_preview_flat_image_does_not_divide_by_zero():
    data = np.full((64, 64), 1234, dtype=np.uint16)
    jpeg, _, _ = preview_jpeg_bytes(data)
    assert len(jpeg) > 0


def test_small_image_is_not_upscaled():
    data = np.zeros((100, 200), dtype=np.uint8)
    _, width, height = preview_jpeg_bytes(data, max_width=768)
    assert (width, height) == (200, 100)


def test_acquire_image_preview_endpoint(armed_client):
    resp = armed_client.post(PREVIEW, headers=AUTH, json={"answer": "preview"})
    assert resp.status_code == 200
    answer = resp.json()["result"]
    assert answer["path"] is None  # an acquire saves nothing
    assert answer["summary"]["kind"] == "image"
    assert answer["summary"]["beam_type"] == "ION"
    body = answer["preview"]
    jpeg = base64.b64decode(body["image_b64_jpeg"])
    decoded = PILImage.open(io.BytesIO(jpeg))
    assert decoded.format == "JPEG"
    assert body["width"] <= 768
    assert decoded.size == (body["width"], body["height"])
    assert body["full_width"] >= body["width"]
    assert body["beam_type"] == "ION"
    assert body["hfw"] is not None
    # metres per SOURCE pixel: the scale an agent needs to map coordinates
    # onto the preview without assuming a field of view.
    assert body["pixelsize"]["x"] == pytest.approx(body["hfw"] / body["full_width"])
    # metres per PREVIEW pixel, for measuring on the preview itself.
    assert body["preview_pixelsize"] == pytest.approx(
        body["pixelsize"]["x"] * body["full_width"] / body["width"]
    )
    # The summary is the image's metadata block: where and how it was taken.
    summary = answer["summary"]
    assert summary["pixel_size"] == pytest.approx(body["pixelsize"]["x"])
    assert set(summary["stage_position"]) == {
        "x",
        "y",
        "z",
        "r_deg",
        "t_deg",
        "coordinate_system",
    }
    assert summary["scan_rotation_deg"] is not None
    assert set(summary["beam"]) == {
        "voltage",
        "current",
        "working_distance",
        "dwell_time",
    }
    assert set(summary["detector"]) == {"type", "mode"}
    assert len(json.dumps(summary)) < 1000


def test_preview_needs_the_token(armed_client):
    # The level story is covered in test_server_factory; here just assert the
    # endpoint rejects a missing token so it is provably gated.
    resp = armed_client.post(PREVIEW, json={"answer": "preview"})
    assert resp.status_code == 401


def test_the_preview_tool_images_through_the_imaging_service(armed_client):
    """FIB-1261: acquire_image_preview calls imaging.acquire, with the agent's
    image settings when it gives them."""
    from fibsem.server.catalog import CATALOG, route_call

    spec = next(t for t in CATALOG if t.name == "beam_acquire_image")
    method, path, body = route_call(
        spec,
        {
            "beam_type": "ION",
            "image_settings": {"hfw": 80e-6, "resolution": [384, 256]},
        },
    )
    assert path == "/microscope/services/imaging/commands/acquire"
    resp = armed_client.request(method, path, headers=AUTH, json=body)
    assert resp.status_code == 200, resp.text
    preview = resp.json()["result"]["preview"]
    assert preview["beam_type"] == "ION"
    assert preview["full_width"] == 384
    assert preview["hfw"] == pytest.approx(80e-6)


def test_a_camera_frame_answers_the_same_shape():
    """A chamber camera or FM frame (an array, a Frame, a z-stack) answers an
    agent as an image does: summary, path, preview."""
    from fibsem.devices.wire import Frame
    from fibsem.server.images import image_answer, is_image_result

    frame = np.random.randint(0, 4096, (200, 300), dtype=np.uint16)
    answer = image_answer(frame)
    assert answer["summary"] == {
        "kind": "array",
        "shape": [200, 300],
        "dtype": "uint16",
    }
    assert answer["path"] is None
    assert (answer["preview"]["width"], answer["preview"]["height"]) == (300, 200)

    answer = image_answer(Frame(frame, {"exposure_time": 0.1}))
    assert answer["summary"]["kind"] == "frame"
    assert answer["summary"]["metadata"] == {"exposure_time": 0.1}

    stack = [Frame(frame, {"z": z}) for z in (0.0, 1e-6, 2e-6)]
    answer = image_answer(stack)
    assert answer["summary"]["count"] == 3
    assert answer["summary"]["preview_of"] == "maximum projection"
    assert [m["z"] for m in answer["summary"]["metadata"]] == [0.0, 1e-6, 2e-6]

    assert image_answer(np.stack([frame, frame]))["summary"]["preview_of"]
    assert not is_image_result(np.zeros(3))  # a plain vector is a value
    assert not is_image_result([1, 2])


def test_a_saved_image_answers_its_path(tmp_path):
    from fibsem.server.images import image_answer
    from fibsem.structures import FibsemImage

    image = FibsemImage(data=np.zeros((64, 96), dtype=np.uint8), metadata=None)
    written = image.save(path=str(tmp_path / "frame.tif"))
    answer = image_answer(image)
    assert answer["path"] == written
    assert answer["summary"]["shape"] == [64, 96]


def test_reference_images_answer_one_image_answer_each(armed_client):
    """A command that returns several images (the reference pair) answers each in
    its place, None for a beam not asked for; no program format carries a list."""
    resp = armed_client.post(
        "/microscope/services/imaging/commands/reference_images",
        headers=AUTH,
        json={
            "kwargs": {
                "image_settings": {"resolution": [96, 64], "save": False},
                "electron": False,
            }
        },
    )
    assert resp.status_code == 200, resp.text
    electron, ion = resp.json()["result"]
    assert electron is None
    assert ion["summary"]["beam_type"] == "ION"
    assert ion["preview"]["image_b64_jpeg"]

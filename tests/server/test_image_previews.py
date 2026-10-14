"""Tests for the preview-rendition endpoints and the downscale helper."""

import base64
import io
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
    body = resp.json()
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


def test_preview_needs_the_token(armed_client):
    # The level story is covered in test_server_factory; here just assert the
    # endpoint rejects a missing token so it is provably gated.
    resp = armed_client.post(PREVIEW, json={"answer": "preview"})
    assert resp.status_code == 401


def test_the_preview_tool_images_through_the_imaging_service(armed_client):
    """FIB-1261: acquire_image_preview calls imaging.acquire, with the agent's
    image settings when it gives them."""
    from fibsem.server.catalog import CATALOG, route_call

    spec = next(t for t in CATALOG if t.name == "acquire_image_preview")
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
    preview = resp.json()
    assert preview["beam_type"] == "ION"
    assert preview["full_width"] == 384
    assert preview["hfw"] == pytest.approx(80e-6)

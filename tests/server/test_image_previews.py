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

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture(scope="module")
def armed_client():
    previous = os.environ.get("FIBSEM_SIM_NO_DELAY")
    os.environ["FIBSEM_SIM_NO_DELAY"] = "1"
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=True, token=TOKEN)
    )
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
    resp = armed_client.post(
        "/acquire_image_preview", headers=AUTH, json={"beam_type": "ION"}
    )
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


def test_last_image_preview_endpoint(armed_client):
    armed_client.post(
        "/acquire_image_preview", headers=AUTH, json={"beam_type": "ELECTRON"}
    )
    resp = armed_client.post(
        "/last_image_preview", headers=AUTH, json={"beam_type": "ELECTRON"}
    )
    assert resp.status_code == 200
    assert resp.json()["image_b64_jpeg"]


def test_preview_requires_hardware_scope(armed_client):
    # Same app, wrong scope story is covered in test_server_factory; here just
    # assert the endpoint rejects a missing token so it is provably gated.
    resp = armed_client.post("/acquire_image_preview", json={"beam_type": "ION"})
    assert resp.status_code == 401

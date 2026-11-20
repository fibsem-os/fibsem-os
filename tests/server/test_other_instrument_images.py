"""An instrument that isn't a FIB-SEM brings its own image class: any object with
its pixels as ``data`` and an ``agent_summary()`` answers an agent, and shows on
the display, as a beam image does (`fibsem.server.images.own_summary`)."""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fibsem.devices import Device
from fibsem.devices.core import command
from fibsem.instrument import DeviceInstrument
from fibsem.server import AuthConfig, build_server
from fibsem.server.image_payloads import image_info, image_payload
from fibsem.server.images import image_answer, is_image_result, own_summary

TOKEN = "t"


@dataclass
class OtherImage:
    """Another instrument's image: its own file and facts, no fibsem metadata."""

    data: np.ndarray
    filepath: Optional[str] = "/data/image_0001.mrc"
    metadata: Dict[str, Any] = field(default_factory=lambda: {"anything": "its own"})

    def agent_summary(self) -> Dict[str, Any]:
        return {
            "pixel_size_m": 4e-10,
            "acquired_at": "2026-11-20T10:00:00",
            "dose": "2 e/px/s",
        }


def _image() -> OtherImage:
    return OtherImage(np.random.randint(0, 4096, (300, 400), dtype=np.uint16))


def test_it_answers_an_agent_as_a_beam_image_does():
    image = _image()
    assert is_image_result(image)
    answer = image_answer(image)
    assert answer["summary"]["kind"] == "image"
    assert answer["summary"]["dose"] == "2 e/px/s"
    assert answer["path"] == "/data/image_0001.mrc"
    preview = answer["preview"]
    assert base64.b64decode(preview["image_b64_jpeg"])[:2] == b"\xff\xd8"
    assert (preview["full_width"], preview["full_height"]) == (400, 300)
    assert preview["pixelsize"] == {"x": 4e-10, "y": 4e-10}


def test_it_shows_on_the_display_with_its_scale_and_time():
    payload = image_payload(_image())
    assert payload is not None and payload["pixelsize"]["x"] == 4e-10
    assert payload["acquired_at"] == "2026-11-20T10:00:00"
    # Its facts are its summary, not a beam image's fields.
    assert image_info(_image()) is None


def test_an_object_that_says_nothing_of_itself_is_not_an_image():
    class NotAnImage:
        data = np.zeros((10, 10))

    assert own_summary(NotAnImage()) is None
    assert not is_image_result(NotAnImage())


class Camera(Device):
    @command
    def acquire(self) -> OtherImage:
        """One image."""
        return _image()

    @command
    def acquire_with_facts(self) -> Dict[str, Any]:
        """One image, under ``image``, with facts beside it."""
        return {"image": _image(), "exposures": 1}


@pytest.fixture
def client():
    camera = Camera("camera")
    instrument = DeviceInstrument({"camera": camera})
    app = build_server(
        auth=AuthConfig.generate(level="drive", token=TOKEN), instruments=[instrument]
    )
    c = TestClient(app)
    c.headers.update({"Authorization": f"Bearer {TOKEN}"})
    return c


@pytest.mark.parametrize("answer", [None, "preview"])
def test_a_command_answers_it_as_an_image_with_or_without_preview(client, answer):
    body = {"kwargs": {}} if answer is None else {"kwargs": {}, "answer": answer}
    out = client.post("/microscope/devices/camera/commands/acquire", json=body)
    assert out.status_code == 200, out.text
    result = out.json()["result"]
    assert result["summary"]["kind"] == "image"
    assert result["preview"]["image_b64_jpeg"]


def test_an_image_under_image_in_a_dict_answers_as_one(client):
    out = client.post(
        "/microscope/devices/camera/commands/acquire_with_facts",
        json={"kwargs": {}, "answer": "preview"},
    )
    assert out.status_code == 200, out.text
    result = out.json()["result"]
    assert result["exposures"] == 1
    assert result["image"]["summary"]["dose"] == "2 e/px/s"

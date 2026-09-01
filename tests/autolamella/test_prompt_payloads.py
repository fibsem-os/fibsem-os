"""Prompt payloads carry the scale needed to use their images.

The mirrored-overlay incident: an agent mapped coordinates onto a preview
using the live view's field of view, which differed from the acquisition's.
Every image payload now carries its own scale (source dimensions, hfw,
pixelsize) and every coordinate-bearing payload states its convention —
these tests pin that contract with real domain objects."""

import json

import numpy as np
import pytest

pytest.importorskip("PIL")

from fibsem.applications.autolamella.server.prompts import serialize_request
from fibsem.applications.autolamella.workflows.interaction import (
    EditAlignmentArea,
    PickPOI,
)
from fibsem.structures import FibsemImage, FibsemRectangle, Point


def _image(resolution=(1536, 1024), hfw=100e-6) -> FibsemImage:
    return FibsemImage.generate_blank_image(resolution=resolution, hfw=hfw, random=True)


def test_pick_poi_payload_carries_scale_and_convention():
    image = _image()
    payload = serialize_request(PickPOI(image=image, initial=Point(1e-6, -2e-6)))

    entry = payload["image"]
    assert entry["full_width"] == 1536
    assert entry["full_height"] == 1024
    assert entry["hfw"] == pytest.approx(100e-6)
    # metres per SOURCE pixel — the preview is downscaled, the pixelsize is not.
    assert entry["pixelsize"]["x"] == pytest.approx(100e-6 / 1536)
    assert entry["width"] <= entry["full_width"]
    assert "+y UP" in payload["coordinates"]
    json.dumps(payload)  # wire-safe


def test_edit_alignment_area_payload_declares_the_frame_slot():
    # The frame is display state the request doesn't carry: the serializer
    # declares the slot, AgentContext.pending_prompt fills it from the FIB
    # display cache (pinned over HTTP in test_agent_prompt_surface.py).
    area = FibsemRectangle(left=0.4, top=0.4, width=0.2, height=0.2)
    payload = serialize_request(EditAlignmentArea(initial=area))

    assert payload["initial"]["width"] == pytest.approx(0.2)
    assert payload["image"] is None
    assert "fractions of the frame" in payload["coordinates"]
    json.dumps(payload)


def test_confirm_detection_bare_array_reports_source_dimensions():
    # DetectedFeatures.image is a bare ndarray — no metadata to draw scale
    # from, but the source dimensions still anchor the feature px coordinates.
    from fibsem.detection.detection import DetectedFeatures, LamellaCentre

    detection = DetectedFeatures(
        features=[LamellaCentre(px=Point(10, 20))],
        image=np.random.randint(0, 255, size=(1024, 1536), dtype=np.uint8),
        mask=None,
        rgb=None,
        pixelsize=6.5e-8,
    )
    from fibsem.applications.autolamella.workflows.interaction import (
        ConfirmDetection,
    )

    payload = serialize_request(ConfirmDetection(detection=detection))
    assert payload["pixelsize"] == pytest.approx(6.5e-8)
    assert payload["image"]["full_width"] == 1536
    assert payload["image"]["full_height"] == 1024
    assert (
        "pixelsize" not in payload["image"] or payload["image"].get("pixelsize") is None
    )
    assert "source-image pixels" in payload["coordinates"]
    json.dumps(payload)

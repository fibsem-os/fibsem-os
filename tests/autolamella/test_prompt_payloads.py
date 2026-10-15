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
    RunMillingTask,
    RunSpotBurn,
)
from fibsem.imaging.spot import SpotBurnSettings
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
    # declares the slot, AutoLamellaAppContext.pending_prompt fills it from the FIB
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


def test_run_milling_payload_tells_the_first_ask_from_a_re_ask():
    """At the first ask no goes on without milling; after a pass no accepts
    the mill. The payload says which, so an agent need not infer it from
    history (FIB-1329)."""
    from fibsem.milling import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    config = FibsemMillingTaskConfig(name="Rough Mill", stages=[FibsemMillingStage()])
    first = serialize_request(RunMillingTask(config=config))
    again = serialize_request(RunMillingTask(config=config, passes=1))

    assert first["passes"] == 0
    assert "without milling" in first["message"]
    assert again["passes"] == 1
    assert "accepts what was milled" in again["message"]
    json.dumps(first)
    json.dumps(again)


def test_spot_burn_payload_carries_points_and_settings():
    settings = SpotBurnSettings(
        coordinates=[Point(0.25, 0.5)], milling_current=60e-12, exposure_time=5.0
    )
    payload = serialize_request(RunSpotBurn(settings=settings, message="Burn"))

    assert payload["points"] == [{"x": 0.25, "y": 0.5}]
    assert payload["milling_current"] == pytest.approx(60e-12)
    assert payload["exposure_time"] == 5.0
    assert payload["positive"] == "Run Spot Burn"
    assert "fractions of the frame" in payload["coordinates"]
    json.dumps(payload)


def _refuse(message):
    return {"invalid_value": message}


def _detection():
    from fibsem.detection.detection import DetectedFeatures, LamellaCentre

    return DetectedFeatures(
        features=[LamellaCentre()],
        image=np.zeros((8, 10), dtype=np.uint8),
        mask=np.zeros((8, 10), dtype=np.uint8),
        rgb=np.zeros((8, 10, 3), dtype=np.uint8),
        pixelsize=1e-9,
    )


def test_a_detection_value_moves_named_features_inside_the_image():
    from fibsem.applications.autolamella.server.context import _parse_detection_value
    from fibsem.applications.autolamella.workflows.interaction import (
        ConfirmDetection,
    )

    request = ConfirmDetection(detection=_detection())
    name = request.detection.features[0].name

    moved, refusal = _parse_detection_value(
        request, {"features": [{"name": name, "px": {"x": 9, "y": 7}}]}, _refuse
    )
    assert refusal is None and moved == {name: Point(9.0, 7.0)}

    for bad in (
        {"features": []},
        {"features": [{"name": "Nope", "px": {"x": 1, "y": 1}}]},
        {"features": [{"name": name, "px": {"x": 10, "y": 1}}]},  # x == width
        {"features": [{"name": name, "px": {"x": 1}}]},
        {"features": [{"name": name, "px": {"x": 1, "y": 1}}] * 2},
    ):
        parsed, refusal = _parse_detection_value(request, bad, _refuse)
        assert parsed is None and "invalid_value" in refusal, bad


def test_a_spot_burn_value_is_points_inside_the_frame():
    from fibsem.applications.autolamella.server.context import _parse_spot_burn_value

    points, refusal = _parse_spot_burn_value(
        {"points": [{"x": 0, "y": 1}, {"x": 0.5, "y": 0.25}]}, _refuse
    )
    assert refusal is None and points == [Point(0.0, 1.0), Point(0.5, 0.25)]

    for bad in (
        {},
        {"points": [{"x": 1.2, "y": 0.5}]},
        {"points": [{"x": "a", "y": 0.5}]},
        {"points": [{"x": float("nan"), "y": 0.5}]},
    ):
        parsed, refusal = _parse_spot_burn_value(bad, _refuse)
        assert parsed is None and "invalid_value" in refusal, bad

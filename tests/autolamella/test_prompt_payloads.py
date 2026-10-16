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


def _milling_config():
    from fibsem.milling import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    return FibsemMillingTaskConfig(
        name="Rough Mill",
        stages=[FibsemMillingStage(name="Rough 1"), FibsemMillingStage(name="Rough 2")],
    )


def test_run_milling_payload_tells_the_first_ask_from_a_re_ask():
    """Before a pass the actions are run/edit/skip; after one, run mills
    again and accept goes on. The payload says which, so an agent need not
    infer it from history (FIB-1329)."""
    config = _milling_config()
    first = serialize_request(RunMillingTask(config=config))
    again = serialize_request(RunMillingTask(config=config, passes=1))

    assert first["passes"] == 0 and first["actions"] == ["run", "edit", "skip"]
    assert (first["positive"], first["negative"]) == ("Run Milling", "Skip")
    assert again["passes"] == 1 and again["actions"] == ["run", "edit", "accept"]
    assert (again["positive"], again["negative"]) == ("Mill Again", "Accept")
    assert first["value_required"] is True and "action" in first["value"]
    json.dumps(first)
    json.dumps(again)


def test_run_milling_payload_explains_every_field():
    """The config and what each of its options does, with the ones nothing
    reads marked (Patrick 2026-10-16)."""
    payload = serialize_request(RunMillingTask(config=_milling_config()))

    assert payload["config"]["stages"][0]["name"] == "Rough 1"
    fields = payload["fields"]
    assert fields["task"]["alignment.interval"]["not_used"]
    assert fields["stage"]["alignment"]["not_used"]
    assert "milling_current" in json.dumps(fields["stage"])


def test_a_milling_answer_is_an_action_with_its_edits():
    from fibsem.applications.autolamella.workflows.interaction import (
        InvalidAnswerValue,
        apply_milling_edit,
    )

    request = RunMillingTask(config=_milling_config())
    edit = request.parse_value(
        {"action": "edit", "patch": {"stages.Rough 2.pattern.depth": 2e-6}}
    )
    assert edit.action == "edit"
    edited = apply_milling_edit(request.config, edit)
    assert edited.stages[1].pattern.depth == pytest.approx(2e-6)
    assert request.config.stages[1].pattern.depth != pytest.approx(2e-6)

    for bad in (
        {},
        {"action": "mill"},
        {"action": "accept"},  # only after a pass
        {"action": "run", "patch": {"stages.Nope.pattern.depth": 1e-6}},
        {"action": "run", "patch": {"stages.0.alignment.enabled": True}},
        {"action": "run", "patch": {"alignment.interval": 5}},
        {"action": "run", "extra": 1},
    ):
        with pytest.raises(InvalidAnswerValue):
            request.parse_value(bad)


def test_a_whole_config_reorders_and_removes_stages():
    request = RunMillingTask(config=_milling_config())
    document = request.config.to_dict()
    document["stages"] = [document["stages"][1]]
    edit = request.parse_value({"action": "run", "config": document})

    from fibsem.applications.autolamella.workflows.interaction import (
        apply_milling_edit,
    )

    edited = apply_milling_edit(request.config, edit)
    assert [stage.name for stage in edited.stages] == ["Rough 2"]


def test_spot_burn_payload_carries_points_and_settings():
    settings = SpotBurnSettings(
        coordinates=[Point(0.25, 0.5)], milling_current=60e-12, exposure_time=5.0
    )
    payload = serialize_request(RunSpotBurn(settings=settings, message="Burn"))

    assert payload["points"] == [{"x": 0.25, "y": 0.5}]
    assert payload["milling_current"] == pytest.approx(60e-12)
    assert payload["exposure_time"] == 5.0
    assert payload["positive"] == "Run Spot Burn"
    assert payload["negative"] == "Skip"
    assert payload["actions"] == ["run", "edit", "skip"]
    assert payload["passes"] == 0
    assert payload["exposure_limits"] == [0.1, 60.0]
    assert "fractions of the frame" in payload["coordinates"]
    json.dumps(payload)


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
    from fibsem.applications.autolamella.workflows.interaction import (
        ConfirmDetection,
        InvalidAnswerValue,
    )

    request = ConfirmDetection(detection=_detection())
    name = request.detection.features[0].name

    moved = request.parse_value({"features": [{"name": name, "px": {"x": 9, "y": 7}}]})
    assert moved == {name: Point(9.0, 7.0)}

    for bad in (
        {"features": []},
        {"features": [{"name": "Nope", "px": {"x": 1, "y": 1}}]},
        {"features": [{"name": name, "px": {"x": 10, "y": 1}}]},  # x == width
        {"features": [{"name": name, "px": {"x": 1}}]},
        {"features": [{"name": name, "px": {"x": 1, "y": 1}}] * 2},
    ):
        with pytest.raises(InvalidAnswerValue):
            request.parse_value(bad)


def test_a_spot_burn_value_is_an_action_and_its_edits():
    from fibsem.applications.autolamella.workflows.interaction import (
        InvalidAnswerValue,
        SpotBurnEdit,
        apply_spot_burn_edit,
    )

    settings = SpotBurnSettings(
        coordinates=[Point(0.5, 0.5)], milling_current=60e-12, exposure_time=5.0
    )
    request = RunSpotBurn(settings=settings)
    edit = request.parse_value(
        {
            "action": "edit",
            "points": [{"x": 0, "y": 1}, {"x": 0.5, "y": 0.25}],
            "milling_current": 1e-10,
            "exposure_time": 2,
        }
    )
    assert edit == SpotBurnEdit(
        action="edit",
        points=[Point(0.0, 1.0), Point(0.5, 0.25)],
        milling_current=1e-10,
        exposure_time=2.0,
    )
    edited = apply_spot_burn_edit(settings, edit)
    assert edited.coordinates == [Point(0.0, 1.0), Point(0.5, 0.25)]
    assert (edited.milling_current, edited.exposure_time) == (1e-10, 2.0)
    assert settings.coordinates == [Point(0.5, 0.5)]  # a new settings

    # Only the action: everything stays as it stands.
    assert request.parse_value({"action": "run"}) == SpotBurnEdit(action="run")
    assert request.same_value(settings, SpotBurnEdit(action="run"))

    for bad in (
        {},
        {"points": []},  # no action
        {"action": "accept"},  # not before a burn
        {"action": "run", "points": [{"x": 1.2, "y": 0.5}]},
        {"action": "run", "points": [{"x": "a", "y": 0.5}]},
        {"action": "run", "points": [{"x": float("nan"), "y": 0.5}]},
        {"action": "run", "exposure_time": 0},
        {"action": "run", "exposure_time": 600},
        {"action": "run", "milling_current": True},
        {"action": "run", "milling_current": -1e-10},
        {"action": "run", "dwell": 1},
    ):
        with pytest.raises(InvalidAnswerValue):
            request.parse_value(bad)

    after = RunSpotBurn(settings=settings, passes=1)
    assert after.parse_value({"action": "accept"}).action == "accept"
    with pytest.raises(InvalidAnswerValue):
        after.parse_value({"action": "skip"})


def test_an_answer_says_what_it_decided():
    """Sim session 2026-10-16: the agent card's Recent says what an answer
    decided, not the prompt's instructions."""
    from fibsem.applications.autolamella.workflows.interaction import (
        Confirm,
        MillingDecision,
    )

    image = FibsemImage.generate_blank_image(resolution=(64, 64), hfw=100e-6)
    poi = PickPOI(image=image)
    assert poi.decided(Point(5e-6, -2e-6)) == "Point of interest at (+5.0, −2.0) µm"
    assert poi.decided(None) == "No point of interest"
    area = EditAlignmentArea(initial=FibsemRectangle(0, 0, 1, 1))
    assert area.decided(FibsemRectangle(0.25, 0.5, 0.5, 0.25)) == (
        "Alignment area 50% × 25% of the frame"
    )
    assert Confirm("Go on?").decided(True) is None
    first = RunMillingTask(config=None)
    assert first.decided(MillingDecision(action="skip", config=None)) == (
        "Skip, without milling"
    )
    again = RunMillingTask(config=None, passes=6)
    assert again.decided(MillingDecision(action="accept", config=None)) == (
        "Accept: don't mill again"
    )
    assert again.decided(MillingDecision(action="run", config=None)) == "Mill Again"

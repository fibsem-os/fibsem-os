"""What the agent's waiting request or suggestion would do, drawn on the views
(FIB-1404)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.agent_previews import (  # noqa: E402
    PATTERNS_OVERLAY,
    SUGGESTION_OVERLAY,
    TARGET_OVERLAY,
    AgentPreviews,
    request_patterns,
    request_targets,
)
from fibsem.structures import BeamType, FibsemImage, Point  # noqa: E402
from fibsem.ui.tokens import PURPLE_COLOR  # noqa: E402

_RECTANGLE = {
    "type": "Rectangle",
    "width": 2e-6,
    "height": 1e-6,
    "depth": 2e-7,
    "centre_x": 1e-6,
    "centre_y": 0.0,
}


def _request(status="pending", request_id="ab12"):
    return {
        "request_id": request_id,
        "status": status,
        "actions": [
            {
                "tool": "call_service_command",
                "arguments": {
                    "service": "milling",
                    "command": "prepare",
                    "kwargs": {"patterns": [_RECTANGLE, _RECTANGLE]},
                },
            },
            {
                "tool": "stable_move",
                "arguments": {"dx": 10e-6, "dy": -5e-6, "beam_type": "ION"},
            },
            {"tool": "vertical_move", "arguments": {"dx": 0.0, "dy": 2e-6}},
            {"tool": "move_to_orientation", "arguments": {"orientation": "FIB"}},
        ],
    }


class _Controller:
    def __init__(self):
        self.overlays = {}
        self._images = {
            beam: FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
            for beam in (BeamType.ELECTRON, BeamType.ION)
        }

    def image(self, beam):
        return self._images.get(beam)

    def set_overlay(self, beam, spec):
        self.overlays[(beam, spec.id)] = spec

    def remove_overlay(self, beam, overlay_id):
        self.overlays.pop((beam, overlay_id), None)


def test_a_request_s_patterns_and_move_targets():
    request = _request()
    patterns = request_patterns(request)
    assert len(patterns) == 2
    assert patterns[0].width == pytest.approx(2e-6)
    assert request_targets(request) == [
        (BeamType.ION, Point(10e-6, -5e-6)),
        (BeamType.ION, Point(0.0, 2e-6)),  # vertical_move works in the ion beam
    ]


def test_a_waiting_request_is_drawn_dashed_in_purple_and_comes_down_when_decided():
    controller = _Controller()
    previews = AgentPreviews(lambda: controller)
    previews.show_request(_request())

    spec = controller.overlays[(BeamType.ION, PATTERNS_OVERLAY)]
    assert spec.colour == PURPLE_COLOR and spec.dashed
    assert len(spec.stages[0].define_patterns()) == 2
    assert spec.stages[0].name == "Agent · waiting for you"
    targets = controller.overlays[(BeamType.ION, TARGET_OVERLAY)]
    # 1 µm a pixel at 100 µm over 100 px wide... 200 px: 0.5 µm a pixel, centre (100, 50).
    assert targets.points == [(120.0, 60.0), (100.0, 46.0)]
    assert targets.color == PURPLE_COLOR and targets.modal

    previews.show_request(_request(status="applying"))
    assert controller.overlays == {}
    assert previews.drawn == []


def test_a_new_request_replaces_the_last_one_s_previews():
    controller = _Controller()
    previews = AgentPreviews(lambda: controller)
    previews.show_request(_request())
    only_a_mill = _request(request_id="cd34")
    only_a_mill["actions"] = only_a_mill["actions"][:1]
    previews.show_request(only_a_mill)
    assert set(controller.overlays) == {(BeamType.ION, PATTERNS_OVERLAY)}


def test_a_suggested_point_is_drawn_until_the_suggestion_goes():
    controller = _Controller()
    previews = AgentPreviews(lambda: controller)
    previews.show_suggested_point((40.0, 30.0))
    spec = controller.overlays[(BeamType.ION, SUGGESTION_OVERLAY)]
    assert spec.points == [(40.0, 30.0)]
    assert spec.legend_label == "Agent's suggestion"
    previews.show_suggested_point(None)
    assert controller.overlays == {}


def test_without_views_nothing_is_drawn():
    previews = AgentPreviews(lambda: None)
    previews.show_request(_request())
    previews.show_suggested_point((1.0, 2.0))
    assert previews.drawn == []


def test_a_suggested_point_of_interest_is_in_the_question_s_pixels():
    from fibsem.applications.autolamella.ui.qt_responder import QtResponder
    from fibsem.applications.autolamella.workflows.interaction import PickPOI

    image = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    request = PickPOI(image=image)
    assert QtResponder._suggested_pixel(request, Point(-10e-6, 5e-6)) == (80.0, 40.0)
    assert QtResponder._suggested_pixel(request, None) is None

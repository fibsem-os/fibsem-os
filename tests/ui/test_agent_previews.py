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
                "tool": "service_call_command",
                "arguments": {
                    "service": "milling",
                    "command": "prepare",
                    "kwargs": {"patterns": [_RECTANGLE, _RECTANGLE]},
                },
            },
            {
                "tool": "stage_stable_move",
                "arguments": {"dx": 10e-6, "dy": -5e-6, "beam_type": "ION"},
            },
            {"tool": "stage_vertical_move", "arguments": {"dx": 0.0, "dy": 2e-6}},
            {"tool": "stage_move_to_orientation", "arguments": {"orientation": "FIB"}},
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
    assert spec.legend_loc == "lower left"  # clear of the view's toolbar
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


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def emit(self, *args):
        for slot in self.slots:
            slot(*args)


class _FramedController(_Controller):
    """A controller whose views take new frames, and say so."""

    def __init__(self):
        super().__init__()
        self.frame_changed = _Signal()

    def show(self, beam, image):
        self._images[beam] = image
        self.frame_changed.emit(beam)


def test_a_move_target_stays_on_the_sample_when_the_view_takes_another_frame():
    # FIB-1411: the cross was fixed at its first pixels; on a closer frame it
    # pointed somewhere else.
    controller = _FramedController()
    previews = AgentPreviews(lambda: controller)
    request = _request()
    request["actions"] = request["actions"][1:2]  # the stable move, 10 µm right
    previews.show_request(request)
    assert controller.overlays[(BeamType.ION, TARGET_OVERLAY)].points == [
        pytest.approx((120.0, 60.0))
    ]
    closer = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=50e-6)
    controller.show(BeamType.ION, closer)
    assert controller.overlays[(BeamType.ION, TARGET_OVERLAY)].points == [
        pytest.approx((140.0, 70.0))
    ]


def test_a_move_target_with_no_frame_yet_is_drawn_when_one_arrives():
    controller = _FramedController()
    controller._images[BeamType.ION] = None
    previews = AgentPreviews(lambda: controller)
    request = _request()
    request["actions"] = request["actions"][1:2]
    previews.show_request(request)
    assert (BeamType.ION, TARGET_OVERLAY) not in controller.overlays
    frame = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    controller.show(BeamType.ION, frame)
    assert controller.overlays[(BeamType.ION, TARGET_OVERLAY)].points == [
        pytest.approx((120.0, 60.0))
    ]
    previews.show_request(None)
    controller.show(BeamType.ION, frame)
    assert controller.overlays == {}


def test_a_run_stage_or_run_task_request_draws_the_stages_it_would_mill():
    # FIB-1411: run_stage and run_task were drawn nothing, though their stages
    # carry every pattern.
    from fibsem.applications.autolamella.ui.agent_previews import request_stages
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig

    trench = FibsemMillingStage(name="Trench")
    off = FibsemMillingStage(name="Off", enabled=False)
    polish = FibsemMillingStage(name="Polish")
    task = FibsemMillingTaskConfig.from_stages([off, polish], name="Polishing")

    def call(command, **kwargs):
        return {
            "tool": "service_call_command",
            "arguments": {"service": "milling", "command": command, "kwargs": kwargs},
        }

    request = {
        "request_id": "ef56",
        "status": "pending",
        "actions": [
            call("run_stage", stage=trench.to_dict()),
            call("run_task", config=task.to_dict()),
            call("run_task", config={"not": "a config"}),
        ],
    }
    assert [stage.name for stage in request_stages(request)] == ["Trench", "Polish"]

    controller = _Controller()
    AgentPreviews(lambda: controller).show_request(request)
    spec = controller.overlays[(BeamType.ION, PATTERNS_OVERLAY)]
    assert [stage.name for stage in spec.stages] == ["Trench", "Polish"]
    assert spec.dashed and spec.colour == PURPLE_COLOR


def test_patterns_past_the_edge_of_the_field_are_drawn_again_in_red():
    # FIB-1411: patterns at (-30, +30) µm in a 100 µm field (y spans ±33 µm)
    # reached y ≈ 36-39 µm. The Demo milled them; a real instrument refuses.
    from fibsem.applications.autolamella.ui.agent_previews import (
        OUTSIDE_OVERLAY,
        request_outside_field,
    )
    from fibsem.structures import FibsemMillingSettings
    from fibsem.ui.tokens import RED_COLOR

    inside = dict(_RECTANGLE)
    outside = dict(_RECTANGLE, centre_x=-30e-6, centre_y=30e-6, height=8e-6)
    request = {
        "request_id": "gh78",
        "status": "pending",
        "actions": [
            {
                "tool": "service_call_command",
                "arguments": {
                    "service": "milling",
                    "command": "prepare",
                    "kwargs": {
                        "settings": FibsemMillingSettings(hfw=100e-6).to_dict(),
                        "patterns": [inside, outside],
                    },
                },
            }
        ],
    }
    (past,) = request_outside_field(request)
    assert past.centre_y == pytest.approx(30e-6)

    controller = _Controller()
    AgentPreviews(lambda: controller).show_request(request)
    spec = controller.overlays[(BeamType.ION, OUTSIDE_OVERLAY)]
    assert spec.colour == RED_COLOR
    assert spec.stages[0].name == "Agent · outside the field"
    assert len(spec.stages[0].define_patterns()) == 1
    # Both are drawn in purple too: the red is on top of what would mill.
    assert (
        len(
            controller.overlays[(BeamType.ION, PATTERNS_OVERLAY)]
            .stages[0]
            .define_patterns()
        )
        == 2
    )


def test_a_pattern_s_extent_from_its_numbers():
    import math

    from fibsem.milling.patterning.utils import outside_field, pattern_extent
    from fibsem.structures import (
        FibsemCircleSettings,
        FibsemLineSettings,
        FibsemRectangleSettings,
    )

    turned = FibsemRectangleSettings(
        width=4e-6,
        height=2e-6,
        depth=1e-6,
        centre_x=0,
        centre_y=0,
        rotation=math.pi / 2,
    )
    assert pattern_extent(turned) == pytest.approx((-1e-6, 1e-6, -2e-6, 2e-6))
    ring = FibsemCircleSettings(radius=5e-6, depth=1e-6, centre_x=1e-6, centre_y=0)
    assert pattern_extent(ring) == pytest.approx((-4e-6, 6e-6, -5e-6, 5e-6))
    line = FibsemLineSettings(
        start_x=-1e-6, end_x=2e-6, start_y=3e-6, end_y=0, depth=1e-6
    )
    assert pattern_extent(line) == pytest.approx((-1e-6, 2e-6, 0, 3e-6))
    assert not outside_field(ring, 12e-6, 10e-6)
    assert outside_field(ring, 11e-6, 10e-6)

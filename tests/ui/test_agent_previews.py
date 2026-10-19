"""What the agent's waiting request would do: on the card, on the
image the values were measured on; on the views, what is about to happen there
(FIB-1404, FIB-1411)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.structures import BeamType, FibsemImage, Point  # noqa: E402
from fibsem.ui.agent.previews import (  # noqa: E402
    AgentPreviews,
    request_patterns,
    request_targets,
)
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


def test_a_waiting_request_is_drawn_on_the_card_only_and_comes_down_when_decided():
    # Patrick, 2026-10-15: the agent's previews go on the card, not the live view.
    controller = _Controller()
    card = _Card()
    previews = AgentPreviews(lambda: controller, show_picture=card)
    previews.show_request(_request())
    frame, layer = _layer(previews.request_picture)
    assert frame is controller.image(BeamType.ION)
    assert len(layer.patterns) == 2
    # 0.5 µm a pixel, centre (100, 50). A move target is a crosshair.
    assert layer.targets == [pytest.approx((120.0, 60.0)), pytest.approx((100.0, 46.0))]
    assert layer.crosses == [] and ("target", "move target") in layer.keys
    assert controller.overlays == {}
    assert card.shown is not None

    previews.show_request(_request(status="applying"))
    assert previews.request_picture is None and card.shown is None


def test_a_new_request_replaces_the_last_one_s_picture():
    controller = _Controller()
    previews = AgentPreviews(lambda: controller)
    previews.show_request(_request())
    only_a_mill = _request(request_id="cd34")
    only_a_mill["actions"] = only_a_mill["actions"][:1]
    previews.show_request(only_a_mill)
    _, layer = _layer(previews.request_picture)
    assert len(layer.patterns) == 2 and layer.crosses == []


def test_a_request_filed_before_the_views_had_images_is_drawn_once_they_do():
    """Sim session 2026-10-16: no picture came until the request was filed
    again; the picture was worked out once per request."""
    controller = _Controller()
    controller._images = {}
    card = _Card()
    previews = AgentPreviews(lambda: controller, show_picture=card)
    previews.show_request(_request())
    assert previews.request_picture is None
    controller._images = _Controller()._images
    previews.show_request(_request())
    frame, _ = _layer(previews.request_picture)
    assert frame is controller.image(BeamType.ION)
    assert card.shown is not None


def test_without_views_nothing_is_drawn():
    card = _Card()
    previews = AgentPreviews(lambda: None, show_picture=card)
    previews.show_request(_request())
    assert previews.request_picture is None


class _Card:
    """What the card was given to show under Why."""

    def __init__(self):
        self.shown = None
        self.notes = ""

    def __call__(self, image, caption, legend=()):
        self.shown = (image, caption) if image is not None else None
        self.notes = caption if image is None else ""


def _layer(picture, colour=PURPLE_COLOR):
    frame, layers = picture
    (layer,) = [layer for layer in layers if layer.colour == colour]
    return frame, layer


def test_a_move_target_on_another_frame_is_re_projected_onto_it():
    # FIB-1411: measured on one frame, drawn on another, the target stays on
    # the sample.
    from fibsem.ui.agent.previews import (
        _Mark,
        _offset_pixel,
    )

    wide = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    closer = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=50e-6)
    mark = _Mark(
        BeamType.ION, "target", [(wide, _offset_pixel(wide, Point(10e-6, -5e-6)))]
    )
    assert mark.pixels(wide) == [pytest.approx((120.0, 60.0))]
    assert mark.pixels(closer) == [pytest.approx((140.0, 70.0))]


def test_a_run_stage_or_run_task_request_draws_the_stages_it_would_mill():
    # FIB-1411: run_stage and run_task were drawn nothing, though their stages
    # carry every pattern.
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.ui.agent.previews import request_stages

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
    # The same through the milling tools (#1841): their arguments are the kwargs.
    tools = {
        "request_id": "ef57",
        "status": "pending",
        "actions": [
            {"tool": "milling_run_stage", "arguments": {"stage": trench.to_dict()}},
            {"tool": "milling_run_task", "arguments": {"config": task.to_dict()}},
        ],
    }
    assert [stage.name for stage in request_stages(tools)] == ["Trench", "Polish"]

    controller = _Controller()
    previews = AgentPreviews(lambda: controller)
    previews.show_request(request)
    _, layer = _layer(previews.request_picture)
    stages = request_stages(request)
    assert len(layer.patterns) == sum(len(s.define_patterns()) for s in stages)
    # Areas to mill: filled, see-through (FIB-1424), not dashed outlines.
    assert layer.fill and layer.keys == [("fill", "patterns")]


def test_patterns_past_the_edge_of_the_field_are_drawn_again_in_red():
    # FIB-1411: patterns at (-30, +30) µm in a 100 µm field (y spans ±33 µm)
    # reached y ≈ 36-39 µm. The Demo milled them; a real instrument refuses.
    from fibsem.structures import FibsemMillingSettings
    from fibsem.ui.agent.previews import (
        request_outside_field,
    )
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
    previews = AgentPreviews(lambda: controller)
    previews.show_request(request)
    _, red = _layer(previews.request_picture, RED_COLOR)
    assert len(red.patterns) == 1
    # Both are drawn in purple too: the red is on top of what would mill.
    _, purple = _layer(previews.request_picture)
    assert len(purple.patterns) == 2


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


def test_a_pattern_s_outline_from_its_numbers():
    import math

    from fibsem.milling.patterning.utils import pattern_outline
    from fibsem.structures import FibsemLineSettings, FibsemRectangleSettings

    turned = FibsemRectangleSettings(
        width=4e-6,
        height=2e-6,
        depth=1e-6,
        centre_x=1e-6,
        centre_y=0,
        rotation=math.pi / 2,
    )
    (ring,) = pattern_outline(turned)
    assert ring == [
        pytest.approx((2e-6, -2e-6)),
        pytest.approx((2e-6, 2e-6)),
        pytest.approx((0.0, 2e-6)),
        pytest.approx((0.0, -2e-6)),
    ]
    line = FibsemLineSettings(start_x=0, end_x=1e-6, start_y=0, end_y=0, depth=1e-6)
    assert pattern_outline(line) == [[(0, 0), (1e-6, 0)]]


def test_a_review_decision_s_point_is_drawn_on_the_card_on_its_own_image():
    # FIB-1411: proposal_decide's POI was only JSON on the card. It is drawn on
    # the proposal's reference image, under Why; not on the views.
    reference = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=200e-6)
    asked = []

    def reference_image(arguments):
        asked.append(arguments["proposal_id"])
        return reference

    controller = _Controller()  # its FIB frame: 100 µm wide
    card = _Card()
    previews = AgentPreviews(
        lambda: controller, reference_image=reference_image, show_picture=card
    )
    request = {
        "request_id": "ij90",
        "status": "pending",
        "actions": [
            {
                "tool": "proposal_decide",
                "what": "01-full-magpie · Setup Lamella",
                "arguments": {
                    "item_id": "01",
                    "task_name": "Setup Lamella",
                    "proposal_id": "p1",
                    "outcome": "Confirmed",
                    "values": {"poi": {"x": 10e-6, "y": 5e-6}},
                },
            }
        ],
    }
    previews.show_request(request)
    assert asked == ["p1"]
    frame, layer = _layer(previews.request_picture)
    assert frame is reference
    # 10 µm right, 5 µm up of the centre, on a 1 µm/px image.
    assert layer.crosses == [pytest.approx((110.0, 45.0))]
    assert controller.overlays == {}
    image, caption = card.shown
    assert image.width() == 200
    assert caption.startswith("01-full-magpie · Setup Lamella · FIB")
    # The legend on it names what the cross is.
    assert layer.keys == [("cross", "point of interest")]
    previews.show_request(None)
    assert previews.request_picture is None and card.shown is None


def _pass_request(proposal, **arguments):
    """A waiting request to decide *proposal*, a loop's pass."""
    return {
        "request_id": "pq34",
        "status": "pending",
        "actions": [
            {
                "tool": "proposal_decide",
                "what": "01-full-magpie · Rough Milling",
                "arguments": {
                    "item_id": "01",
                    "task_name": "Rough Milling",
                    "proposal_id": proposal.id,
                    "outcome": "Confirmed",
                    **arguments,
                },
            }
        ],
    }


def _pass_previews(proposal, reference):
    card = _Card()
    previews = AgentPreviews(
        lambda: _Controller(),
        reference_image=lambda arguments: reference,
        show_picture=card,
        pending_proposal=lambda arguments: (
            proposal if arguments.get("proposal_id") == proposal.id else None
        ),
    )
    return previews, card


def test_a_request_to_run_a_mill_pass_draws_its_patterns_as_patched():
    """A pass the agent asks to run draws what it would mill, its patch
    applied, on the image the pass was asked on: what the operator approves."""
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.workflows.proposals import MILLING, Proposal

    config = FibsemMillingTaskConfig.from_stages(
        [FibsemMillingStage(name="Trench")], name="Rough Milling"
    )
    proposal = Proposal(kind=MILLING, values={"milling_config": config})
    reference = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    previews, card = _pass_previews(proposal, reference)

    previews.show_request(_pass_request(proposal, action="run"))
    frame, layers = previews.request_picture
    assert frame is reference
    purple = layers[1]
    assert purple.fill and len(purple.patterns) == len(
        config.stages[0].define_patterns()
    )
    assert card.shown[1].startswith("01-full-magpie · Rough Milling · FIB")

    wider = config.stages[0].pattern.width * 2
    previews.show_request(None)
    previews.show_request(
        _pass_request(
            proposal, action="run", patch={"stages.Trench.pattern.width": wider}
        )
    )
    _, layers = previews.request_picture
    assert layers[1].patterns != purple.patterns
    # the pass itself is not changed
    assert config.stages[0].pattern.width == pytest.approx(wider / 2)


def test_a_request_to_accept_a_mill_pass_outlines_what_is_not_milled():
    from fibsem.milling.base import FibsemMillingStage
    from fibsem.milling.tasks import FibsemMillingTaskConfig
    from fibsem.workflows.proposals import MILLING, Proposal

    config = FibsemMillingTaskConfig.from_stages(
        [FibsemMillingStage(name="Trench")], name="Fiducial"
    )
    proposal = Proposal(kind=MILLING, values={"milling_config": config})
    reference = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    previews, _card = _pass_previews(proposal, reference)
    previews.show_request(_pass_request(proposal, action="accept"))
    _, layers = previews.request_picture
    assert not layers[1].patterns, "nothing to mill"
    unmilled = layers[-1]
    assert unmilled.patterns and not unmilled.fill
    assert unmilled.keys == [("outline", "not milled")]


def test_a_request_to_run_a_spot_burn_pass_draws_its_points():
    from fibsem.imaging.spot import SpotBurnSettings
    from fibsem.workflows.proposals import SPOT_BURN, Proposal

    proposal = Proposal(
        kind=SPOT_BURN,
        values={"spot_burn": SpotBurnSettings(coordinates=[Point(0.5, 0.5)])},
    )
    reference = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    previews, _card = _pass_previews(proposal, reference)
    # The call's own points, not the pass's.
    values = {
        "spot_burn": {
            "coordinates": [{"x": 0.25, "y": 0.5}, {"x": 0.75, "y": 0.25}],
            "milling_current": 60e-12,
            "exposure_time": 5,
        }
    }
    previews.show_request(_pass_request(proposal, action="run", values=values))
    frame, layers = previews.request_picture
    assert frame is reference
    dots = [d for layer in layers for d in layer.dots]
    assert dots == [pytest.approx((50.0, 50.0)), pytest.approx((150.0, 25.0))]
    # Skipped, nothing is burned: nothing drawn.
    previews.show_request(None)
    previews.show_request(_pass_request(proposal, action="skip"))
    assert previews.request_picture is None


def _item_edit(request_id="kl12"):
    return {
        "request_id": request_id,
        "status": "pending",
        "actions": [
            {
                "tool": "item_update",
                "what": "01-full-magpie",
                "arguments": {
                    "item_name": "01-full-magpie",
                    "patch": {
                        "poi.x": -10e-6,
                        "poi.y": 0.0,
                        "alignment_area.left": 0.25,
                        "alignment_area.top": 0.5,
                        "alignment_area.width": 0.5,
                        "alignment_area.height": 0.25,
                    },
                    "version": "v",
                },
            }
        ],
    }


def test_an_item_edit_s_point_and_alignment_area_are_drawn_on_the_item_s_image():
    # FIB-1424: they were drawn on the live FIB frame, which was 114 µm away
    # at another lamella. They are in the item's own frame: its last FIB image.
    controller = _Controller()
    item_image = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    asked = []

    def image_of(item):
        asked.append(item)
        return item_image

    card = _Card()
    previews = AgentPreviews(lambda: controller, item_image=image_of, show_picture=card)
    previews.show_request(_item_edit())
    assert asked == ["01-full-magpie"]
    frame, layer = _layer(previews.request_picture)
    assert frame is item_image  # 0.5 µm/px
    assert layer.crosses == [pytest.approx((80.0, 50.0))]
    assert layer.areas == [[pytest.approx((50.0, 50.0)), pytest.approx((150.0, 75.0))]]
    assert controller.overlays == {}
    # Captioned with whose image it is.
    _, caption = card.shown
    assert caption.startswith("01-full-magpie · FIB")


def test_an_item_edit_s_nested_alignment_area_is_drawn_too():
    # Sim session 2026-10-16 F4: the server takes {"alignment_area": {...}};
    # only the dotted keys were drawn.
    item_image = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    request = _item_edit()
    request["actions"][0]["arguments"]["patch"] = {
        "alignment_area": {"left": 0.25, "top": 0.5, "width": 0.5, "height": 0.25}
    }
    previews = AgentPreviews(
        lambda: _Controller(), item_image=lambda item: item_image, show_picture=_Card()
    )
    previews.show_request(request)
    _, layer = _layer(previews.request_picture)
    assert layer.areas == [[pytest.approx((50.0, 50.0)), pytest.approx((150.0, 75.0))]]


def test_an_item_edit_without_an_image_of_the_item_says_so():
    # Never the live view instead: it may be at another lamella.
    controller = _Controller()
    card = _Card()
    previews = AgentPreviews(lambda: controller, show_picture=card)
    previews.show_request(_item_edit("mn56"))
    assert previews.request_picture is None
    assert card.notes == "No FIB image of 01-full-magpie yet to draw on"


def test_the_card_s_picture_is_the_image_with_its_marks_drawn():
    from PyQt5.QtGui import QColor

    from fibsem.ui.agent.previews import (
        _Layer,
        draw_picture,
    )

    frame = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    layer = _Layer(PURPLE_COLOR)
    layer.crosses.append((100.0, 50.0))
    image = draw_picture(frame, [layer])
    assert (image.width(), image.height()) == (200, 100)
    assert image.pixelColor(100, 50) == QColor(PURPLE_COLOR)
    assert image.pixelColor(10, 10) != QColor(PURPLE_COLOR)


def test_a_task_config_edit_draws_the_milling_as_it_would_be_and_as_it_is():
    # FIB-1411: pattern edits through item_update_task_config showed nothing.
    from fibsem.applications.autolamella.workflows.tasks.basic_milling import (
        BasicMillingTaskConfig,
    )

    config = BasicMillingTaskConfig()
    width = config.milling["milling"].stages[0].pattern.width
    asked = []

    def task_config(item, task):
        asked.append((item, task))
        return config

    controller = _Controller()
    card = _Card()
    previews = AgentPreviews(
        lambda: controller, task_config=task_config, show_picture=card
    )
    request = {
        "request_id": "mn34",
        "status": "pending",
        "actions": [
            {
                "tool": "item_update_task_config",
                "arguments": {
                    "item_name": "01-full-magpie",
                    "task_name": "Basic Milling",
                    "patch": {"milling.milling.stages.0.pattern.width": width * 2},
                    "version": "v",
                },
            }
        ],
    }
    previews.show_request(request)
    assert asked == [("01-full-magpie", "Basic Milling")]
    _, grey = _layer(previews.request_picture, "#ffffff")
    _, purple = _layer(previews.request_picture)
    assert purple.patterns[0].width == pytest.approx(width * 2)
    assert grey.patterns[0].width == pytest.approx(width)
    # As it is now: solid white, seen on a grey image; the legend says which
    # is which.
    assert not grey.fill and purple.fill
    assert grey.keys == [("outline", "as now")]
    assert purple.keys == [("fill", "after the edit")]
    # The item's own config is untouched: the edit was drawn on a copy.
    assert config.milling["milling"].stages[0].pattern.width == pytest.approx(width)


def test_the_legend_is_drawn_on_the_picture_at_its_own_size(qapp):
    # Patrick, 2026-10-15: the key goes on the image, the same size however
    # wide the image shows.
    from PyQt5.QtGui import QColor, QImage, QPainter

    from fibsem.ui.agent.previews import draw_legend

    sizes = []
    for width in (300, 900):
        image = QImage(width, 200, QImage.Format_RGB32)
        image.fill(QColor("#ffffff"))
        painter = QPainter(image)
        draw_legend(painter, [(PURPLE_COLOR, "dot", "spot burn points")], 12)
        painter.end()
        # The box's darkened corner, and how far right it reaches.
        assert image.pixelColor(10, 10).lightness() < 128
        sizes.append(
            max(x for x in range(width) if image.pixelColor(x, 10).lightness() < 128)
        )
    assert sizes[0] == sizes[1]


def test_patterns_to_mill_are_filled_and_an_area_is_only_outlined(qapp):
    # FIB-1424 (decided): patterns as see-through solid areas, one colour;
    # dashed outlines only for what is not milled.
    from fibsem.ai.agent.request_text import parse_patterns
    from fibsem.ui.agent.previews import (
        _Layer,
        draw_picture,
    )

    frame = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    milled = _Layer(PURPLE_COLOR, fill=True)
    milled.patterns = parse_patterns(
        [{**_RECTANGLE, "width": 40e-6, "height": 20e-6, "centre_x": -25e-6}]
    )
    area = _Layer(PURPLE_COLOR)
    area.areas = [[(120.0, 30.0), (180.0, 70.0)]]
    image = draw_picture(frame, [milled, area])
    background = image.pixelColor(100, 95)
    inside_pattern = image.pixelColor(50, 50)  # 25 µm left of centre
    inside_area = image.pixelColor(150, 50)
    assert inside_pattern != background
    assert inside_pattern.blue() > inside_pattern.green()  # tinted purple
    assert inside_area == background


def _one(tool, arguments):
    return {
        "request_id": f"{tool}-{sorted(arguments)}",
        "status": "pending",
        "actions": [{"tool": tool, "arguments": arguments}],
    }


def test_a_new_lamella_is_marked_where_it_would_go():
    # FIB-1411: item_add drew nothing.
    controller = _Controller()  # 200 × 100 px
    overview = FibsemImage.generate_blank_image(resolution=(400, 300), hfw=1e-3)
    asked = []

    def overview_image(grid, filename):
        asked.append((grid, filename))
        return overview

    previews = AgentPreviews(lambda: controller, overview_image=overview_image)
    # Where the stage is: the centre of the FIB view.
    previews.show_request(_one("item_add", {}))
    frame, layer = _layer(previews.request_picture)
    assert frame is controller.image(BeamType.ION)
    assert layer.crosses == [pytest.approx((100.0, 50.0))]
    assert layer.keys == [("cross", "new lamella")]
    # On a grid's overview: on that overview.
    point = {"grid_name": "G1", "filename": "overview.tif", "fraction": [0.25, 0.5]}
    previews.show_request(_one("item_add", {"overview": point}))
    frame, layer = _layer(previews.request_picture)
    assert asked == [("G1", "overview.tif")] and frame is overview
    assert layer.crosses == [pytest.approx((100.0, 150.0))]


def test_a_pose_move_and_a_pose_set_are_marked_through_the_stage(monkeypatch):
    # FIB-1411: item_move_to_pose and item_set_pose drew nothing.
    from fibsem.geometry import projection
    from fibsem.structures import FibsemStagePosition

    pose = FibsemStagePosition(x=1e-6)
    placed = []

    def stage_to_pixel(frame, position):
        placed.append(position)
        return (30.0, 20.0)

    monkeypatch.setattr(projection, "stage_to_pixel", stage_to_pixel)
    controller = _Controller()
    previews = AgentPreviews(
        lambda: controller,
        item_pose=lambda item, name: (
            pose if (item, name) == ("01", "MILLING") else None
        ),
    )
    previews.show_request(
        _one("item_move_to_pose", {"item_name": "01", "pose": "MILLING"})
    )
    _, layer = _layer(previews.request_picture)
    assert placed and all(position is pose for position in placed)
    assert layer.targets == [(30.0, 20.0)]
    assert layer.keys == [("target", "01 MILLING pose")]
    # Set: the new pose where the stage is, the pose as it is now in white.
    previews.show_request(_one("item_set_pose", {"item_name": "01", "pose": "MILLING"}))
    _, purple = _layer(previews.request_picture)
    _, white = _layer(previews.request_picture, "#ffffff")
    assert purple.targets == [pytest.approx((100.0, 50.0))]
    assert white.targets == [(30.0, 20.0)]
    assert white.keys == [("target", "MILLING pose now")]


def test_a_mark_off_the_picture_is_not_drawn_and_its_key_says_so(monkeypatch):
    # Sim session 2026-10-15b (F3): the pose now was keyed but off the left edge.
    from fibsem.geometry import projection
    from fibsem.structures import FibsemStagePosition

    monkeypatch.setattr(projection, "stage_to_pixel", lambda frame, pose: (-537, 43))
    controller = _Controller()  # 200 × 100 px
    previews = AgentPreviews(
        lambda: controller, item_pose=lambda item, name: FibsemStagePosition(x=0.0)
    )
    previews.show_request(_one("item_set_pose", {"item_name": "01", "pose": "MILLING"}))
    _, white = _layer(previews.request_picture, "#ffffff")
    assert white.targets == []
    assert white.keys == [("target", "MILLING pose now, off the picture")]


def test_a_pose_off_the_view_is_drawn_on_its_grid_overview(monkeypatch):
    # A pose the FIB view does not show is drawn on the item's grid overview.
    from fibsem.geometry import projection
    from fibsem.structures import FibsemStagePosition

    controller = _Controller()  # 200 × 100 px
    overview = FibsemImage.generate_blank_image(resolution=(400, 300), hfw=1e-3)
    pose = FibsemStagePosition(x=1e-3)
    monkeypatch.setattr(
        projection,
        "stage_to_pixel",
        lambda frame, position: (120.0, 80.0) if frame is overview else (-900, 40),
    )
    asked = []
    previews = AgentPreviews(
        lambda: controller,
        item_pose=lambda item, name: pose,
        item_overview=lambda item: asked.append(item) or overview,
    )
    previews.show_request(
        _one("item_move_to_pose", {"item_name": "01", "pose": "MILLING"})
    )
    frame, layer = _layer(previews.request_picture)
    assert asked == ["01"] and frame is overview
    assert layer.targets == [(120.0, 80.0)]
    assert layer.keys == [("target", "01 MILLING pose, on the grid overview")]


def test_a_pose_on_the_view_stays_on_the_view(monkeypatch):
    from fibsem.geometry import projection
    from fibsem.structures import FibsemStagePosition

    controller = _Controller()
    monkeypatch.setattr(projection, "stage_to_pixel", lambda frame, pose: (30, 20))
    asked = []
    previews = AgentPreviews(
        lambda: controller,
        item_pose=lambda item, name: FibsemStagePosition(x=0.0),
        item_overview=lambda item: asked.append(item),
    )
    previews.show_request(
        _one("item_move_to_pose", {"item_name": "01", "pose": "MILLING"})
    )
    frame, _ = _layer(previews.request_picture)
    assert frame is controller.image(BeamType.ION) and asked == []


def test_a_pose_off_the_view_with_no_overview_draws_no_picture(monkeypatch):
    from fibsem.geometry import projection
    from fibsem.structures import FibsemStagePosition

    controller = _Controller()
    monkeypatch.setattr(projection, "stage_to_pixel", lambda frame, pose: (-900, 40))
    previews = AgentPreviews(
        lambda: controller,
        item_pose=lambda item, name: FibsemStagePosition(x=0.0),
        item_overview=lambda item: None,
    )
    previews.show_request(
        _one("item_move_to_pose", {"item_name": "01", "pose": "MILLING"})
    )
    # Nothing to see on the view: no picture, as before.
    assert previews.request_picture is None


def test_past_the_field_off_the_picture_says_so():
    from fibsem.structures import FibsemRectangleSettings
    from fibsem.ui.agent.previews import (
        _key_past_field,
        _Layer,
    )

    frame = FibsemImage.generate_blank_image(resolution=(200, 100), hfw=100e-6)
    layer = _Layer("#ff0000")
    layer.patterns = [
        FibsemRectangleSettings(
            width=4e-6, height=4e-6, depth=1e-6, centre_x=72e-6, centre_y=0.0
        )
    ]
    _key_past_field(frame, layer)
    assert layer.keys == [("outline", "past the field, off the picture")]
    seen = _Layer("#ff0000")
    seen.patterns = [
        FibsemRectangleSettings(
            width=4e-6, height=4e-6, depth=1e-6, centre_x=48e-6, centre_y=0.0
        )
    ]
    _key_past_field(frame, seen)
    assert seen.keys == [("outline", "past the field")]


def test_the_agent_s_own_image_is_the_picture_with_the_values_on_it():
    # FIB-1411: the agent names the image its evidence is in; the card shows it
    # under Why, the request's values drawn on it, even with none to draw.
    evidence = FibsemImage.generate_blank_image(resolution=(400, 200), hfw=100e-6)
    evidence.metadata.image_settings.beam_type = BeamType.ELECTRON
    controller = _Controller()
    card = _Card()
    previews = AgentPreviews(
        lambda: controller, show_picture=card, load_image=lambda path: evidence
    )
    move = {"dx": 10e-6, "dy": 0.0, "beam_type": "ELECTRON"}
    request = _one("stage_stable_move", move)
    request["image"] = "/exp/ref_trench.tif"
    previews.show_request(request)
    frame, layer = _layer(previews.request_picture)
    assert frame is evidence
    # 0.25 µm a pixel: 10 µm right of the centre (200, 100).
    assert layer.targets == [pytest.approx((240.0, 100.0))]
    assert card.shown[1].startswith("The agent's image · SEM")
    stop = _one("workflow_stop", {"reason": "the trench is off"})
    stop["image"] = "/exp/ref_trench.tif"
    previews.show_request(stop)
    assert previews.request_picture[0] is evidence

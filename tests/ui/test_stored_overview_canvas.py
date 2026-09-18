"""The stored-overview canvas: images placed from their own metadata, positions
marked on them, nothing live. See the module docstring."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

import fibsem.config as fibsem_config
from fibsem import utils
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import BeamType, FibsemImage, FibsemStagePosition, ImageSettings
from fibsem.ui.widgets.stored_overview_canvas import VIEW_FM, StoredOverviewCanvas

_app = QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def microscope():
    scope, _ = utils.setup_session(manufacturer="Demo")
    yield scope
    scope.disconnect()


@pytest.fixture(scope="module")
def arctis():
    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    assert scope.fm is not None
    yield scope
    scope.disconnect()


@pytest.fixture
def widget(destroy_widgets_after_test):
    w = StoredOverviewCanvas()
    w.resize(800, 600)
    w.show()
    yield w
    w.close()


def _beam_image(
    microscope, position, beam=BeamType.ELECTRON, shape=(64, 96), hfw=200e-6, item=None
):
    image = FibsemImage.generate_blank_image(resolution=(shape[1], shape[0]), hfw=hfw)
    image.data = (np.random.default_rng(1).random(shape) * 255).astype(np.uint8)
    state = microscope.get_microscope_state(beam_type=beam)
    state.stage_position = position
    image.metadata.image_settings = ImageSettings(hfw=hfw, beam_type=beam)
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    if item is not None:
        image.metadata.experiment.item_id, image.metadata.experiment.item_name = item
    return image


def _fm_image(arctis, position, size=128, pixel_size=1e-7):
    image = FluorescenceImage(
        data=(np.random.default_rng(2).random((1, 1, size, size)) * 4000).astype(
            np.uint16
        ),
        metadata=FluorescenceImageMetadata(
            acquisition_date="2026-09-14T10:00:00",
            pixel_size_x=pixel_size,
            pixel_size_y=pixel_size,
            stage_position=position,
            channels=[
                FluorescenceChannelMetadata(
                    name="GFP",
                    excitation_wavelength=488.0,
                    power=0.5,
                    exposure_time=0.1,
                    gain=1.0,
                    offset=0.0,
                    color="cyan",
                )
            ],
        ),
    )
    image.metadata.geometry = arctis.fm_image_geometry()
    return image


def _settle(widget):
    widget.canvas._detail_timer.stop()
    widget.canvas.refresh_detail()
    _app.processEvents()


def _at(base, dx=0.0, dy=0.0, name=None):
    p = FibsemStagePosition(x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t)
    p.name = name
    return p


class TestPlacingImages:
    def test_a_beam_overview_is_placed_from_its_own_metadata(self, widget, microscope):
        base = microscope.get_stage_position()
        rid = widget.set_image(_beam_image(microscope, base, item=("g-1", "aspen")))
        assert rid is not None
        assert widget.view.startswith("SEM @")
        assert widget.item_of(rid) == ("g-1", "aspen")
        _settle(widget)
        (cx, cy), _ = widget.overviews[0].extent
        assert widget.record_at(*widget.canvas.metres_to_canvas(cx, cy)).id == rid
        assert widget.record_at(*widget.canvas.metres_to_canvas(cx + 5e-3, cy)) is None

    def test_an_image_with_no_position_is_refused(self, widget, microscope):
        image = _beam_image(microscope, microscope.get_stage_position())
        image.metadata.microscope_state.stage_position = None
        assert widget.set_image(image) is None
        assert widget.overviews == []

    def test_images_of_the_same_view_share_a_frame_and_another_view_switches(
        self, widget, microscope
    ):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        widget.set_image(_beam_image(microscope, _at(base, dx=300e-6)))
        assert len(widget.views) == 1 and len(widget.overviews) == 2
        fib_pose = _at(base)
        fib_pose.t = base.t + 0.5
        fib = widget.set_image(_beam_image(microscope, fib_pose, beam=BeamType.ION))
        assert len(widget.views) == 2 and widget.view.startswith("FIB @")
        assert widget.canvas.placed_keys == [fib]
        widget.show_view(widget.views[0])
        assert widget.view.startswith("SEM @")
        assert len(widget.canvas.placed_keys) == 2

    def test_a_fluorescence_overview_is_placed_too(self, widget, arctis):
        fm_pose = arctis.get_stage_position()
        rid = widget.set_image(_fm_image(arctis, fm_pose))
        assert rid is not None and widget.view == VIEW_FM
        _settle(widget)
        (cx, cy), _ = widget.overviews[0].extent
        x, y = widget.canvas.metres_to_canvas(cx, cy)
        assert widget.record_at(x, y).id == rid
        target = widget.stage_position_at(x, y)
        assert target is not None
        assert abs(target.x - fm_pose.x) < 1e-6 and abs(target.y - fm_pose.y) < 1e-6


class TestPositions:
    def test_a_position_round_trips_through_the_frame(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        marked = _at(base, dx=40e-6, dy=-25e-6, name="01-aspen")
        widget.set_positions([marked])
        _settle(widget)
        x, y = widget._frame().to_canvas(marked)
        assert widget.position_at(x, y) == "01-aspen"
        # The frame maps a canvas point onto the tilted sample plane, so a stage
        # position off that plane does not come back verbatim; what must hold is
        # that the position it answers draws at the point that was asked about.
        back = widget.stage_position_at(x, y)
        bx, by = widget._frame().to_canvas(back)
        assert abs(bx - x) < 1e-6 and abs(by - y) < 1e-6
        assert abs(back.x - marked.x) < 1e-7
        assert widget.position_at(x + 400, y + 400) is None

    def test_a_click_selects_and_the_menu_offers_a_move(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        marked = _at(base, dx=10e-6, name="02-birch")
        widget.set_positions([marked])
        _settle(widget)
        x, y = widget._frame().to_canvas(marked)
        seen = []
        widget.position_selected.connect(seen.append)
        assert len(widget.position_menu(x, y).actions) == 1
        widget._on_canvas_clicked(x, y)
        assert seen == ["02-birch"] and widget.selected_position == "02-birch"
        assert len(widget.position_menu(x, y).actions) == 2

    def test_a_position_shown_for_context_offers_no_move(self, widget, microscope):
        """A caller drawing what already exists, rather than editing it, says
        so once: the click still selects, and the menu never offers to move it."""
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        marked = _at(base, dx=10e-6, name="03-cedar")
        widget.set_positions([marked], movable=False)
        _settle(widget)
        x, y = widget._frame().to_canvas(marked)
        widget._on_canvas_clicked(x, y)
        assert widget.selected_position == "03-cedar"
        assert [a.label for a in widget.position_menu(x, y).actions] == [
            "Add Position Here"
        ]


class TestDrafts:
    """Positions placed but not committed: what a review holds until it is
    confirmed. The canvas draws them and says which one a click is on; nothing
    here writes anything."""

    def test_a_draft_is_drawn_and_found_by_index(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        first = _at(base, dx=30e-6)
        second = _at(base, dx=-30e-6)
        widget.set_draft_positions([first, second])
        _settle(widget)
        x, y = widget._frame().to_canvas(second)
        assert widget.draft_at(x, y) == 1
        assert widget.draft_at(x + 400, y + 400) is None
        assert len(widget.draft_positions) == 2

    def test_a_draft_is_not_one_of_the_positions(self, widget, microscope):
        """The two layers are separate: a draft is not something the experiment
        holds, so it never answers as a marked position."""
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        draft = _at(base, dx=30e-6)
        widget.set_draft_positions([draft])
        _settle(widget)
        x, y = widget._frame().to_canvas(draft)
        assert widget.position_at(x, y) is None
        assert widget.draft_at(x, y) == 0

    def test_the_menu_over_a_draft_offers_to_remove_that_one(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        widget.set_draft_positions([_at(base, dx=30e-6), _at(base, dx=-30e-6)])
        _settle(widget)
        x, y = widget._frame().to_canvas(widget.draft_positions[1])
        seen = []
        widget.draft_remove_requested.connect(seen.append)
        config = widget.position_menu(x, y)
        assert [a.label for a in config.actions] == ["Remove Position"], (
            "adding another on top of one is not the offer"
        )
        config.actions[0].callback()
        assert seen == [1], "the one under the cursor, by index"

    def test_away_from_a_draft_the_menu_is_the_ordinary_one(self, widget, microscope):
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        widget.set_draft_positions([_at(base, dx=30e-6)])
        _settle(widget)
        x, y = widget._frame().to_canvas(_at(base, dx=-30e-6))
        assert [a.label for a in widget.position_menu(x, y).actions] == [
            "Add Position Here"
        ]

    def test_placing_off_offers_neither_add_nor_remove(self, widget, microscope):
        """A caller showing a record that is already decided turns placing off,
        and the right-click then offers nothing rather than an action that
        would do nothing."""
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        draft = _at(base, dx=30e-6)
        widget.set_draft_positions([draft])
        _settle(widget)
        on_draft = widget._frame().to_canvas(draft)
        elsewhere = widget._frame().to_canvas(_at(base, dx=-30e-6))
        assert [a.label for a in widget.position_menu(*elsewhere).actions] == [
            "Add Position Here"
        ]

        widget.set_placing_enabled(False)

        assert widget.position_menu(*elsewhere) is None
        assert widget.position_menu(*on_draft) is None, "nor taking one back off"

    def test_a_draft_is_not_drawn_where_it_cannot_be_placed(self, widget, microscope):
        """No frame, nothing found -- the same rule the other two layers follow,
        rather than marks left over an image they do not belong to. The list
        survives the image going away, so putting one back draws them again."""
        base = microscope.get_stage_position()
        widget.set_image(_beam_image(microscope, base))
        draft = _at(base, dx=30e-6)
        widget.set_draft_positions([draft])
        _settle(widget)
        x, y = widget._frame().to_canvas(draft)
        assert widget.draft_at(x, y) == 0

        widget.clear()
        _settle(widget)
        assert widget.draft_at(x, y) is None
        assert widget.draft_positions == [draft], "kept, not drawn"

        widget.set_image(_beam_image(microscope, base))
        _settle(widget)
        assert widget.draft_at(x, y) == 0

    def test_an_add_request_names_the_overview_it_was_made_on(self, widget, microscope):
        base = microscope.get_stage_position()
        rid = widget.set_image(_beam_image(microscope, base, item=("g-1", "aspen")))
        _settle(widget)
        seen = []
        widget.position_add_requested.connect(lambda pos, r: seen.append((pos, r)))
        (cx, cy), _ = widget.overviews[0].extent
        widget.request_add_at(*widget.canvas.metres_to_canvas(cx + 20e-6, cy))
        widget.request_add_at(*widget.canvas.metres_to_canvas(cx + 5e-3, cy))
        assert [r for _, r in seen] == [rid, None]
        assert abs(seen[0][0].x - (base.x + 20e-6)) < 1e-7

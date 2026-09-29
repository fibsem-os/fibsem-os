"""A click on a marked position selects it, even when the tile grid is over the top.

The tile grid claimed every left press inside its bounding box and turned it into a
tile toggle. Claiming also suppresses the canvas's `canvas_clicked` signal -- which is
what stops the view panning out from under a grid drag -- and position picking hangs
off that signal. So while a grid was displayed, selecting a marked position inside its
footprint was impossible, and the footprint is exactly where the marked positions are
(FIB-767).

The grid now asks the host first, through `TileGridOverlay.set_reserved`. These tests
drive the real dispatch order -- the canvas connects its own handlers in `__init__`,
before any overlay exists, so a press reaches the canvas and then the overlay -- rather
than asserting on the seam and hoping the two halves meet.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_marked_position_wins_the_click.py
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemImage,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture(scope="module")
def microscope():
    import fibsem.config as fibsem_config

    path = os.path.join(
        os.path.dirname(fibsem_config.__file__),
        "config",
        "sim-arctis-configuration.yaml",
    )
    scope, _ = utils.setup_session(manufacturer="Demo", config_path=path)
    return scope


def _at(base, dx=0.0, dy=0.0, name=None):
    p = FibsemStagePosition(x=base.x + dx, y=base.y + dy, z=base.z, r=base.r, t=base.t)
    p.name = name
    return p


@pytest.fixture
def widget(microscope):
    """A beam overview with one lamella at the stage position and a grid over it."""
    w = FibsemOverviewWidget(microscope)
    w.resize(1000, 800)
    base = microscope.get_stage_position()

    image = FibsemImage.generate_blank_image(resolution=(512, 512), hfw=800e-6)
    image.data = np.zeros((512, 512), dtype=np.uint8)
    state = microscope.get_microscope_state(beam_type=BeamType.ELECTRON)
    state.stage_position = _at(base)
    image.metadata.image_settings = ImageSettings(
        hfw=800e-6, beam_type=BeamType.ELECTRON
    )
    image.metadata.microscope_state = state
    image.metadata.system_info = microscope.system.info
    image.metadata.hardware_geometry = microscope.hardware_geometry()
    w.place_image(image)
    w.set_positions([_at(base, 0.0, 0.0, "Lamella-01")])
    w._refresh_tile_grid()
    _app.processEvents()
    yield w
    w.close()


def _marker(widget, microscope):
    """Canvas coordinates of the one marked position."""
    return widget._frame().to_canvas(_at(microscope.get_stage_position()))


def _bare_grid(widget):
    """A canvas point inside the grid that no marker has claimed."""
    overlay = widget.tile_grid_overlay
    left, top, width, height = overlay._bounds()
    for fx, fy in ((0.9, 0.9), (0.1, 0.1), (0.9, 0.1), (0.1, 0.9)):
        point = (left + fx * width, top + fy * height)
        if overlay._within(*point) and not overlay._is_reserved(*point):
            return point
    raise AssertionError("the grid is entirely reserved; the fixture is wrong")


def _event(widget, x, y):
    transform = widget.canvas._ax.transData
    px, py = transform.transform((x, y))
    return SimpleNamespace(
        button=1,
        inaxes=widget.canvas._ax,
        xdata=x,
        ydata=y,
        x=px,
        y=py,
        dblclick=False,
        key=None,
        guiEvent=None,
    )


def _click(widget, x, y):
    """One press and release, dispatched the way matplotlib would.

    The canvas connects its handlers in `__init__`, before any overlay is added, so it
    sees both events first. Order matters: the canvas's release reads the claim before
    clearing it.
    """
    overlay = widget.tile_grid_overlay
    selected, toggled = [], []
    widget.position_selected.connect(selected.append)
    overlay.tile_toggled.connect(lambda r, c, e: toggled.append((r, c, e)))

    press = _event(widget, x, y)
    widget.canvas._on_press(press)
    overlay._on_press(press)
    claimed = widget.canvas._overlay_consuming_event

    release = _event(widget, x, y)
    widget.canvas._on_release(release)
    overlay._on_release(release)
    if overlay._toggle_timer.isActive():
        overlay._toggle_timer.stop()
        overlay._emit_pending_toggle()
    _app.processEvents()
    return SimpleNamespace(claimed=claimed, selected=selected, toggled=toggled)


class TestAMarkedPositionInsideTheGrid:
    def test_the_grid_really_is_over_the_marker(self, widget, microscope):
        """The precondition. Without it every test below passes for the wrong reason."""
        overlay = widget.tile_grid_overlay
        assert overlay._tiles, "no grid drawn"
        assert overlay._within(*_marker(widget, microscope)), (
            "grid is not over the marker"
        )

    def test_clicking_it_selects_it(self, widget, microscope):
        result = _click(widget, *_marker(widget, microscope))

        assert result.claimed is False, "the grid claimed the press and ate the click"
        assert result.selected == ["Lamella-01"]

    def test_clicking_it_does_not_toggle_the_tile_underneath(self, widget, microscope):
        """Not claiming is only half of it -- a release that still toggled would edit
        the mask behind a click meant for the marker."""
        result = _click(widget, *_marker(widget, microscope))

        assert result.toggled == []


class TestTheRestOfTheGridIsUnchanged:
    def test_a_click_on_bare_grid_still_toggles_its_tile(self, widget):
        result = _click(widget, *_bare_grid(widget))

        assert result.claimed is True
        assert len(result.toggled) == 1
        assert result.selected == []

    def test_a_click_on_bare_grid_selects_nothing(self, widget):
        result = _click(widget, *_bare_grid(widget))

        assert result.selected == []


class TestTheReservationFollowsTheMarkers:
    """It is wired to `_position_at`, not to a second opinion about where markers are.
    So everything that decides whether a click would select -- the field-of-view box,
    and the rule that hidden markers are not targets -- decides this too."""

    def test_hiding_the_markers_gives_the_tile_back(self, widget, microscope):
        """Turned off means gone. A grid standing aside for a marker nothing is drawing
        would be a tile you cannot toggle with nothing on screen to say why."""
        point = _marker(widget, microscope)
        assert _click(widget, *point).toggled == [], "reserved while shown"

        widget.overlay_controls.set_visible("positions", False)
        _app.processEvents()

        result = _click(widget, *point)
        assert result.claimed is True
        assert len(result.toggled) == 1

    def test_the_reservation_is_the_crosshair_not_the_whole_box(
        self, widget, microscope
    ):
        """The box is too big to reserve.

        It is one whole tile on the fluorescence tab by construction -- a tile is one
        camera frame and so is the box -- and one whole tile here at any HFW of 100 um
        or under. Reserving it leaves tiles that cannot be toggled at all, with nothing
        on screen to say why.

        Reserving only the crosshair costs at worst a click that toggles a tile you
        meant to select. That greys out visibly and undoes with one more click, which is
        the better failure: a wrong action that announces itself beats a dead end that
        does not.
        """
        overlay = widget.tile_grid_overlay
        centre = _marker(widget, microscope)
        half_w, _ = widget.position_overlay.half_extent_canvas()
        inside_box = (centre[0] + half_w * 0.8, centre[1])

        transform = widget.canvas._ax.transData
        a, b = transform.transform(centre), transform.transform(inside_box)
        distance = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        assert distance > 12, "probe is within the crosshair radius; it proves nothing"
        assert overlay._within(*inside_box), "probe left the grid; it proves nothing"

        assert not overlay._is_reserved(*inside_box)

    def test_a_click_in_the_box_but_off_the_crosshair_toggles_the_tile(
        self, widget, microscope
    ):
        """The other half of it, through the real dispatch rather than the predicate."""
        centre = _marker(widget, microscope)
        half_w, _ = widget.position_overlay.half_extent_canvas()
        inside_box = (centre[0] + half_w * 0.8, centre[1])

        result = _click(widget, *inside_box)

        assert result.claimed is True
        assert len(result.toggled) == 1

    def test_selection_still_uses_the_box(self, widget, microscope):
        """Only the reservation narrowed. What a click *selects* is unchanged, so the
        box keeps its aim benefit everywhere the grid is not contending for the point.
        """
        centre = _marker(widget, microscope)
        half_w, _ = widget.position_overlay.half_extent_canvas()
        inside_box = (centre[0] + half_w * 0.8, centre[1])

        assert widget._position_at(*inside_box) == "Lamella-01"
        assert widget._position_at(*inside_box, crosshair_only=True) is None


# ── the fluorescence tab ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fm_widget(qapp):
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    scope = build_microscope()
    scope.fm.objective.insert()
    w = FMOverviewWidget(scope)
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    w._refresh_tile_grid()
    qapp.processEvents()
    return w


def test_the_fluorescence_tab_stands_aside_too(qapp, fm_widget):
    """Both tabs draw the same overlay over the same kind of marker, so both wire it.
    A fix on one tab only would be the sort of asymmetry `_position_at` already had."""
    position = FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(-180.0))
    position.name = "Lamella-01"
    fm_widget.set_positions([position])
    qapp.processEvents()

    overlay = fm_widget.tile_grid_overlay
    point = fm_widget._frame().to_canvas(position)
    assert overlay._tiles, "no grid drawn"
    assert overlay._within(*point), "the grid is not over the marker"

    assert overlay._is_reserved(*point)
    assert fm_widget._position_at(*point) == "Lamella-01"

    fm_widget.set_positions([])
    qapp.processEvents()
    assert not overlay._is_reserved(*point), "the reservation outlived the marker"

"""Moving the mouse over the overview canvas must not talk to the microscope.

Found on hardware: dragging the tile grid stuttered. The cause was not the drag — it
was `_frame()`, which every drawing path goes through, resolving
`FMStageProjection.from_microscope` on each call. That reads `camera.resolution`,
`camera.pixel_size` and `fm_image_geometry()`, and on a TFS system the first two go
through `camera.binning`, which is a device read inside an `active_channel()` scope.

Two handlers call `_frame()` on **every mouse-move event** — the cursor readout and the
grid drag — so the widget was issuing AutoScript round trips continuously while the
pointer was over the canvas, on the connection the beams share. The tile-grid overlay's
own docstring asks the host to keep that response cheap.

The split pinned here is the one `ObjectiveLens.position_changed` established: displays
use the kept value, anything that commands the instrument reads the device. Exactly one
caller commands anything — `_stage_position_at`, which turns a double-click into a stage
move — and it must keep re-reading, because a stale binning there sends the stage
somewhere other than where the click was.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

_app = QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def widget():
    from fibsem.ui.fm.overview_app import build_microscope
    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    microscope = build_microscope()
    microscope.fm.objective.insert()
    w = FMOverviewWidget(microscope)
    w.resize(900, 600)
    yield w
    w.deleteLater()


@pytest.fixture
def reads(widget, monkeypatch):
    """Every read of the camera's geometry, by whatever route.

    Two earlier versions of this fixture were wrong, and both failures are worth keeping
    in mind because they are easy to repeat:

    * counting `FMStageProjection.from_microscope` went green with the bug reinstated,
      because `_refresh_tile_grid` reads `camera.resolution` **directly** rather than
      through the projection;
    * counting `active_channel()` entries counted nothing at all, because only the *TFS*
      camera's `binning` takes the channel. The simulated one returns a plain attribute,
      so the very cost being tested does not exist here — the same blind spot that sent
      the whole FIB-517 class to hardware to be found (FIB-518).

    So this counts the geometry access itself, which is what the UI does and is identical
    on both. On hardware each of these is an `active_channel()` scope and a handful of
    AutoScript round trips; here it is free, and only the *count* carries over.
    """
    camera = type(widget.fm.camera)
    seen = []

    for name in ("resolution", "pixel_size"):
        original = getattr(camera, name)

        def counting(self, _original=original, _name=name):
            seen.append(_name)
            return _original.fget(self)

        monkeypatch.setattr(camera, name, property(counting))

    widget.invalidate_projection()  # so the count starts from a known state
    return seen


class TestTheProjectionIsKept:
    def test_repeated_frames_resolve_it_once(self, widget, reads):
        """The cost must not scale with the number of frames drawn.

        Asserted against the first frame's own cost rather than against a fixed number:
        one resolve is several scopes -- `resolution` and `pixel_size` each take one --
        and how many is an implementation detail. That it does not *grow* is not.
        """
        widget._frame()
        first = len(reads)
        assert first > 0, "the first frame did not read the device at all"

        for _ in range(19):
            widget._frame()

        assert len(reads) == first, (
            f"{len(reads)} channel scopes for 20 frames against {first} for one — the "
            f"cost is scaling with frames, and each scope is AutoScript round trips on "
            f"the connection the beams share"
        )

    def test_moving_the_mouse_does_not_read_the_microscope(self, widget, reads):
        """The worst of the two: this fires whenever the pointer is over the canvas."""
        widget._frame()  # the one legitimate read
        reads.clear()

        for i in range(30):
            widget._on_cursor_moved(float(i), float(i))

        assert reads == [], (
            f"the cursor readout resolved the projection {len(reads)} times over 30 "
            f"mouse-move events — it renders a text label, and it was reading the camera "
            f"to do it"
        )

    def test_dragging_the_grid_does_not_read_the_microscope(self, widget, reads):
        widget._frame()
        reads.clear()

        for i in range(30):
            widget._on_grid_move(float(i) * 1e-6, 0.0)

        assert reads == [], (
            f"the grid drag resolved the projection {len(reads)} times over 30 motion "
            f"events — this is what stuttered on hardware"
        )

    def test_redrawing_the_planned_grid_does_not_read_the_microscope(
        self, widget, reads
    ):
        """`_refresh_tile_grid` had its own camera reads, separate from `_frame`.

        Caching the frame's projection alone left these, so a drag still took the shared
        channel twice per motion event -- visible as the active view flicking in the
        microscope's own UI, which is how it was reported.
        """
        widget._frame()
        reads.clear()

        for _ in range(30):
            widget._refresh_tile_grid()

        assert reads == [], (
            f"the grid redraw took the channel {len(reads)} times over 30 calls"
        )

    def test_showing_a_preview_tile_does_not_read_the_microscope(self, widget, reads):
        """Runs once a tile *during* a run, so it competes with the acquisition."""
        import numpy as np

        widget._frame()
        reads.clear()

        payload = {
            "image": np.zeros((1, 64, 64), np.uint8),
            "preview_stride": 4,
        }
        for _ in range(10):
            widget._show_preview(payload)

        assert reads == [], (
            f"the preview took the channel {len(reads)} times over 10 tiles — "
            f"that is the display taking the channel from the run that is using it"
        )


class TestWhatStillReadsTheDevice:
    def test_resolving_a_click_to_a_stage_position_re_reads(self, widget, reads):
        """The one caller that commands the instrument.

        A kept binning would be a stale pixel size, and the stage would go somewhere
        other than where the click was — worse than the lag this all fixes.
        """
        widget._frame()
        reads.clear()

        widget._stage_position_at(10.0, 10.0)

        assert reads, (
            "clicking to move used the kept projection — that value drives the stage, "
            "so it has to come from the device"
        )

    def test_invalidating_forces_the_next_frame_to_re_read(self, widget, reads):
        widget._frame()
        reads.clear()

        widget.invalidate_projection()
        widget._frame()

        assert reads, "invalidating did not force the next frame to re-read"

    def test_starting_a_run_invalidates(self):
        """A run plans a grid and drives the stage, so it starts from a fresh read.

        Structural: `acquire` opens a modal confirmation dialog, which blocks a headless
        run rather than failing it.
        """
        import ast
        import inspect
        import textwrap

        from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

        source = textwrap.dedent(inspect.getsource(FMOverviewWidget.acquire))
        called = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "invalidate_projection" in called, (
            "`acquire` does not invalidate the projection, so a run could plan its grid "
            "and drive the stage from a binning read taken before it was set up"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

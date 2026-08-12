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
    """Every resolve of the projection, i.e. every burst of device reads."""
    import fibsem.ui.fm.widgets.fm_overview_widget as module

    seen = []
    original = module.FMStageProjection.from_microscope

    def counting(microscope):
        seen.append(1)
        return original(microscope)

    monkeypatch.setattr(module.FMStageProjection, "from_microscope", staticmethod(counting))
    widget.invalidate_projection()  # so the count starts from a known state
    return seen


class TestTheProjectionIsKept:
    def test_repeated_frames_resolve_it_once(self, widget, reads):
        for _ in range(20):
            widget._frame()

        assert len(reads) == 1, (
            f"{len(reads)} projection reads for 20 frames — each one is several "
            f"AutoScript round trips on the shared connection"
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


class TestWhatStillReadsTheDevice:
    def test_resolving_a_click_to_a_stage_position_re_reads(self, widget, reads):
        """The one caller that commands the instrument.

        A kept binning would be a stale pixel size, and the stage would go somewhere
        other than where the click was — worse than the lag this all fixes.
        """
        widget._frame()
        reads.clear()

        widget._stage_position_at(10.0, 10.0)

        assert len(reads) == 1, (
            "clicking to move used the kept projection — that value drives the stage, "
            "so it has to come from the device"
        )

    def test_invalidating_forces_the_next_frame_to_re_read(self, widget, reads):
        widget._frame()
        reads.clear()

        widget.invalidate_projection()
        widget._frame()

        assert len(reads) == 1

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

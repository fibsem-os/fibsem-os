"""The FM overview's frame is anchored at the sample's height on an offset FM.

A METEOR's FM entry gives the origin's x and y only (odemis's CALIB dx, dy), and the
overview read the missing z as 0: its sample plane sat 32 mm below the sample. A
double-click near an overview's corner drove the stage 15 mm in y and 21 mm in z under
the inserted objective (odemis simulator, deployment B, 2026-10-26). These hold the
frame to the stage's own position, and the last line of defence to its refusal.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from copy import deepcopy  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PyQt5")

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.imaging.overview_session import FLUORESCENCE, OverviewSession  # noqa: E402
from fibsem.structures.settings import FibsemStagePosition  # noqa: E402
from fibsem.ui.fm.widgets import fm_overview_widget as module  # noqa: E402

# A METEOR's sample height: odemis's SAMPLE_CENTERS put the grids at z = 32 mm.
SAMPLE_Z = 32.0e-3


@pytest.fixture
def widget(qapp, monkeypatch):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.SHIPPED_CONFIG_PATH, "sim-iflm-configuration.yaml")
    )
    sem = microscope.get_orientation("SEM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=SAMPLE_Z, r=sem.r, t=sem.t)
    )
    microscope.stage_movement.move_to_device("FM")
    assert microscope.get_stage_position().z == pytest.approx(SAMPLE_Z)

    toasts = []
    monkeypatch.setattr(
        module.notification_service,
        "show_toast",
        lambda message, *a, **k: toasts.append(message),
    )
    built = module.FMOverviewWidget(OverviewSession(microscope, FLUORESCENCE))
    built.toasts = toasts
    yield built
    built.close()


def test_the_stage_maps_back_onto_itself(widget):
    """Where the stage is, drawn and read back, is where the stage is: a frame
    anchored at the wrong height reads it back somewhere else along the tilted plane."""
    here = widget.microscope.get_stage_position()
    frame = widget._frame(fresh=True)

    back = frame.to_stage(*frame.to_canvas(here))

    assert back.y == pytest.approx(here.y, abs=1e-6)
    assert back.z == pytest.approx(here.z, abs=1e-6)


def test_a_click_beside_the_stage_stays_at_its_height(widget):
    here = widget.microscope.get_stage_position()
    frame = widget._frame(fresh=True)
    cx, cy = frame.to_canvas(here)
    near = deepcopy(here)
    near.x += 100e-6
    nx, ny = frame.to_canvas(near)

    target = widget._stage_position_at(nx + (nx - cx), cy)

    assert target is not None, widget.toasts
    assert abs(target.z - here.z) < module.MAX_CLICK_DZ
    assert abs(target.y - here.y) < 1e-3


def test_a_target_far_off_in_height_is_refused(widget):
    far = deepcopy(widget.microscope.get_stage_position())
    far.z -= 21e-3

    assert not widget._within_click_height(far)
    assert widget.toasts and "Refused" in widget.toasts[-1]

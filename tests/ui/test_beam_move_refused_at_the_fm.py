"""On an offset FM, a move picked from a beam image is refused while the stage is at
the FM.

At the FM the SEM and FIB look at a point about 50 mm from the sample, so a move
worked out from their pixels (a double-click on a beam canvas, or on the FIBSEM
overview) lands nowhere in particular, under an inserted objective. Found on the odemis
simulator, deployment B, 2026-10-26. A compustage, where the beams and the FM are one
place, is unchanged.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

import fibsem.config as cfg  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.ui.widgets.stage_control_widget import (  # noqa: E402
    AT_THE_FM,
    beam_move_refusal,
)


def _session(config: str):
    microscope, _ = utils.setup_session(
        config_path=os.path.join(cfg.SHIPPED_CONFIG_PATH, config)
    )
    return microscope


def test_an_offset_fm_refuses_beam_moves_at_the_fm_only():
    microscope = _session("sim-iflm-configuration.yaml")
    assert beam_move_refusal(microscope) is None, "refused at the beams"

    microscope.stage_movement.move_to_device("FM")
    assert beam_move_refusal(microscope) == AT_THE_FM

    microscope.stage_movement.move_to_device("FIBSEM")
    assert beam_move_refusal(microscope) is None, "still refused back at the beams"


def test_a_compustage_never_refuses():
    microscope = _session("sim-arctis-configuration.yaml")
    microscope.move_stage_absolute(microscope.get_orientation("FM"))
    assert beam_move_refusal(microscope) is None


def test_the_fibsem_overview_asks_the_same_question(qapp):
    from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

    microscope = _session("sim-iflm-configuration.yaml")
    microscope.stage_movement.move_to_device("FM")
    assert (
        FibsemOverviewWidget._orientation_refusal.__get__(
            type("Host", (), {"microscope": microscope})()
        )()
        == AT_THE_FM
    )

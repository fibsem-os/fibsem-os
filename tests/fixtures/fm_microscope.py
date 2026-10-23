"""The microscope the FM widget tests run against.

The simulated compustage instrument (``sim-arctis-configuration.yaml``: a compustage
with no pre-tilt, and a fluorescence microscope), posed for fluorescence. That is the
geometry the FM tile projection is built around, so it is the one the tests exercise.

The tests used to import these from ``fibsem.ui.fm.overview_app`` and
``fibsem.ui.fm.sparse_selection_app``, two developer apps in the package. What a test
runs against belongs with the tests, so they no longer depend on those apps.
"""

import logging
import os
from typing import Dict, List

import numpy as np

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings

SIM_COMPUSTAGE_CONFIG = os.path.join(
    cfg.SHIPPED_CONFIG_PATH, "sim-arctis-configuration.yaml"
)
"""The simulated compustage instrument: no pre-tilt, a fluorescence microscope fitted."""

OVERVIEW_HFW = 900e-6
"""Big enough to hold several grid squares, so a sparse selection has a choice."""


def build_microscope() -> FibsemMicroscope:
    """The simulated compustage instrument, at x = y = z = 0 in its FM orientation.

    The pose comes from the microscope's own ``FM`` orientation rather than a number
    written here, so the tests follow the configuration if it changes.
    """
    microscope, _ = utils.setup_session(
        ip_address="localhost", config_path=SIM_COMPUSTAGE_CONFIG
    )
    pose = microscope.get_orientation("FM")
    microscope.move_stage_absolute(
        FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
    )
    assert microscope.fm is not None, f"{SIM_COMPUSTAGE_CONFIG} fits no FM"
    return microscope


def acquire_beam_overviews(
    microscope: FibsemMicroscope, hfw: float = OVERVIEW_HFW
) -> Dict[object, List[object]]:
    """One overview per beam view, acquired at that view's own orientation, so each
    image carries the pose a sparse selection resolves against. Leaves the stage at
    the last one (the milling orientation)."""
    from fibsem.ui.widgets.overview_widget import OverviewView

    views: Dict[object, List[object]] = {}
    for orientation, beam_type in (
        ("SEM", BeamType.ELECTRON),
        ("MILLING", BeamType.ION),
    ):
        pose = microscope.get_orientation(orientation)
        microscope.move_stage_absolute(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=pose.r, t=pose.t)
        )
        image = microscope.acquire_image(
            ImageSettings(
                hfw=hfw, resolution=[1024, 1024], beam_type=beam_type, save=False
            )
        )
        views[OverviewView(beam_type, orientation)] = [image]
        logging.debug(
            f"acquired {beam_type.name} @ {orientation}, "
            f"t = {np.degrees(pose.t):.1f} deg"
        )
    return views

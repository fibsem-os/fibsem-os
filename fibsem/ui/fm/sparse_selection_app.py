"""Standalone app for the sparse FM selection panes.

Opens the selection dialog against a real or simulated microscope, acquires the beam
overviews it needs, and prints what comes back -- so the whole path can be driven before
anything in AutoLamella constructs it.

    python -m fibsem.ui.fm.sparse_selection_app                    # Demo, two beam views
    python -m fibsem.ui.fm.sparse_selection_app --manufacturer Thermo --ip 10.0.0.1

Right-click the left pane to add a region, then drag it or its corners. The right pane
shows the FM tiles that selection would acquire, recomputed on every change. Accepting
prints the grid, the mask and the centre the acquisition would be given.
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional

import numpy as np

from fibsem.microscope import FibsemMicroscope
from fibsem.structures import BeamType, FibsemStagePosition, ImageSettings
from fibsem.ui.fm.overview_app import build_microscope

OVERLAP = 0.1
# Big enough to hold several grid squares, so there is something worth being sparse about.
OVERVIEW_HFW = 900e-6


def acquire_beam_overviews(
    microscope: FibsemMicroscope, hfw: float = OVERVIEW_HFW
) -> Dict[object, List[object]]:
    """One overview per beam view, so the chip strip has something to switch between.

    Acquired here rather than loaded, because the point is to exercise the panes against
    images that carry their own pose -- which is what the selection resolves against.
    """
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
        logging.info(
            f"acquired {beam_type.name} @ {orientation}, "
            f"t = {np.degrees(pose.t):.1f} deg"
        )
    return views


def run_dialog(microscope: FibsemMicroscope, hfw: float = OVERVIEW_HFW):
    """Acquire the beam overviews, run the dialog, and return what it produced."""
    from fibsem.fm.structures import OverviewParameters
    from fibsem.ui.fm.widgets.fm_sparse_selection_dialog import FMSparseSelectionDialog

    views = acquire_beam_overviews(microscope, hfw)
    # Back to the fluorescence pose, as the FM tab would leave it. The preview must be
    # right from either, which is what pinning its frame to the FM orientation is for --
    # comment this out and nothing about the right-hand pane should change.
    microscope.move_to_microscope("FM")

    channels = getattr(microscope.fm, "channel_settings", None)
    return FMSparseSelectionDialog.choose(
        microscope,
        views,
        OverviewParameters(overlap=OVERLAP),
        channel_settings=list(channels) if channels else None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufacturer", default="Demo")
    parser.add_argument("--ip", default="localhost")
    parser.add_argument(
        "--hfw", type=float, default=OVERVIEW_HFW,
        help="beam overview field width, in metres",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv)
    microscope = build_microscope(
        manufacturer=args.manufacturer, ip_address=args.ip
    )
    selection = run_dialog(microscope, args.hfw)
    if selection is None:
        logging.info("cancelled -- nothing selected")
        return 0

    parameters = selection.parameters
    enabled = sum(sum(1 for on in row if on) for row in parameters.tile_mask)
    logging.info(
        f"grid {parameters.rows} x {parameters.cols}, "
        f"{enabled} of {parameters.rows * parameters.cols} tiles, "
        f"overlap {parameters.overlap:.0%}"
    )
    logging.info(f"centred on {selection.centre_position.pretty}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

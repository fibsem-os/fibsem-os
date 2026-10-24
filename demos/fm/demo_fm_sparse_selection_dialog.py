"""FMSparseSelectionDialog on the simulated microscope, with beam overviews to pick on.

Run with:
    python demos/fm/demo_fm_sparse_selection_dialog.py
    python demos/fm/demo_fm_sparse_selection_dialog.py --hfw 600e-6

Starts the Demo microscope from the compustage simulator configuration, acquires one
SEM and one FIB overview, each at its own orientation, then sends the stage to the FM
and opens the dialog. Right-click the left pane to add a region, then drag it or its
corners; the right pane shows the FM tiles that selection would acquire. Accepting
prints the grid, the mask and the centre the acquisition would be given.
"""

import argparse
import os
import sys

from PyQt5.QtWidgets import QApplication

import fibsem.config as cfg
from fibsem import utils
from fibsem.fm.structures import OverviewParameters
from fibsem.structures.settings import BeamType, FibsemStagePosition, ImageSettings
from fibsem.ui.fm.widgets.fm_sparse_selection_dialog import FMSparseSelectionDialog
from fibsem.ui.widgets.overview_widget import OverviewView

SIM_COMPUSTAGE_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
OVERLAP = 0.1
# Big enough to hold several grid squares, so there is something worth being sparse about.
OVERVIEW_HFW = 900e-6


def acquire_beam_overviews(microscope, hfw: float) -> dict:
    """One overview per beam, each acquired at that beam's own orientation, so the
    images carry the pose the selection resolves against."""
    views = {}
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
    return views


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hfw",
        type=float,
        default=OVERVIEW_HFW,
        help="beam overview field width, in metres",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)  # noqa: F841 - the dialog needs one to exist

    microscope, _ = utils.setup_session(
        manufacturer="Demo", config_path=SIM_COMPUSTAGE_CONFIG
    )
    views = acquire_beam_overviews(microscope, args.hfw)
    # Back to the FM, as the FM tab would leave it. The preview must be right from
    # either pose, which is what pinning its frame to the FM orientation is for.
    microscope.move_to_device("FM")

    channels = getattr(microscope.fm, "channel_settings", None)
    selection = FMSparseSelectionDialog.choose(
        microscope,
        views,
        OverviewParameters(overlap=OVERLAP),
        channel_settings=list(channels) if channels else None,
    )
    if selection is None:
        print("Cancelled: nothing selected")
        return 0

    parameters = selection.parameters
    enabled = sum(sum(1 for on in row if on) for row in parameters.tile_mask)
    print(
        f"grid {parameters.rows} x {parameters.cols}, "
        f"{enabled} of {parameters.rows * parameters.cols} tiles, "
        f"overlap {parameters.overlap:.0%}"
    )
    print(f"centred on {selection.centre_position.pretty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

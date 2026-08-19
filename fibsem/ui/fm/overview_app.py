"""Standalone FM overview acquisition app.

Opens :class:`FMOverviewWidget` against a real or simulated microscope, so the
overview UI can be exercised without launching AutoLamella.

    python -m fibsem.ui.fm.overview_app                 # Demo, compustage at t = -180
    python -m fibsem.ui.fm.overview_app --manufacturer Thermo --ip 10.0.0.1
    python -m fibsem.ui.fm.overview_app --output ~/fm-overviews    # and save them

The default puts the simulator in the Arctis fluorescence pose -- compustage, no
pre-tilt, stage tilted to -180 degrees -- because that is the geometry the tile
projection is built around and the one worth exercising.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition


def build_microscope(
    manufacturer: str = "Demo",
    ip_address: str = "localhost",
    compustage: bool = True,
    tilt_deg: float = -180.0,
) -> FibsemMicroscope:
    """Connect, attach a fluorescence detector, and pose the stage for FM."""
    microscope, _ = utils.setup_session(
        manufacturer=manufacturer, ip_address=ip_address
    )

    if manufacturer == "Demo":
        # Pose the simulator rather than assuming the ambient configuration: the tile
        # projection depends on the compustage flag and the pre-tilt, so a run started
        # from the wrong pose would exercise different geometry than intended.
        microscope.stage_is_compustage = compustage
        microscope.system.stage.shuttle_pre_tilt = 0
        microscope._update_orientations()
        # Re-read the travel limits, which `FibsemStage.__init__` cached before the flag
        # above was set. They are not a detail: a compustage travels +/-999.9 um in x and
        # only +/-377.8 um in y, against a grid boundary of 1000 um radius -- so in y the
        # stage runs out of travel well inside the grid, and it is the limits rather than
        # the boundary that say how far an overview can reach. Left stale, the simulator
        # reports +/-100 mm and anything drawing them shows nothing at grid scale.
        microscope._stage.limits = microscope._get_axis_limits()
        microscope.move_stage_absolute(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg))
        )

    if microscope.fm is None:
        from fibsem.fm.microscope import FluorescenceMicroscope

        microscope.fm = FluorescenceMicroscope(parent=microscope)

    logging.info(
        f"FM overview app: {manufacturer}, compustage={microscope.stage_is_compustage}, "
        f"stage={microscope.get_stage_position().pretty}"
    )
    return microscope


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manufacturer", default="Demo",
                        help="Demo, Thermo, Tescan (default: Demo)")
    parser.add_argument("--ip", default="localhost", dest="ip_address")
    parser.add_argument("--tilt", type=float, default=-180.0,
                        help="Stage tilt in degrees for the Demo pose (default: -180)")
    parser.add_argument("--no-compustage", action="store_false", dest="compustage",
                        help="Pose the Demo microscope as an offset FM mount instead")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help="Write acquired overviews under DIR. Each run gets its own "
                             "subdirectory of tiles, with the stitched mosaic beside it. "
                             "Omitted, overviews are shown but not saved.")
    args = parser.parse_args(argv)

    from PyQt5.QtWidgets import QApplication, QMainWindow

    from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget

    app = QApplication(sys.argv)
    try:
        from fibsem.ui import stylesheets

        app.setStyleSheet(stylesheets.NAPARI_STYLE)
    except Exception as e:  # pragma: no cover - styling is cosmetic
        logging.debug(f"Could not apply the napari stylesheet: {e}")

    microscope = build_microscope(
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        compustage=args.compustage,
        tilt_deg=args.tilt,
    )

    window = QMainWindow()
    window.setWindowTitle(
        f"FM Overview — {args.manufacturer}"
        f"{' · compustage' if microscope.stage_is_compustage else ''}"
        f" · t = {np.rad2deg(microscope.get_stage_position().t):.0f}°"
    )
    widget = FMOverviewWidget(microscope)
    # Opt-in rather than defaulted to the working directory: this app is mostly run
    # against a simulator to exercise the UI, and those runs should not leave tiles
    # behind wherever it happened to be launched from.
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        widget.set_save_directory(args.output)
        logging.info(f"Overviews will be saved under {args.output}")
    window.setCentralWidget(widget)
    window.resize(1500, 900)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

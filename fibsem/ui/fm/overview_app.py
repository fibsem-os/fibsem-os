"""Standalone FM overview acquisition app.

Opens :class:`FMOverviewWidget` against a real or simulated microscope, so the
overview UI can be exercised without launching AutoLamella.

    python -m fibsem.ui.fm.overview_app                 # simulated Arctis, at t = -180
    python -m fibsem.ui.fm.overview_app --config ~/arctis.yaml --ip 10.0.0.1
    python -m fibsem.ui.fm.overview_app --output ~/fm-overviews    # and save them

The default is the simulated Arctis (``sim-arctis-configuration.yaml``: compustage, no
pre-tilt) tilted to -180 degrees, the fluorescence pose, because that is the geometry
the tile projection is built around and the one worth exercising. The configuration
says what the microscope is; the app asks nothing about its manufacturer. It moves the
stage only when given a tilt, which it is by default only on that simulator.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import numpy as np

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import FibsemStagePosition

SIM_ARCTIS_CONFIG = os.path.join(cfg.CONFIG_PATH, "sim-arctis-configuration.yaml")
"""The simulated Arctis: a compustage with no pre-tilt, the FM overview's own geometry."""

SIM_ARCTIS_TILT_DEG = -180.0
"""The fluorescence pose of the simulated Arctis, where it is posed by default."""


def default_tilt(
    config_path: Optional[str], manufacturer: Optional[str]
) -> Optional[float]:
    """The tilt an app poses the stage at when none is given: the fluorescence pose on
    the simulated Arctis, and none on anything else, so an instrument is never moved
    unasked."""
    if config_path == SIM_ARCTIS_CONFIG and manufacturer is None:
        return SIM_ARCTIS_TILT_DEG
    return None


def build_microscope(
    config_path: Optional[str] = SIM_ARCTIS_CONFIG,
    manufacturer: Optional[str] = None,
    ip_address: str = "localhost",
    tilt_deg: Optional[float] = None,
) -> FibsemMicroscope:
    """Connect, attach a fluorescence detector, and pose the stage at a tilt.

    The configuration decides the stage type and the pre-tilt, which the tile projection
    depends on, so the run exercises the geometry it names. A tilt moves the stage to
    x = y = z = r = 0 at that tilt first. Without one, the tilt is `default_tilt`: the
    simulated Arctis is posed for fluorescence, and anything else is left where it is.
    """
    if tilt_deg is None:
        tilt_deg = default_tilt(config_path, manufacturer)
    microscope, _ = utils.setup_session(
        manufacturer=manufacturer, ip_address=ip_address, config_path=config_path
    )

    if tilt_deg is not None:
        microscope.move_stage_absolute(
            FibsemStagePosition(x=0.0, y=0.0, z=0.0, r=0.0, t=np.deg2rad(tilt_deg))
        )

    if microscope.fm is None:
        from fibsem.drivers.demo.microscope import DemoFluorescenceMicroscope

        microscope.fm = DemoFluorescenceMicroscope(parent=microscope)

    logging.info(
        f"FM overview app: {microscope.system.info.manufacturer}, "
        f"compustage={microscope._fm_is_a_pose()}, "
        f"stage={microscope.get_stage_position().pretty}"
    )
    return microscope


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=SIM_ARCTIS_CONFIG,
        dest="config_path",
        help="Microscope configuration file (default: the simulated Arctis). "
        "fibsem/config/sim-iflm-configuration.yaml simulates an offset FM mount.",
    )
    parser.add_argument(
        "--manufacturer",
        default=None,
        help="Override the configuration's manufacturer (Demo, Thermo, Tescan)",
    )
    parser.add_argument("--ip", default="localhost", dest="ip_address")
    parser.add_argument(
        "--tilt",
        type=float,
        default=None,
        help="Move the stage to this tilt in degrees, at x = y = z = r = 0, before "
        f"opening (default: {SIM_ARCTIS_TILT_DEG:.0f} on the simulated Arctis, "
        "no move otherwise)",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="DIR",
        help="Write acquired overviews under DIR. Each run gets its own "
        "subdirectory of tiles, with the stitched mosaic beside it. "
        "Omitted, overviews are shown but not saved.",
    )
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
        config_path=args.config_path,
        manufacturer=args.manufacturer,
        ip_address=args.ip_address,
        tilt_deg=args.tilt,
    )

    window = QMainWindow()
    window.setWindowTitle(
        f"FM Overview — {microscope.system.info.manufacturer}"
        f"{' · compustage' if microscope._fm_is_a_pose() else ''}"
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

"""Manual launcher for the correlation window's lamella Setup tab, on mock data.

The Setup section is installed only when the autolamella protocol editor opens
the correlation window, so seeing it normally means an experiment, a lamella and
its images. This builds the same thing from mock data — no microscope, no
experiment, no files needed.

Usage
-----
    # a lamella with spot burns, two previous runs and both images loaded
    python -m fibsem.ui.correlation.widgets.test_correlation_setup_section

    # opened with no FIB image, to see the spot-burn rule
    python -m fibsem.ui.correlation.widgets.test_correlation_setup_section --no-fib

What to try
-----------
* Switch Starting Coordinates between Previous / Spot-burn / None and watch the
  points change on the real canvases — the canvas *is* the preview.
* Drag a seeded point, then switch the radio (or the run dropdown) and Cancel:
  the control goes back to the source actually on the canvas (FIB-319). Add an FM
  surface point first and the same confirm covers it.
* ``--no-fib``: the burns are normalised to the FIB image they were placed on, so
  with no image the radio is disabled and says so. The FIB entries are written to
  real files, so picking one loads it for real and the radio comes alive;
  switching between them raises the discard confirm (FIB-317).
"""
import os
import sys
import tempfile
from datetime import datetime

import numpy as np
from PyQt5.QtWidgets import QApplication

from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.ui.correlation.widgets.correlation_tab_widget import CorrelationTabWidget
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemImage, Point
from fibsem.ui.stylesheets import NAPARI_STYLE

_BURNS = [Point(0.33, 0.51), Point(0.5, 0.51), Point(0.66, 0.51)]


def _fib_image(burn_y: int = 205, seed: int = 1) -> FibsemImage:
    """A lamella strip with three spot burns melted into it."""
    img = FibsemImage.generate_blank_image(resolution=(600, 400), hfw=100e-6)
    yy, xx = np.mgrid[0:400, 0:600]
    g = np.full((400, 600), 60, np.float32)
    g[burn_y - 25 : burn_y + 25, :] = 110
    for cx in (200, 300, 400):
        g += 150 * np.exp(-((xx - cx) ** 2 + (yy - burn_y) ** 2) / 72)
    img.data[:] = np.clip(
        g + np.random.default_rng(seed).normal(0, 6, (400, 600)), 0, 255
    ).astype(img.data.dtype)
    return img


def _write_fib_images() -> list:
    """Two FIB images on disk, so the picker has something real to load.

    The second sits lower in frame, so swapping between them is obvious — and
    raises the coordinate-discard confirm.
    """
    directory = tempfile.mkdtemp(prefix="correlation-mock-")
    paths = []
    for name, img in (
        ("ref_MillRough_final_ib.tif", _fib_image()),
        ("ref_Trench_final_ib.tif", _fib_image(burn_y=260, seed=2)),
    ):
        path = os.path.join(directory, name)
        img.save(path)
        paths.append(path)
    return paths


def _fm_image(nz: int = 11) -> FluorescenceImage:
    """A single-channel stack with one blob, anisotropic in z."""
    data = np.zeros((1, nz, 200, 200), np.uint16)
    yy, xx = np.mgrid[0:200, 0:200]
    for z in range(nz):
        data[0, z] = (2000 * np.exp(-((xx - 100) ** 2 + (yy - 95) ** 2) / 648)).astype(
            np.uint16
        )
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        power=0.3,
        exposure_time=0.05,
        gain=1.5,
        offset=50.0,
    )
    return FluorescenceImage(
        data=data,
        metadata=FluorescenceImageMetadata(
            acquisition_date=datetime(2026, 7, 24).isoformat(),
            pixel_size_x=40e-9,
            pixel_size_y=40e-9,
            pixel_size_z=200e-9,
            resolution=(200, 200),
            channels=[channel],
        ),
    )


def _history() -> LamellaCorrelation:
    """Two previous runs: an empty one, and one worth seeding from."""
    previous = CorrelationInputData(
        fib_coordinates=[
            Coordinate(point=PointXYZ(x=x, y=205, z=0), point_type=PointType.FIB)
            for x in (200, 300, 400)
        ],
        fm_coordinates=[
            Coordinate(point=PointXYZ(x=x, y=y, z=5), point_type=PointType.FM)
            for x, y in ((60, 60), (140, 70), (100, 150))
        ],
        poi_coordinates=[
            Coordinate(point=PointXYZ(x=100, y=95, z=5), point_type=PointType.POI)
        ],
    )
    return LamellaCorrelation(
        runs=[
            CorrelationRun(
                path="/tmp/a",
                name="2026-07-23_09-10-00",
                state=CorrelationState(input_data=CorrelationInputData()),
            ),
            CorrelationRun(
                path="/tmp/b",
                name="2026-07-24_14-32-00",
                state=CorrelationState(input_data=previous),
            ),
        ]
    )


def main() -> None:
    no_fib = "--no-fib" in sys.argv

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(NAPARI_STYLE)

    config = CorrelationConfig()
    config.fit.fm_poi_channel = "GFP"
    fib_paths = _write_fib_images()

    widget = CorrelationTabWidget()
    widget.setWindowTitle(
        "Correlation — Lamella 03 (mock)" + (" — no FIB image" if no_fib else "")
    )
    if not no_fib:
        widget.set_fib_image(FibsemImage.load(fib_paths[0]))
    widget.set_fm_image(_fm_image())

    section = widget.add_lamella_setup(
        spot_burns=_BURNS,
        history=_history(),
        config=config,
        fib_options=fib_paths,
        fib_current="" if no_fib else fib_paths[0],
        fm_options=["/lam/Lamella03_FM_G1.ome.tiff"],
        fm_current="/lam/Lamella03_FM_G1.ome.tiff",
    )
    section.emit_current_seed()  # what the editor does on open

    widget.resize(1360, 900)
    widget.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

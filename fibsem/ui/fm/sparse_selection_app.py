"""Standalone app for the sparse FM selection panes.

Opens the two halves side by side against a real or simulated microscope, so they can be
driven before the dialog that will hold them exists.

    python -m fibsem.ui.fm.sparse_selection_app                    # Demo, two beam views
    python -m fibsem.ui.fm.sparse_selection_app --manufacturer Thermo --ip 10.0.0.1

Right-click the left pane to add a region, then drag it or its corners. The right pane
shows the FM tiles that selection would acquire, recomputed on every change.

The wiring here is a stand-in for the dialog, not its design: it does the minimum to
make the two panes talk, so the pieces can be exercised on hardware while the dialog is
still being built.
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


def build_window(microscope: FibsemMicroscope):
    from PyQt5.QtWidgets import (
        QHBoxLayout,
        QLabel,
        QVBoxLayout,
        QWidget,
    )

    from fibsem.ui.fm.widgets.fm_region_selector import FMRegionSelectorWidget
    from fibsem.ui.fm.widgets.fm_tile_preview import FMTilePreviewWidget
    from fibsem.ui.stylesheets import CANVAS_BG
    from fibsem.ui.tokens import TEXT_COLOR, TEXT_MUTED_COLOR

    views = acquire_beam_overviews(microscope)
    # Back to the fluorescence pose, as the FM tab would leave it. The preview must be
    # right from either, which is what pinning its frame to the FM orientation is for --
    # try commenting this out, and nothing about the right pane should change.
    microscope.move_to_microscope("FM")

    selector = FMRegionSelectorWidget(microscope)
    preview = FMTilePreviewWidget(microscope)
    selector.set_views(views)

    status = QLabel()
    status.setStyleSheet(f"color:{TEXT_COLOR};font-size:11px;padding:4px 8px;")
    hint = QLabel(
        "right-click the beam overview to add a region  ·  drag it or its corners  ·  "
        "click a tile on the right to add or drop it"
    )
    hint.setStyleSheet(f"color:{TEXT_MUTED_COLOR};font-size:10px;padding:0 8px 4px;")

    def recompute():
        regions = selector.regions
        if not regions or selector.base is None or selector.projection is None:
            preview.clear()
        else:
            preview.set_selection(
                regions, selector.base, selector.projection, OVERLAP
            )
        describe()

    def describe():
        plan = preview.plan
        if plan is None:
            status.setText(f"{selector.describe_selected()}   —   no tiles selected")
            return
        minutes = preview.enabled_tiles * 3 * 4 // 60
        status.setText(
            f"{selector.describe_selected()}   —   "
            f"{plan.rows} x {plan.cols} grid, "
            f"{preview.enabled_tiles} of {preview.total_tiles} tiles, "
            f"~{minutes} min"
        )

    selector.regions_changed.connect(recompute)
    selector.selection_changed.connect(lambda _index: describe())
    preview.selection_changed.connect(describe)

    panes = QHBoxLayout()
    panes.setSpacing(8)
    panes.addWidget(selector, 3)
    panes.addWidget(preview, 2)

    window = QWidget()
    window.setWindowTitle("Sparse FM selection — panes")
    window.setStyleSheet(f"QWidget {{ background: {CANVAS_BG}; }}")
    layout = QVBoxLayout(window)
    layout.setContentsMargins(8, 8, 8, 8)
    layout.addLayout(panes)
    layout.addWidget(status)
    layout.addWidget(hint)
    window.resize(1180, 680)
    describe()
    return window


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
    window = build_window(microscope)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())

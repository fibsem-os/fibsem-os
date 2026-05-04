"""Test script for CorrelationPointPickerWidget.

Loads real FIB and FM images, pre-seeds coordinates from CSV, and shows
the full three-panel picker widget with an event log below.

Usage
-----
    python fibsem/correlation/ui/widgets/test_correlation_point_picker_widget.py
"""
import os
import sys

import pandas as pd
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.stylesheets import NAPARI_STYLE

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    PointType,
    PointXYZ,
)
from fibsem.correlation.ui.widgets.correlation_point_picker_widget import (
    CorrelationPointPickerWidget,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage

_DEV_PATH = "/home/patrick/github/fibsem/fibsem/applications/test-data"
_FIB_IMAGE = "ref_ReferenceImage-Spot-Burn-Fiducial-10-36-30_res_02_ib.tif"
_FM_IMAGE = "zstack-Feature-1-Active-002.ome.tiff"
_CSV = "data2.csv"

_TYPE_MAP = {
    "FIB": PointType.FIB,
    "FM": PointType.FM,
    "POI": PointType.POI,
    "Surface": PointType.SURFACE,
}


def _load_coords(csv_path: str) -> list[Coordinate]:
    df = pd.read_csv(csv_path)
    coords = []
    for _, row in df.iterrows():
        pt = _TYPE_MAP.get(row["type"])
        if pt is None or pt == PointType.SURFACE:
            continue
        coords.append(Coordinate(PointXYZ(row["x"], row["y"], row["z"]), pt))
    return coords


class TestWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("CorrelationPointPickerWidget — test")
        self.resize(1400, 800)

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        # Picker widget
        self.picker = CorrelationPointPickerWidget()
        root_layout.addWidget(self.picker, stretch=1)

        # Event log strip
        log_container = QWidget()
        log_container.setFixedHeight(120)
        log_container.setStyleSheet("background: #1e2124;")
        log_layout = QVBoxLayout(log_container)
        log_layout.setContentsMargins(8, 4, 8, 4)
        log_layout.setSpacing(2)
        log_layout.addWidget(QLabel("<b>Event log</b>"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setStyleSheet("font-family: arial; font-size: 10px; background: #1e2124; border: none;")
        log_layout.addWidget(self.log)
        root_layout.addWidget(log_container)

        self.picker.data_changed.connect(self._on_data_changed)

        self._load_dev()

    def _load_dev(self) -> None:
        fib_path = os.path.join(_DEV_PATH, _FIB_IMAGE)
        fm_path = os.path.join(_DEV_PATH, _FM_IMAGE)
        csv_path = os.path.join(_DEV_PATH, _CSV)

        if os.path.exists(fib_path):
            try:
                fib = FibsemImage.load(fib_path)
                self.picker.set_fib_image(fib)
                self._log(f"FIB loaded: {_FIB_IMAGE}  shape={fib.data.shape}")
            except Exception as exc:
                self._log(f"FIB load error: {exc}")
        else:
            self._log(f"FIB image not found: {fib_path}")

        if os.path.exists(fm_path):
            try:
                fm = FluorescenceImage.load(fm_path)
                self.picker.set_fm_image(fm)
                self._log(f"FM loaded: {_FM_IMAGE}  shape={fm.data.shape}")
            except Exception as exc:
                self._log(f"FM load error: {exc}")
        else:
            self._log(f"FM image not found: {fm_path}")

        if os.path.exists(csv_path):
            try:
                all_coords = _load_coords(csv_path)
                fib_coords = [c for c in all_coords if c.point_type == PointType.FIB]
                fm_coords  = [c for c in all_coords if c.point_type == PointType.FM]
                poi_coords = [c for c in all_coords if c.point_type == PointType.POI]
                data = CorrelationInputData(
                    fib_coordinates=fib_coords,
                    fm_coordinates=fm_coords,
                    poi_coordinates=poi_coords,
                )
                self.picker.set_data(data)
                self._log(
                    f"Coords loaded: {len(fib_coords)} FIB, "
                    f"{len(fm_coords)} FM, {len(poi_coords)} POI"
                )
            except Exception as exc:
                self._log(f"CSV load error: {exc}")

    def _on_data_changed(self, data: CorrelationInputData) -> None:
        self._log(
            f"data_changed: {len(data.fib_coordinates)} FIB, "
            f"{len(data.fm_coordinates)} FM, {len(data.poi_coordinates)} POI"
        )

    def _log(self, msg: str) -> None:
        self.log.append(msg)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(NAPARI_STYLE)

    win = TestWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

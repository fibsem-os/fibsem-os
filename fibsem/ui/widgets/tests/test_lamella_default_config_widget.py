"""Test script for LamellaDefaultConfigWidget."""
import sys

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from fibsem.applications.autolamella.structures import LamellaDefaultConfig
from fibsem.structures import DEFAULT_ALIGNMENT_AREA, FibsemRectangle, Point
from fibsem.ui.widgets.lamella_default_config_widget import LamellaDefaultConfigWidget


def main():
    app = QApplication(sys.argv)

    win = QWidget()
    win.setWindowTitle("LamellaDefaultConfigWidget test")
    win.setStyleSheet("background: #1a1b1e;")
    layout = QVBoxLayout(win)
    layout.setContentsMargins(12, 12, 12, 12)
    layout.setSpacing(8)

    # status label shows emitted template on every change
    status = QLabel("(change a value to see emitted template)")
    status.setStyleSheet("color: #a0a0a0; font-size: 10px;")
    status.setWordWrap(True)

    widget = LamellaDefaultConfigWidget()

    # load a non-default template so all controls are exercised
    template = LamellaDefaultConfig(
        use_petname=False,
        name_prefix="GridA",
        alignment_area=FibsemRectangle.from_dict(DEFAULT_ALIGNMENT_AREA),
        poi=Point(x=1.5e-6, y=-2.0e-6),
    )
    widget.set_template(template)

    def _on_changed(t: LamellaDefaultConfig):
        aa = t.alignment_area
        poi = t.poi
        aa_str = f"FibsemRectangle({aa.left:.3f}, {aa.top:.3f}, {aa.width:.3f}, {aa.height:.3f})" if aa else "None"
        poi_str = f"Point({poi.x*1e6:.3f} µm, {poi.y*1e6:.3f} µm)" if poi else "None"
        status.setText(
            f"use_petname={t.use_petname}  prefix='{t.name_prefix}'\n"
            f"alignment_area={aa_str}\n"
            f"poi={poi_str}"
        )

    widget.template_changed.connect(_on_changed)

    layout.addWidget(widget)
    layout.addWidget(status)
    layout.addStretch()

    win.resize(780, 340)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""Standalone test script for MillingTaskViewerWidget.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_task_viewer_widget.py
"""
import sys

import napari
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import utils
from fibsem.config import AUTOLAMELLA_TASK_PROTOCOL_PATH
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.structures import FibsemImage
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

import warnings

# Suppress a specific upstream Napari/NumPy warning from shapes miter computation.
warnings.filterwarnings(
    "ignore",
    message=r"'where' used without 'out', expect unit?ialized memory in output\. If this is intentional, use out=None\.",
    category=UserWarning,
    module=r"napari\.layers\.shapes\._shapes_utils",
)

def _load_task_configs(path: str) -> dict:
    try:
        protocol = AutoLamellaTaskProtocol.load(path)
    except Exception as exc:
        print(f"[warn] Could not load protocol from {path}: {exc}")
        return {}
    configs = {}
    for task_name, task_cfg in protocol.task_config.items():
        for milling_name, milling_cfg in task_cfg.milling.items():
            configs[f"{task_name} / {milling_name}"] = milling_cfg
    return configs


def _add_fib_image(viewer: napari.Viewer, hfw: float = 150e-6):
    """Add a blank FIB image to the viewer and return (FibsemImage, napari layer)."""
    image = FibsemImage.generate_blank_image(hfw=hfw, random=True)
    layer = viewer.add_image(
        image.data,
        name="FIB Image",
        colormap="gray",
        opacity=0.9,
        blending="additive",
    )
    return image, layer


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    task_configs = _load_task_configs(AUTOLAMELLA_TASK_PROTOCOL_PATH)
    initial_config = next(iter(task_configs.values()), FibsemMillingTaskConfig())

    # ── napari viewer ────────────────────────────────────────────────────
    viewer = napari.Viewer(title="MillingTaskViewerWidget — test")

    # Add a blank FIB image so pattern display works from the start
    fib_image, fib_layer = _add_fib_image(viewer, hfw=initial_config.field_of_view)

    # ── sidebar container ────────────────────────────────────────────────
    sidebar = QWidget()
    sidebar.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    sidebar.resize(480, 900)
    sidebar_layout = QVBoxLayout(sidebar)
    sidebar_layout.setContentsMargins(8, 8, 8, 8)
    sidebar_layout.setSpacing(8)

    # ── task selector ────────────────────────────────────────────────────
    row = QWidget()
    row_layout = QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)

    lbl = QLabel("Task:")
    lbl.setStyleSheet("font-weight: bold;")
    row_layout.addWidget(lbl)

    task_list = QListWidget()
    task_list.setStyleSheet("background: #1e2124; border: none;")
    task_list.setFixedHeight(120)
    for key in task_configs:
        task_list.addItem(key)
    row_layout.addWidget(task_list, 1)
    sidebar_layout.addWidget(row)

    # ── viewer widget ────────────────────────────────────────────────────
    widget = MillingTaskViewerWidget(
        microscope=microscope,
        viewer=viewer,
        milling_task_config=initial_config,
        milling_enabled=True,
    )
    # Inject the blank FIB image so patterns render immediately
    widget.set_fib_image(fib_image, fib_layer)
    sidebar_layout.addWidget(widget, 1)

    # ── status + buttons ─────────────────────────────────────────────────
    status = QLabel("Load a task to see patterns drawn in the viewer.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    sidebar_layout.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)

    btn_print = QPushButton("Print get_config()")
    btn_refresh = QPushButton("Refresh patterns")
    btn_clear = QPushButton("Clear")
    btn_layout.addWidget(btn_print)
    btn_layout.addWidget(btn_refresh)
    btn_layout.addWidget(btn_clear)
    sidebar_layout.addWidget(btn_row)

    # ── dock into napari ─────────────────────────────────────────────────
    viewer.window.add_dock_widget(sidebar, area="right", name="Milling Task")

    # ── connections ──────────────────────────────────────────────────────
    def on_task_selected(item) -> None:
        key = item.text()
        cfg = task_configs.get(key)
        if cfg is None:
            return
        widget.update_from_settings(cfg)
        # Regenerate blank image at the task's field of view
        nonlocal fib_image, fib_layer
        try:
            viewer.layers.remove("FIB Image")
        except Exception:
            pass
        fib_image, fib_layer = _add_fib_image(viewer, hfw=cfg.field_of_view)
        widget.set_fib_image(fib_image, fib_layer)
        status.setText(f"Loaded: {key}  ({len(cfg.stages)} stages)")

    def on_settings_changed(cfg: FibsemMillingTaskConfig) -> None:
        status.setText(
            f"settings_changed: {cfg.name}  fov={cfg.field_of_view * 1e6:.1f} µm  "
            f"{len(cfg.stages)} stages"
        )

    def on_print() -> None:
        cfg = widget.get_config()
        print(f"\nget_config():")
        print(f"  name={cfg.name}  fov={cfg.field_of_view * 1e6:.1f} µm")
        for s in cfg.stages:
            print(
                f"  [{s.name}]  pattern={s.pattern.__class__.__name__}"
                f"  depth={getattr(s.pattern, 'depth', None)}  "
                f"current={s.milling.milling_current * 1e12:.1f} pA  "
                f"strategy={s.strategy.name}"
            )

    task_list.itemClicked.connect(on_task_selected)
    widget.settings_changed.connect(on_settings_changed)
    btn_print.clicked.connect(on_print)
    btn_refresh.clicked.connect(widget._update_pattern_display)
    btn_clear.clicked.connect(widget.clear)

    # Select first task
    if task_list.count():
        task_list.setCurrentRow(0)
        on_task_selected(task_list.item(0))

    napari.run()


if __name__ == "__main__":
    main()

"""Standalone test script for MillingTaskConfigWidget2.

Run directly:
    python fibsem/ui/widgets/tests/test_milling_task_config_widget2.py
"""
import sys

from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt

from fibsem import utils
from fibsem.config import AUTOLAMELLA_TASK_PROTOCOL_PATH
from fibsem.applications.autolamella.structures import AutoLamellaTaskProtocol
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui.widgets.milling_task_config_widget2 import MillingTaskConfigWidget2


def _load_task_configs(path: str) -> dict:
    """Load FibsemMillingTaskConfig entries from the default task protocol."""
    try:
        protocol = AutoLamellaTaskProtocol.load(path)
    except Exception as exc:
        print(f"[warn] Could not load protocol from {path}: {exc}")
        return {}

    configs = {}
    for task_name, task_cfg in protocol.task_config.items():
        for milling_name, milling_cfg in task_cfg.milling.items():
            key = f"{task_name} / {milling_name}"
            configs[key] = milling_cfg
    return configs


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")

    task_configs = _load_task_configs(AUTOLAMELLA_TASK_PROTOCOL_PATH)
    initial_config = next(iter(task_configs.values()), FibsemMillingTaskConfig())

    # ── window ──────────────────────────────────────────────────────────
    win = QWidget()
    win.setWindowTitle("MillingTaskConfigWidget2 — test")
    win.setStyleSheet("background: #2b2d31; color: #d1d2d4;")
    win.resize(900, 800)

    outer = QHBoxLayout(win)
    outer.setContentsMargins(12, 12, 12, 12)
    outer.setSpacing(8)

    # ── left: task selector ─────────────────────────────────────────────
    left = QWidget()
    left_layout = QVBoxLayout(left)
    left_layout.setContentsMargins(0, 0, 0, 0)
    left_layout.setSpacing(4)

    lbl_tasks = QLabel("Tasks")
    lbl_tasks.setStyleSheet("font-weight: bold;")
    left_layout.addWidget(lbl_tasks)

    task_list = QListWidget()
    task_list.setStyleSheet("background: #1e2124; border: none;")
    task_list.setFixedWidth(220)
    for key in task_configs:
        task_list.addItem(key)
    left_layout.addWidget(task_list)

    outer.addWidget(left)

    # ── right: config widget + status + buttons ──────────────────────────
    right = QWidget()
    right_layout = QVBoxLayout(right)
    right_layout.setContentsMargins(0, 0, 0, 0)
    right_layout.setSpacing(8)

    widget = MillingTaskConfigWidget2(microscope=microscope, milling_task_config=initial_config)
    right_layout.addWidget(widget)

    status = QLabel("Select a task to load it, or change a setting to see settings_changed.")
    status.setStyleSheet("color: #909090; font-style: italic;")
    right_layout.addWidget(status)

    btn_row = QWidget()
    btn_layout = QHBoxLayout(btn_row)
    btn_layout.setContentsMargins(0, 0, 0, 0)
    btn_print = QPushButton("Print get_settings()")
    btn_clear = QPushButton("Clear")
    btn_layout.addWidget(btn_print)
    btn_layout.addWidget(btn_clear)
    right_layout.addWidget(btn_row)

    outer.addWidget(right, stretch=1)

    # ── connections ──────────────────────────────────────────────────────
    def on_task_selected(item) -> None:
        key = item.text()
        cfg = task_configs.get(key)
        if cfg is not None:
            widget.update_from_settings(cfg)
            status.setText(f"Loaded: {key}")

    def on_settings_changed(cfg: FibsemMillingTaskConfig) -> None:
        status.setText(
            f"settings_changed: {cfg.name}, fov={cfg.field_of_view*1e6:.1f} µm, "
            f"{len(cfg.stages)} stages"
        )

    def on_print() -> None:
        cfg = widget.get_settings()
        print(f"name={cfg.name}  fov={cfg.field_of_view*1e6:.1f} µm")
        for s in cfg.stages:
            print(f"  {s.name}  pattern={s.pattern.__class__.__name__}  strategy={s.strategy.name}")

    task_list.itemClicked.connect(on_task_selected)
    widget.settings_changed.connect(on_settings_changed)
    btn_print.clicked.connect(on_print)
    btn_clear.clicked.connect(widget.clear)

    # select first item if any
    if task_list.count():
        task_list.setCurrentRow(0)

    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

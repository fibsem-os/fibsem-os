"""The PC's own folder dialog, for the gateway's page: ``python -m
fibsem.gateway.ui.pick_folder <start> [title]`` prints the chosen folder as
JSON (``null`` when cancelled)."""

from __future__ import annotations

import json
import sys
from typing import List, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    start = args[0] if args else ""
    title = args[1] if len(args) > 1 else "Choose a folder"
    app = QApplication.instance() or QApplication([])
    # Windows won't let a process the browser didn't start take the
    # foreground, so the dialog is owned by a shown, invisible, topmost
    # window: owned windows stay above their owner, so it opens above the
    # browser (the first click gives it the keyboard).
    owner = QWidget(None, Qt.Tool | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    owner.setWindowOpacity(0.0)
    owner.resize(1, 1)
    screen = app.primaryScreen()
    if screen is not None:
        owner.move(screen.availableGeometry().center())
    owner.show()
    owner.raise_()
    owner.activateWindow()
    chosen = QFileDialog.getExistingDirectory(owner, title, start)
    owner.deleteLater()
    app.processEvents()
    print(json.dumps(chosen or None), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

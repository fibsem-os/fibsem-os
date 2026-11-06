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
    # Above the browser that asked for it.
    owner = QWidget(None, Qt.WindowStaysOnTopHint)
    chosen = QFileDialog.getExistingDirectory(owner, title, start)
    owner.deleteLater()
    app.processEvents()
    print(json.dumps(chosen or None), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

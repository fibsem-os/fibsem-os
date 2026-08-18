"""Loading an experiment's workflow must not make Qt complain.

Opening any experiment printed one ``QPixmap::scaled: Pixmap is a null pixmap``
per task in its workflow. ``WorkflowTaskRowWidget`` draws a drag handle from an
SVG under ``fibsem/ui/icons``, and it derived that path by counting
``os.path.dirname`` calls up from its own module -- the count that is right for
the three draggable list widgets that live under ``fibsem/ui``, but two short
for this one, which sits under ``fibsem/applications/autolamella/ui``. It
resolved to a directory that has never existed, ``QPixmap`` loaded nothing, and
scaling a null pixmap warns. The handle was silently absent from every row of a
list whose own caption says "Drag to reorder".

A Qt warning is not a Python warning: it goes to the message handler, so the
``filterwarnings = ["error"]`` in pyproject never sees it and it survives as
terminal noise indefinitely. Hence the handler below.

Two properties, because either alone lets the bug back in:

* **Populating the workflow emits no Qt warning.** Pinned at
  ``LamellaWorkflowWidget.set_workflow_config`` -- the call
  ``AutoLamellaMainUI._on_experiment_update`` makes when an experiment is
  adopted, which is where the four warnings came from.
* **Every draggable list widget's handle asset actually loads.** The warning is
  only the audible half of the failure; a guard that skips the scale silences it
  while leaving the handle missing. Three of these four modules spell the path
  differently, so the one that is wrong cannot be caught by reading them.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
from contextlib import contextmanager
from typing import List, Tuple

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QtWarningMsg, qInstallMessageHandler
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.ui.lamella_workflow_widget import (
    LamellaWorkflowWidget,
)

_app = QApplication.instance() or QApplication(sys.argv)

# The four widgets that draw a drag handle from the shared on-disk asset.
_DRAG_HANDLE_MODULES = [
    "fibsem.applications.autolamella.ui.workflow_config_widget",
    "fibsem.ui.widgets.milling_stage_list_widget",
    "fibsem.ui.fm.widgets.channel_list_widget",
    "fibsem.ui.correlation.widgets.coordinate_list_widget",
]

_TASKS = ["Setup Lamella Position", "Mill Fiducial", "Rough Milling", "Polishing"]


@contextmanager
def captured_qt_messages():
    """Collect (type, text) for every Qt message logged inside the block."""
    messages: List[Tuple[int, str]] = []

    def handler(mode, context, message):
        messages.append((mode, message))

    previous = qInstallMessageHandler(handler)
    try:
        yield messages
    finally:
        qInstallMessageHandler(previous)


def _workflow_config() -> AutoLamellaWorkflowConfig:
    return AutoLamellaWorkflowConfig(
        tasks=[
            AutoLamellaTaskDescription(name=name, supervise=False, required=False)
            for name in _TASKS
        ]
    )


def test_populating_the_workflow_emits_no_qt_warning():
    """One warning per task row is what an experiment load used to print."""
    widget = LamellaWorkflowWidget()

    with captured_qt_messages() as messages:
        widget.set_workflow_config(_workflow_config())

    warnings = [text for mode, text in messages if mode >= QtWarningMsg]
    assert warnings == []

    widget.close()


@pytest.mark.parametrize("module_name", _DRAG_HANDLE_MODULES)
def test_every_drag_handle_asset_loads(module_name):
    """A path that resolves nowhere leaves the handle missing, warning or not."""
    module = __import__(module_name, fromlist=["_DRAG_HANDLE_PATH"])
    path = module._DRAG_HANDLE_PATH

    assert os.path.exists(path), f"{module_name} points at a missing asset: {path}"
    assert not QPixmap(path).isNull(), f"{module_name} cannot load {path}"

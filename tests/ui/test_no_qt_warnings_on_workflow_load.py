"""Loading an experiment's workflow must not make Qt complain, and every
draggable row must actually end up with a handle.

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
* **Every draggable row's handle actually loads.** The warning is only the
  audible half of the failure; a guard that skips the scale silences it while
  leaving the handle missing.

The four modules no longer spell that path four different ways -- it lives once
in ``fibsem.ui.icon``, with ``drag_handle_pixmap()`` doing the ``isNull()``
check that used to be each caller's problem. So the second property is now
pinned on what each row ends up drawing, plus a guard that no module goes back
to naming the asset itself.
"""
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("napari")

from PyQt5.QtCore import (  # noqa: E402
    Qt,
    QtWarningMsg,
    qInstallMessageHandler,
)
from PyQt5.QtGui import QPixmap  # noqa: E402
from PyQt5.QtWidgets import QLabel  # noqa: E402

from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskDescription,
    AutoLamellaWorkflowConfig,
)
from fibsem.applications.autolamella.ui.lamella_workflow_widget import (  # noqa: E402
    LamellaWorkflowWidget,
)
from fibsem.applications.autolamella.ui.workflow_config_widget import (  # noqa: E402
    WorkflowTaskRowWidget,
)
from fibsem.correlation.structures import Coordinate  # noqa: E402
from fibsem.fm.structures import ChannelSettings  # noqa: E402
from fibsem.milling.base import FibsemMillingStage  # noqa: E402
from fibsem.milling.patterning import get_pattern_names  # noqa: E402
from fibsem.milling.strategy import get_strategy_names  # noqa: E402
from fibsem.ui import icon  # noqa: E402
from fibsem.ui.correlation.widgets.coordinate_list_widget import (  # noqa: E402
    CoordinateRowWidget,
)
from fibsem.ui.fm.widgets.channel_list_widget import ChannelRowWidget  # noqa: E402
from fibsem.ui.icon import (  # noqa: E402
    DRAG_HANDLE_HEIGHT,
    DRAG_HANDLE_PATH,
    DRAG_HANDLE_WIDTH,
    drag_handle_pixmap,
)
from fibsem.ui.widgets.milling_stage_list_widget import (  # noqa: E402
    MillingStageRowWidget,
)

NULL_PIXMAP_WARNING = "null pixmap"

TASKS = ["Setup Lamella Position", "Mill Fiducial", "Rough Milling", "Polishing"]


def _milling_row():
    return MillingStageRowWidget(
        FibsemMillingStage(),
        index=0,
        pattern_names=get_pattern_names(),
        strategy_names=get_strategy_names(),
    )


def _channel_row():
    # Empty filter-set lists are what the widget passes with no microscope attached.
    return ChannelRowWidget(
        channel=ChannelSettings(), emission_items=[], excitation_items=[]
    )


def _coordinate_row():
    return CoordinateRowWidget(coord=Coordinate(), name="P1")


def _workflow_row():
    return WorkflowTaskRowWidget(
        AutoLamellaTaskDescription(name="Mill Rough", supervise=True, required=True)
    )


# Every module that draws a drag handle, with a builder for its row. Adding a row
# type here is the cheap way to keep it drawing the shared, loaded asset.
DRAG_HANDLE_ROWS = {
    "fibsem.applications.autolamella.ui.workflow_config_widget": _workflow_row,
    "fibsem.ui.widgets.milling_stage_list_widget": _milling_row,
    "fibsem.ui.fm.widgets.channel_list_widget": _channel_row,
    "fibsem.ui.correlation.widgets.coordinate_list_widget": _coordinate_row,
}


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
            for name in TASKS
        ]
    )


def _drag_handle_label(row) -> QLabel:
    """The row's drag handle: the one QLabel carrying the open-hand cursor."""
    handles = [
        label
        for label in row.findChildren(QLabel)
        if label.cursor().shape() == Qt.CursorShape.OpenHandCursor
    ]
    assert len(handles) == 1, f"expected one drag handle, found {len(handles)}"
    return handles[0]


def test_populating_the_workflow_emits_no_qt_warning(qapp):
    """One warning per task row is what an experiment load used to print."""
    widget = LamellaWorkflowWidget()

    with captured_qt_messages() as messages:
        widget.set_workflow_config(_workflow_config())

    warnings = [text for mode, text in messages if mode >= QtWarningMsg]
    assert warnings == []

    widget.close()


def test_shared_asset_loads(qapp):
    assert Path(DRAG_HANDLE_PATH).is_file(), DRAG_HANDLE_PATH
    assert not QPixmap(DRAG_HANDLE_PATH).isNull(), DRAG_HANDLE_PATH


def test_helper_returns_the_row_sized_handle(qapp):
    pixmap = drag_handle_pixmap()
    assert not pixmap.isNull()
    assert (pixmap.width(), pixmap.height()) == (DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)


def test_helper_does_not_scale_a_missing_asset(qapp, tmp_path, monkeypatch):
    """A missing asset yields an empty pixmap, not a warning per call site."""
    monkeypatch.setattr(icon, "DRAG_HANDLE_PATH", str(tmp_path / "gone.svg"))

    with captured_qt_messages() as messages:
        pixmap = drag_handle_pixmap()

    assert pixmap.isNull()
    assert not [t for _, t in messages if NULL_PIXMAP_WARNING in t], messages
    QLabel().setPixmap(pixmap)  # still safe to hand straight to a row


@pytest.mark.parametrize("module_name", sorted(DRAG_HANDLE_ROWS))
def test_row_draws_the_loaded_handle(qapp, module_name):
    """A path that resolves nowhere leaves the handle missing, warning or not."""
    row = DRAG_HANDLE_ROWS[module_name]()

    pixmap = _drag_handle_label(row).pixmap()

    assert pixmap is not None and not pixmap.isNull(), f"{module_name}: handle missing"
    assert (pixmap.width(), pixmap.height()) == (DRAG_HANDLE_WIDTH, DRAG_HANDLE_HEIGHT)


@pytest.mark.parametrize("module_name", sorted(DRAG_HANDLE_ROWS))
def test_building_a_row_emits_no_null_pixmap_warning(qapp, module_name):
    # Only the null-pixmap warning is pinned per row: the workflow test above is
    # where "Qt said nothing at all" is asserted, on the path that used to warn.
    with captured_qt_messages() as messages:
        DRAG_HANDLE_ROWS[module_name]()

    assert not [t for _, t in messages if NULL_PIXMAP_WARNING in t], messages


@pytest.mark.parametrize("module_name", sorted(DRAG_HANDLE_ROWS))
def test_row_module_uses_the_shared_handle(qapp, module_name):
    """No module names the asset itself; that is how the paths drifted apart."""
    module = __import__(module_name, fromlist=["__file__"])
    source = Path(module.__file__).read_text(encoding="utf-8")

    assert "drag_handle.svg" not in source, f"{module_name} re-derives the asset path"
    assert "drag_handle_pixmap" in source, f"{module_name} does not use the helper"

"""A defect set in one list must reach every other display, and disk (FIB-564).

Two failures, one report. The visible one was the Experiment tab not redrawing. The one
that mattered was that a defect set from the minimap was never written at all:
persistence hung off a Qt signal that each host wired by hand, and two of the five hosts
did not.

The display half is pinned on the model rather than on any host. `Lamella` is
`@evented`, so a row subscribed to `events.defect` redraws no matter which widget made
the change -- that is the property worth protecting, because wiring N widgets to each
other is what produced the bug.

Driven through `LamellaNameListWidget` rather than a bare `_LamellaRow`: the defect
button is hidden until `enable_defect_button(True)`, and `refresh` skips a hidden
button, so a bare row silently asserts nothing.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import DefectState, DefectType
from fibsem.ui.widgets.lamella_name_list_widget import LamellaNameListWidget, _LamellaRow


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def lamella(tmp_path):
    """A real Lamella -- the events under test are the ones @evented generates."""
    from fibsem.applications.autolamella.structures import Lamella

    return Lamella(petname="01-test", path=str(tmp_path / "01-test"), number=1)


@pytest.fixture()
def row(qapp, lamella):
    """One row, shown, with the defect button enabled -- as AutoLamella builds it."""
    widget = LamellaNameListWidget()
    widget.enable_defect_button(True)
    widget.set_lamella([lamella])
    widget.show()
    widget._test_keepalive = widget  # rows are Qt-owned; keep the list alive for the test
    return list(widget._rows())[0]


def test_row_redraws_when_the_defect_changes_elsewhere(row, lamella):
    """The regression: the row kept the old icon until something refreshed it by hand."""
    assert row.btn_defect.toolTip() == "No defect"

    # Somebody else's widget sets it. Nothing calls into this row.
    lamella.defect = DefectState(state=DefectType.REWORK)

    assert row.btn_defect.toolTip() == "Rework required", (
        "row did not redraw on lamella.events.defect -- it was subscribed to "
        "description only, so a defect set anywhere else left a stale icon"
    )


def test_row_redraws_when_the_task_status_changes_elsewhere(row, lamella):
    """Same gap, same fix: status is the other thing a row draws from the model.

    Both fields, and status last: the label shows the task name only while the task is
    InProgress, so setting the name alone changes nothing and would let this pass
    whether or not the subscription exists.
    """
    from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

    assert row.status_label.text() == ""
    lamella.task_state.name = "Rough Milling"
    lamella.task_state.status = AutoLamellaTaskStatus.InProgress

    assert row.status_label.text() == "Rough Milling", (
        "row did not redraw on task_state.events.status"
    )


def test_row_redraws_when_the_description_changes(row, lamella):
    """The one subscription that already existed, kept honest while three were added."""
    lamella.description = "edited elsewhere"
    assert row.toolTip() == "edited elsewhere"


def test_row_survives_a_lamella_without_task_state(qapp):
    """This list also takes plain positions, unlike its two siblings.

    `_lamella_status_text` reaches for `task_state` with `getattr`, so subscribing to it
    unconditionally would crash on exactly the objects that guard exists for.
    """
    from dataclasses import dataclass, field

    from psygnal import evented

    @evented
    @dataclass
    class PlainPosition:
        name: str = "P1"
        description: str = ""
        defect: object = None
        # deliberately no task_state

    _LamellaRow(PlainPosition())  # must not raise


@pytest.mark.parametrize(
    "module, widget, handler",
    [
        ("fibsem.ui.FibsemMinimapWidget", "FibsemMinimapWidget", "_on_defect_changed"),
        (
            "fibsem.ui.widgets.fluorescence_coincidence_viewer_widget",
            "FluorescenceCoincidenceViewerWidget",
            "_on_lamella_defect_changed",
        ),
    ],
)
def test_every_host_of_the_list_can_persist_a_defect(module, widget, handler):
    """Both hosts embedding the list need somewhere for `defect_changed` to go.

    Read as source text rather than by construction: the minimap needs a napari viewer
    and the coincidence viewer a microscope and an experiment, neither worth standing up
    to assert a connection exists. What is pinned is that the signal is wired and the
    handler saves -- which the text carries.
    """
    import importlib
    from pathlib import Path

    src = Path(importlib.import_module(module).__file__).read_text()
    assert f"defect_changed.connect(self.{handler})" in src, (
        f"{widget} embeds the lamella list but never wires defect_changed -- "
        f"a defect set there is not persisted (FIB-564)"
    )
    assert f"def {handler}" in src, f"{widget}.{handler} is missing"
    assert "experiment.save()" in src, f"{widget}.{handler} must persist the change"

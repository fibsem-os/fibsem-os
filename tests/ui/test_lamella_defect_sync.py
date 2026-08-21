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

import ast
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import DefectState, DefectType
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
    _LamellaRow,
)


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
    """One row, shown, with the defect button enabled -- as AutoLamella builds it.

    Yielded rather than returned, and closed after: the rows are Qt-owned by the list,
    so the list has to outlive the test. An earlier version kept it alive with a
    self-reference, which made it cyclic garbage holding QObjects -- collected at an
    arbitrary later point, which crashed the xdist worker outright. See
    `test_main_thread_gc.py` for why QObject finalizers cannot run just anywhere.
    """
    widget = LamellaNameListWidget()
    widget.enable_defect_button(True)
    widget.set_lamella([lamella])
    widget.show()
    yield list(widget._rows())[0]
    widget.close()


def test_row_redraws_when_the_defect_changes_elsewhere(row, lamella):
    """The regression: the row kept the old icon until something refreshed it by hand."""
    assert row.btn_defect.toolTip() == "No defect"

    # Somebody else's widget sets it. Nothing calls into this row.
    lamella.defect = DefectState(state=DefectType.REWORK)

    assert row.btn_defect.toolTip() == "Rework required", (
        "row did not redraw on lamella.events.defect -- it was subscribed to "
        "description only, so a defect set anywhere else left a stale icon"
    )


def test_the_row_does_not_subscribe_to_task_state(row, lamella):
    """FIB-565 subscribed it; that is reverted, and the reason is worth keeping.

    `task_state.name`/`.status` are written by the workflow on the FunctionWorker
    thread. @ensure_main_thread made the callback land on the GUI thread, which is
    necessary and not sufficient: it also lands *later*, and by then `set_lamella` may
    have replaced the row and Qt destroyed its widgets --

        RuntimeError: wrapped C/C++ object of type QLabel has been deleted

    -- reported from a workflow run. Marshalling fixes which thread touches the widget,
    not whether the widget is still there.

    The row is refreshed from `_on_workflow_update` now, on the same named-lamella call
    as the two lists that were already refreshed there. `defect` and `description` keep
    their subscriptions: both are GUI-thread writes, and nothing else refreshes this
    list for them.
    """
    assert len(lamella.task_state.events.status) == 0, (
        "the row is connected to task_state.status again -- a cross-thread subscription "
        "on a widget that gets replaced"
    )
    assert len(lamella.task_state.events.name) == 0
    assert len(lamella.events.defect) == 1, "the GUI-thread subscriptions must stay"
    assert len(lamella.events.description) == 1


def test_the_workflow_update_refreshes_the_name_list(row, lamella):
    """What replaces the subscription. Without this the row goes stale mid-workflow.

    Nothing else refreshed this list during a run -- unlike the workflow list and the
    cards, which `_on_workflow_update` already covered -- so removing the subscription
    without adding this would have traded a crash for a stale display.
    """
    import inspect

    from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
        AutoLamellaSingleWindowUI,
    )

    source = inspect.getsource(AutoLamellaSingleWindowUI._on_workflow_update)

    assert "autolamella_ui.lamella_list.refresh_lamella(" in source, (
        "`_on_workflow_update` does not refresh the name list, and the row no longer "
        "subscribes -- so nothing keeps it current while a workflow runs"
    )


@pytest.mark.parametrize(
    "module, widget, handler",
    [
        ("fibsem.ui.FibsemMinimapWidget", "FibsemMinimapWidget", "_on_defect_changed"),
        (
            "fibsem.applications.autolamella.ui.fluorescence_coincidence_viewer_widget",
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

    src = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
    assert f"defect_changed.connect(self.{handler})" in src, (
        f"{widget} embeds the lamella list but never wires defect_changed -- "
        f"a defect set there is not persisted (FIB-564)"
    )
    assert f"def {handler}" in src, f"{widget}.{handler} is missing"
    assert "experiment.save()" in src, f"{widget}.{handler} must persist the change"


def _connect_chains(source: str):
    """Every `a.b.c.connect(...)` in *source*, as its attribute chain.

    AST rather than substring matching, because the two task_state connections in the
    list and card widgets are commented out rather than deleted (FIB-604) -- and a
    `"task_state.events" in src` check passes happily on a comment. That is the shape
    of test that passes for the wrong reason; the parser does not read comments at all.

    Hand-walked rather than `ast.unparse`, which is 3.9+. CI never catches that here
    (these tests need PyQt5, which the CI env does not install, so they are skipped
    there) but the package supports 3.8 and someone runs them locally on it.
    """
    chains = []
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "connect":
            continue
        parts, cur = [], node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        chains.append(list(reversed(parts)))
    return chains


@pytest.mark.parametrize(
    "module, widget",
    [
        (
            "fibsem.applications.autolamella.ui.lamella_name_list_widget",
            "_LamellaRow",
        ),
        ("fibsem.applications.autolamella.ui.lamella_list_widget", "LamellaRowWidget"),
        (
            "fibsem.applications.autolamella.ui.lamella_card_widget",
            "LamellaCardWidget",
        ),
    ],
)
def test_no_widget_holds_a_live_task_state_subscription(module, widget):
    """None of the three may subscribe to task_state. Inverted from FIB-565.

    FIB-565 required the opposite -- subscribe, but marshal. That was the wrong
    conclusion, and it took a day to show it. Marshalling fixes *which thread* touches
    the widget and says nothing about *whether the widget still exists*: the callback
    arrives later, by which point the list may have been cleared and Qt may have
    destroyed the receiver.

    The cost of being wrong is not a logged traceback. No `sys.excepthook` is installed
    anywhere in the package, so PyQt5 calls qFatal and the process aborts (FIB-329). On
    2026-08-12 that killed a run mid-workflow and, because the abort landed inside a
    thumbnail write, left a 0-byte file that aborted every subsequent load (FIB-602).

    `_LamellaRow`'s connections are deleted (FIB-603); the other two are commented out
    pending FIB-604, which has to give the refresh a trigger that fires after the write
    before they can go for good. Either way there is nothing live for the parser to
    find, which is what this asserts.

    Resolved through `importlib` rather than a hardcoded path: these files moved package
    twice in one day (FIB-558/559), and a path-based read would have broken silently
    rather than failed loudly.
    """
    import importlib
    from pathlib import Path

    src = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")
    live = [c for c in _connect_chains(src) if "task_state" in c]

    assert live == [], (
        f"{widget} has a live task_state subscription {live} -- written from the "
        f"workflow's worker thread, delivered to a widget that may since have been "
        f"destroyed, and fatal rather than logged when it lands (FIB-329/603/604)"
    )


@pytest.mark.parametrize(
    "module, widget",
    [
        (
            "fibsem.applications.autolamella.ui.lamella_name_list_widget",
            "_LamellaRow",
        ),
        ("fibsem.applications.autolamella.ui.lamella_list_widget", "LamellaRowWidget"),
        (
            "fibsem.applications.autolamella.ui.lamella_card_widget",
            "LamellaCardWidget",
        ),
    ],
)
def test_every_refresh_stays_marshalled(module, widget):
    """@ensure_main_thread stays on all three, even with the subscriptions gone.

    Not redundant. It is what makes re-enabling a connection -- or any future
    cross-thread caller -- land on the GUI thread rather than corrupting Qt state from
    a worker. Removing the marshalling is the change that would make the commented-out
    lines in the list and card widgets dangerous to restore.
    """
    import importlib
    from pathlib import Path

    src = Path(importlib.import_module(module).__file__).read_text(encoding="utf-8")

    assert "from superqt import ensure_main_thread" in src, f"{widget} lost the import"
    assert "    @ensure_main_thread\n    def refresh(self)" in src, (
        f"{widget}.refresh is not marshalled (FIB-565)"
    )

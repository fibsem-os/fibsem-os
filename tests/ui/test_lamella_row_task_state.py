"""A lamella row does not subscribe to task state across threads.

Reported from a workflow run:

    RuntimeError: wrapped C/C++ object of type QLabel has been deleted
      lamella_name_list_widget.py in refresh -> self.name_label.setText(...)

FIB-565 subscribed the row to `task_state.events.name` / `.status` so it would follow a
running workflow. Those are written from the workflow's FunctionWorker thread, so
`refresh` is marshalled and arrives *later* — by which point `set_lamella` may have
replaced the row and Qt may have destroyed its widgets.

The comment above those connections argued no disconnect was needed, because psygnal
holds bound methods weakly and a dropped row would be collected. True of the Python
object and beside the point: Qt's teardown does not wait for the collector, so a live
weakref over a destroyed QLabel is exactly the reachable state. Same family as FIB-550.

The fix is removal rather than a disconnect. What replaces it is
`AutoLamellaMainUI._on_workflow_update`, which already refreshed the workflow list and
the cards on every update, over a Qt signal -- and already knew *which* lamella changed,
so the name list joins them on `refresh_lamella` rather than `refresh_all`. A workflow
touches one lamella at a time, and `refresh_all` is O(rows) of GUI-thread work for it:
~2.4 ms per update at 70 lamellae, against ~0.05 ms targeted.

`defect` and `description` stay. They are written by GUI widgets on the GUI thread, and
nothing else refreshes this list when one changes (FIB-564).
"""

import ast
import inspect
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import sys
import textwrap

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskState,
    AutoLamellaTaskStatus,
    Lamella,
)
from fibsem.applications.autolamella.ui.lamella_name_list_widget import (
    LamellaNameListWidget,
    _LamellaRow,
)

_app = QApplication.instance() or QApplication(sys.argv)


def lamella(tmp_path, number: int = 1) -> Lamella:
    lam = Lamella(
        path=str(tmp_path / f"lam{number}"), number=number, petname=f"lam-{number}"
    )
    lam.task_state = AutoLamellaTaskState(name="MillTrench")
    return lam


@pytest.fixture
def widget():
    w = LamellaNameListWidget()
    yield w
    w.deleteLater()


class TestItDoesNotSubscribeAcrossThreads:
    def test_the_row_does_not_connect_to_task_state(self):
        """Structural, and the whole fix.

        Behavioural coverage cannot carry this alone: reaching the crash needs the row's
        C++ object destroyed while its Python wrapper lives, and `QListWidget.clear()`
        does not destroy it in an offscreen test — measured with `sip.isdeleted`. So what
        is pinned is the absence of the subscription, since that is what makes the state
        unreachable however it would otherwise have been arrived at.
        """
        source = textwrap.dedent(inspect.getsource(_LamellaRow.__init__))
        # Hand-walked rather than `ast.unparse`, which is 3.9+. The package supports
        # 3.8, and CI cannot catch the difference here -- these tests need PyQt5, which
        # the CI env does not install, so they are skipped there and a 3.9-only call
        # would only fail for someone running them locally on 3.8.
        connects = []
        for node in ast.walk(ast.parse(source)):
            if not (
                isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            ):
                continue
            if node.func.attr != "connect":
                continue
            parts, cur = [], node.func
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            connects.append(list(reversed(parts)))
        task_state_subscriptions = [c for c in connects if "task_state" in c]

        assert task_state_subscriptions == [], (
            f"the row subscribes to {task_state_subscriptions} — written from the "
            f"workflow thread and delivered to a row that may have been replaced by "
            f"then. `_on_workflow_update` already calls `refresh_all()` for this"
        )

    def test_the_row_is_not_connected_to_task_state(self, widget, tmp_path):
        """The behavioural counterpart, asserted on the connection itself.

        Not by counting calls: psygnal binds the method at connect time, so replacing
        `row.refresh` afterwards intercepts nothing and such a test passes against the
        unfixed code too. `len(signal)` is the number of live callbacks, which is the
        thing that actually decides whether the row can be reached.
        """
        lam = lamella(tmp_path, 1)

        widget.set_lamella([lam])

        assert len(lam.task_state.events.status) == 0, (
            "the row is connected to task_state.status — written from the workflow "
            "thread, delivered to a row that may since have been replaced"
        )
        assert len(lam.task_state.events.name) == 0


class TestWhatItStillFollows:
    def test_it_still_follows_defect_and_description(self, widget, tmp_path):
        """GUI-thread writes, and nothing else refreshes the list for them (FIB-564)."""
        lam = lamella(tmp_path, 1)

        widget.set_lamella([lam])

        assert len(lam.events.defect) == 1, "the row stopped following defect changes"
        assert len(lam.events.description) == 1

    def test_refresh_lamella_is_what_follows_a_workflow(self, widget, tmp_path):
        """What replaces the subscription, and it must actually update the row."""
        lamellae = [lamella(tmp_path, i) for i in range(1, 4)]
        widget.set_lamella(lamellae)
        running = lamellae[1]
        running.task_state.name = "MillPolishing"
        running.task_state.status = AutoLamellaTaskStatus.InProgress

        widget.refresh_lamella(running)

        row = widget._list.itemWidget(widget._list.item(1))
        assert row.status_label.text() == "MillPolishing", (
            "refresh_lamella did not bring the row up to date, so removing the "
            "subscription does lose the live status after all"
        )

    def test_it_refreshes_only_the_lamella_that_changed(
        self, widget, tmp_path, monkeypatch
    ):
        """The reason it is not `refresh_all`.

        A workflow update names one lamella; repainting the rest is O(rows) of
        GUI-thread work for nothing, on every update of a run (~2.4 ms at 70 lamellae,
        mostly `setStyleSheet` re-parsing). Patching the class attribute works where the
        psygnal equivalent did not -- `refresh_lamella` looks the method up on the
        instance each call, rather than having bound it once.
        """
        lamellae = [lamella(tmp_path, i) for i in range(1, 4)]
        widget.set_lamella(lamellae)
        refreshed: list = []
        monkeypatch.setattr(
            _LamellaRow, "refresh", lambda self: refreshed.append(self.lamella.name)
        )

        widget.refresh_lamella(lamellae[1])

        assert refreshed == [lamellae[1].name], (
            f"refreshed {refreshed} for a change to one lamella -- every extra row is a "
            f"wasted repaint on every workflow update"
        )

    def test_refresh_all_remains_for_an_unidentified_lamella(self, widget, tmp_path):
        """The fallback branch: no name to match on, so every row is refreshed."""
        lamellae = [lamella(tmp_path, i) for i in range(1, 4)]
        widget.set_lamella(lamellae)
        for lam in lamellae:
            lam.task_state.name = "MillPolishing"
            lam.task_state.status = AutoLamellaTaskStatus.InProgress

        widget.refresh_all()

        for i in range(widget._list.count()):
            row = widget._list.itemWidget(widget._list.item(i))
            assert row.status_label.text() == "MillPolishing"

    def test_the_main_ui_refreshes_the_name_list_on_every_workflow_update(self):
        """The claim that makes removal a fix rather than a trade.

        Both branches, because they are what the subscription is traded for: the named
        one on the path a running workflow takes, `refresh_all` when the update cannot
        be tied to a lamella.
        """
        from fibsem.applications.autolamella.ui.AutoLamellaMainUI import (
            AutoLamellaSingleWindowUI,
        )

        source = inspect.getsource(AutoLamellaSingleWindowUI._on_workflow_update)

        # `autolamella_ui.lamella_list`, not `lamella_list_widget`. Those are different
        # widgets -- `lamella_list_widget` is the *workflow* list, which was already
        # refreshed here and is not the one this file is about. An earlier version
        # asserted the wrong one and passed while proving nothing.
        assert "autolamella_ui.lamella_list.refresh_lamella(" in source, (
            "`_on_workflow_update` no longer refreshes the name list, so nothing "
            "follows a running workflow now that the subscription is gone"
        )
        assert "autolamella_ui.lamella_list.refresh_all()" in source, (
            "the name list is not refreshed when the update names no lamella, so it "
            "goes stale on exactly the updates the other two lists still handle"
        )

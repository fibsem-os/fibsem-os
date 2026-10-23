"""A model's signal reaches a widget's slot on the GUI thread, with the
arguments the slot takes, until the widget goes or the connection is dropped."""

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal import Signal
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QWidget

from fibsem.ui.qt.connections import connect_on_gui_thread


class _Model:
    changed = Signal(str, object)


def test_an_emission_from_a_worker_arrives_on_the_gui_thread(qapp):
    model, widget, heard = _Model(), QWidget(), []
    connect_on_gui_thread(
        widget,
        model.changed,
        lambda what, detail: heard.append(
            (what, detail, QThread.currentThread() is qapp.thread())
        ),
    )

    worker = threading.Thread(target=lambda: model.changed.emit("stage", 1))
    worker.start()
    worker.join()
    assert heard == [], "a worker's emission waits for the GUI thread"
    qapp.processEvents()

    assert heard == [("stage", 1, True)]


def test_a_slot_gets_only_the_arguments_it_takes(qapp):
    model, widget, heard = _Model(), QWidget(), []

    def on_changed(what):
        heard.append(what)

    connect_on_gui_thread(widget, model.changed, on_changed)
    model.changed.emit("experiment", object())

    assert heard == ["experiment"]


def test_nothing_arrives_once_disconnected(qapp):
    model, widget, heard = _Model(), QWidget(), []
    disconnect = connect_on_gui_thread(widget, model.changed, heard.append)

    disconnect()
    model.changed.emit("protocol", None)
    qapp.processEvents()

    assert heard == []


def test_a_dropped_widget_is_not_kept_alive_by_the_model(qapp):
    """A bound method is held weakly, as psygnal holds one: a widget nobody closed
    goes when it is dropped, and takes its connection with it."""
    import gc

    class _Listener(QWidget):
        def on_changed(self, what):
            raise AssertionError("a dropped widget was still called")

    model = _Model()
    widget = _Listener()
    connect_on_gui_thread(widget, model.changed, widget.on_changed)
    assert len(model.changed) == 1

    del widget
    gc.collect()

    assert len(model.changed) == 0
    model.changed.emit("stage", None)
    qapp.processEvents()


def test_the_connection_lasts_as_long_as_the_widget(qapp):
    """Held weakly is not dropped early: the widget keeps its relay."""
    import gc

    model, widget, heard = _Model(), QWidget(), []
    connect_on_gui_thread(widget, model.changed, heard.append)

    gc.collect()
    model.changed.emit("stage", None)

    assert heard == ["stage"]

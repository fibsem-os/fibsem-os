"""Headless tests for the mouse-wheel guard on spinboxes and comboboxes.

Guards two contracts:

1. Scrolling over a guarded input never changes its value, and the wheel is
   forwarded to the enclosing scroll area so the panel still scrolls (no
   "dead zone" over every input).
2. Guarding is idempotent — installing twice must not forward the event twice,
   which would scroll at double speed.

Uses PyQt5 directly with the offscreen platform (no pytest-qt dependency).
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from fibsem.ui.utils import install_wheel_blocker, install_wheel_blocker_recursive
from fibsem.ui.widgets.custom_widgets import (
    IntegerValueSpinBox,
    ValueComboBox,
    ValueSpinBox,
)


def _scroll_area(factory, count=40):
    """A scroll area tall enough to scroll, filled with widgets from *factory*."""
    area = QScrollArea()
    inner = QWidget()
    layout = QVBoxLayout(inner)
    widgets = []
    for _ in range(count):
        w = factory()
        layout.addWidget(w)
        widgets.append(w)
    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.resize(300, 200)
    area.show()
    QApplication.processEvents()
    return area, widgets


def _wheel(widget):
    return QWheelEvent(
        QPoint(10, 10),
        widget.mapToGlobal(QPoint(10, 10)),
        QPoint(0, -120),
        QPoint(0, -120),
        -120,
        Qt.Vertical,
        Qt.NoButton,
        Qt.NoModifier,
    )


def _spin(**kwargs):
    w = ValueSpinBox(minimum=0, maximum=1000, **kwargs)
    w.setValue(50)
    return w


def test_unguarded_spinbox_reproduces_the_bug(qapp):
    """Baseline: a plain QDoubleSpinBox eats the scroll AND changes its value."""
    area, widgets = _scroll_area(
        lambda: (lambda w: (w.setRange(0, 1000), w.setValue(50), w)[-1])(QDoubleSpinBox())
    )
    w = widgets[0]
    before = w.value()
    QApplication.sendEvent(w, _wheel(w))
    QApplication.processEvents()

    assert w.value() != before, "expected the unguarded bug: value changes on scroll"
    assert area.verticalScrollBar().value() == 0, "unguarded widget also swallows the scroll"
    area.close()


@pytest.mark.parametrize("index", [0, 3], ids=["auto-focused-first", "unfocused"])
def test_guarded_spinbox_scrolls_without_changing_value(qapp, index):
    """Holds for the auto-focused first widget too.

    Qt focuses the first focusable widget in a form automatically, so a
    focus-conditional guard would leave that one widget still vulnerable.
    """
    area, widgets = _scroll_area(_spin)
    w = widgets[index]
    value_before = w.value()
    scroll_before = area.verticalScrollBar().value()

    QApplication.sendEvent(w, _wheel(w))
    QApplication.processEvents()

    assert w.value() == value_before, "scrolling must not change the value"
    assert area.verticalScrollBar().value() > scroll_before, "panel must still scroll"
    area.close()


def test_guarding_is_idempotent(qapp):
    """Installing repeatedly must not forward the wheel event more than once."""
    area, widgets = _scroll_area(_spin)
    once = widgets[3]
    thrice = widgets[5]

    baseline = area.verticalScrollBar().value()
    QApplication.sendEvent(once, _wheel(once))
    QApplication.processEvents()
    single_step = area.verticalScrollBar().value() - baseline
    assert single_step > 0

    install_wheel_blocker(thrice)  # already guarded by ValueSpinBox
    install_wheel_blocker_recursive(area)  # ...and again, via the tree walker

    baseline = area.verticalScrollBar().value()
    QApplication.sendEvent(thrice, _wheel(thrice))
    QApplication.processEvents()
    repeat_step = area.verticalScrollBar().value() - baseline

    assert repeat_step == single_step, "re-installing the guard double-scrolled"
    area.close()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: (lambda w: (w.setValue(50), w)[-1])(IntegerValueSpinBox(minimum=0, maximum=1000)),
        lambda: ValueComboBox(items=[1, 2, 3, 4, 5], value=3),
    ],
    ids=["IntegerValueSpinBox", "ValueComboBox"],
)
def test_all_value_widgets_are_guarded(qapp, factory):
    area, widgets = _scroll_area(factory)
    w = widgets[0]
    value_before = w.value()
    scroll_before = area.verticalScrollBar().value()

    QApplication.sendEvent(w, _wheel(w))
    QApplication.processEvents()

    assert w.value() == value_before
    assert area.verticalScrollBar().value() > scroll_before
    area.close()


def test_recursive_guard_protects_raw_widgets(qapp):
    """Covers forms built elsewhere, e.g. generated Qt Designer code.

    Those declare plain QSpinBox/QComboBox that we cannot swap for Value* classes,
    so the consuming widget calls install_wheel_blocker_recursive() after setupUi().
    """
    area, widgets = _scroll_area(
        lambda: (lambda w: (w.setRange(0, 1000), w.setValue(50), w)[-1])(QDoubleSpinBox())
    )
    install_wheel_blocker_recursive(area)

    w = widgets[3]
    value_before = w.value()
    scroll_before = area.verticalScrollBar().value()

    QApplication.sendEvent(w, _wheel(w))
    QApplication.processEvents()

    assert w.value() == value_before, "recursive guard must stop the value changing"
    assert area.verticalScrollBar().value() > scroll_before, "panel must still scroll"
    area.close()


def test_guard_outside_scroll_area_still_blocks(qapp):
    """With no scrolling ancestor there is nothing to forward to; still must not change."""
    w = _spin()
    w.show()
    QApplication.processEvents()
    before = w.value()

    QApplication.sendEvent(w, _wheel(w))
    QApplication.processEvents()

    assert w.value() == before
    w.close()


class TestValueComboBoxPopulation:
    """Dynamic population, so combos filled after construction can still use the class."""

    def test_constructs_empty_and_adds(self, qapp):
        combo = ValueComboBox(unit="A", decimals=1)
        assert combo.count() == 0

        combo.add_values([1e-12, 60e-12, 1e-9])
        assert combo.count() == 3
        assert combo.itemText(1) == "60.0 pA"

        combo.add_value(5e-9)
        assert combo.count() == 4

    def test_set_values_keeps_selection(self, qapp):
        combo = ValueComboBox(items=[1e-12, 60e-12, 1e-9], value=60e-12)
        assert combo.value() == 60e-12

        combo.set_values([1e-12, 60e-12, 1e-9, 5e-9])
        assert combo.value() == 60e-12, "repopulating must not silently change selection"
        assert combo.count() == 4

    def test_set_values_falls_back_when_selection_gone(self, qapp):
        combo = ValueComboBox(items=[1e-12, 60e-12], value=60e-12)
        combo.set_values([7e-9, 8e-9])
        assert combo.value() == 7e-9, "closest-match fallback expected"

    def test_set_values_explicit_value_wins(self, qapp):
        combo = ValueComboBox(items=[1, 2, 3], value=2)
        combo.set_values([1, 2, 3], value=3)
        assert combo.value() == 3

    def test_text_api_still_works(self, qapp):
        """ValueComboBox subclasses QComboBox, so text-based call sites keep working."""
        combo = ValueComboBox()
        combo.addItems(["a", "b"])
        combo.setCurrentIndex(1)
        assert combo.currentText() == "b"

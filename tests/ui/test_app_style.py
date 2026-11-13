"""On Qt 6 the app style indents spin box text, which Qt 6 sets against the frame."""

import pytest

pytest.importorskip("qtpy.QtWidgets")

from qtpy.QtWidgets import QDoubleSpinBox, QStyleFactory

from fibsem.ui.qt.style import SPINBOX_TEXT_INDENT, _Qt6Fusion


def test_a_polished_spin_box_indents_its_text(qapp):
    style = _Qt6Fusion(QStyleFactory.create("Fusion"))
    spin = QDoubleSpinBox()
    style.polish(spin)
    assert spin.lineEdit().textMargins().left() == SPINBOX_TEXT_INDENT

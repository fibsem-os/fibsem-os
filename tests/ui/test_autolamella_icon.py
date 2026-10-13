"""The AutoLamella application icon renders at every size, crisp at 16 px."""

import pytest

pytest.importorskip("PyQt5")  # CI installs .[test] only; the UI extra is deliberate

from PyQt5.QtCore import QSize
from PyQt5.QtGui import QColor

from fibsem.ui.icon import APP_ICON_SIZES, autolamella_icon

WHITE = QColor("#FFFFFF").rgb()


def test_icon_carries_every_size(qapp):
    icon = autolamella_icon()

    assert sorted(s.width() for s in icon.availableSizes()) == list(APP_ICON_SIZES)


def test_lamella_is_one_solid_white_row_at_16px(qapp):
    # The 32-unit drawing would put the 1 px lamella across rows 7 and 8 as a
    # pale band; the 16 px drawing puts it on row 7 alone, pure white.
    image = autolamella_icon().pixmap(QSize(16, 16)).toImage()

    white_rows = [y for y in range(16) if image.pixel(8, y) == WHITE]
    assert white_rows == [7]
    assert all(image.pixel(x, 7) == WHITE for x in range(2, 14))

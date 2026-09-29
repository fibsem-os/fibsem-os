"""What a TitledPanel does with the widgets handed to its header.

The header is a fixed 24px. A widget taller than that is capped to fit; a widget
that sized itself smaller keeps its size and sits centred. The second half used to
be wrong: `add_header_widget` raised every widget's maximum to 24px and the layout
stretched it, so a 14px chip filled the header and touched both edges.
"""

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QPoint  # noqa: E402
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton  # noqa: E402

from fibsem.ui.widgets.custom_widgets import TitledPanel, header_chip  # noqa: E402


@pytest.fixture
def qapp():
    yield QApplication.instance() or QApplication([])


def _panel_with(widget):
    panel = TitledPanel("Panel", content=QLabel("body"))
    panel.add_header_widget(widget)
    panel.resize(300, 100)
    panel.show()
    QApplication.processEvents()
    return panel


def test_a_small_header_widget_keeps_its_height_and_is_centred(qapp):
    chip = header_chip("on", "#50a6ff")
    panel = _panel_with(chip)

    assert chip.height() == 14
    top = chip.mapTo(panel._header, QPoint(0, 0)).y()
    assert top == (panel._header.height() - chip.height()) // 2


def test_a_tall_header_widget_is_capped_to_the_header(qapp):
    """A stock push button wants ~30px under the app stylesheet; the header
    caps it at 24px. Without the stylesheet it is shorter, so assert the cap and
    the centring rather than an exact height."""
    button = QPushButton("Advanced")
    button.setMinimumHeight(30)  # what the app stylesheet would give it
    panel = _panel_with(button)

    assert panel._header.height() == 24
    assert button.height() <= 24
    assert button.maximumHeight() == 24

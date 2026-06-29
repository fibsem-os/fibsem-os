"""Quad-view microscope display — reusable 2x2 grid of FibsemImageCanvas.

Replaces the single napari viewer in the main microscope tab. Four cells:

    SEM (electron) | FIB (ion)
    FM (fluorescence) | "No Data" placeholder

``MicroscopeViewController`` wraps the widget and is the object handed to the
control widgets in place of the napari ``Viewer``. Its surface is intentionally
small in Phase 0 (image routing only) and grows in later phases (overlays,
per-beam click signals).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtWidgets import QFrame, QLabel, QSplitter, QVBoxLayout, QWidget

from fibsem.structures import BeamType, FibsemImage
from fibsem.ui.widgets.fm_canvas import FMCanvasWidget
from fibsem.ui.widgets.image_canvas import FibsemImageCanvas

if TYPE_CHECKING:
    from fibsem.fm.structures import FluorescenceImage

_BG = "#1e2124"
_TITLE_STYLE = (
    "color: #888; font-size: 11px; padding: 2px 6px; background: #1e2124;"
)
_PLACEHOLDER_STYLE = "color: #777; font-size: 12px;"


def _titled(title: str, inner: QWidget) -> QWidget:
    """Wrap *inner* with a small title label above it."""
    w = QWidget()
    w.setStyleSheet(f"background: {_BG};")
    lbl = QLabel(title, alignment=Qt.AlignLeft)
    lbl.setStyleSheet(_TITLE_STYLE)
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    lay.addWidget(lbl)
    lay.addWidget(inner)
    return w


def _splitter(orientation, *widgets) -> QSplitter:
    s = QSplitter(orientation)
    s.setChildrenCollapsible(False)
    for w in widgets:
        s.addWidget(w)
    s.setSizes([1000] * len(widgets))
    return s


class PlaceholderPanel(QFrame):
    """Inert 'No Data' panel for the 4th quad-view cell (no canvas, no toolbar)."""

    def __init__(self, text: str = "No Data") -> None:
        super().__init__()
        self.setStyleSheet(f"background: {_BG};")
        lbl = QLabel(text, alignment=Qt.AlignCenter)
        lbl.setStyleSheet(_PLACEHOLDER_STYLE)
        lay = QVBoxLayout(self)
        lay.addWidget(lbl)


class QuadViewWidget(QWidget):
    """2x2 grid: SEM | FIB over FM | placeholder, each a titled panel.

    The SEM/FIB cells are :class:`FibsemImageCanvas` instances (so they inherit
    the reset / scalebar / crosshair / contrast toolbar); the FM cell is an
    :class:`FMCanvasWidget` (multi-channel composite + per-channel controls), and
    the 4th is an inert placeholder. ``fm_canvas`` aliases the FM widget's inner
    canvas so overlays attach the same way they do on SEM/FIB.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.sem_canvas = FibsemImageCanvas()
        self.fib_canvas = FibsemImageCanvas()
        self.fm_widget = FMCanvasWidget()
        self.fm_canvas = self.fm_widget.canvas
        self.placeholder = PlaceholderPanel("No Data")

        left = _splitter(
            Qt.Vertical, _titled("SEM", self.sem_canvas), _titled("FM", self.fm_widget)
        )
        right = _splitter(
            Qt.Vertical,
            _titled("FIB", self.fib_canvas),
            _titled("Placeholder", self.placeholder),
        )
        root = _splitter(Qt.Horizontal, left, right)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(root)


class MicroscopeViewController(QObject):
    """Seam object that replaces the napari ``Viewer`` in the main microscope tab.

    Holds the :class:`QuadViewWidget` and routes images to per-beam canvases.
    Phase 0 surface is image routing only; later phases add overlay management
    and re-expose per-canvas click signals.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._widget = QuadViewWidget()
        self._canvases: Dict[BeamType, FibsemImageCanvas] = {
            BeamType.ELECTRON: self._widget.sem_canvas,
            BeamType.ION: self._widget.fib_canvas,
        }

    @property
    def widget(self) -> QuadViewWidget:
        return self._widget

    @property
    def sem_canvas(self) -> FibsemImageCanvas:
        return self._widget.sem_canvas

    @property
    def fib_canvas(self) -> FibsemImageCanvas:
        return self._widget.fib_canvas

    @property
    def fm_widget(self) -> FMCanvasWidget:
        """The multi-channel FM widget (composite + per-channel controls)."""
        return self._widget.fm_widget

    @property
    def fm_canvas(self) -> FibsemImageCanvas:
        """The FM widget's inner canvas — for overlays / scalebar, like SEM/FIB."""
        return self._widget.fm_canvas

    def get_canvas(self, beam: BeamType) -> Optional[FibsemImageCanvas]:
        """Return the canvas for a charged-particle beam (ELECTRON / ION)."""
        return self._canvases.get(beam)

    def set_image(self, beam: BeamType, image: FibsemImage) -> None:
        """Display *image* on the canvas for *beam* (no-op for unknown beams)."""
        canvas = self._canvases.get(beam)
        if canvas is not None:
            canvas.set_image(image)

    def set_fm_channel(self, name: str, data, color: Optional[str] = None) -> None:
        """Upsert one fluorescence channel into the FM composite (by *name*)."""
        self._widget.fm_widget.set_channel(name, data, color)

    def set_fm_pixel_size(self, pixel_size: Optional[float]) -> None:
        """Set the FM canvas pixel size (metres/px) for the scalebar."""
        self._widget.fm_widget.set_pixel_size(pixel_size)

    def set_fm_image(self, image: "FluorescenceImage") -> None:
        """Composite an acquired ``FluorescenceImage`` (all channels) onto the FM
        canvas. See :meth:`FMCanvasWidget.set_fm_image`."""
        self._widget.fm_widget.set_fm_image(image)

    def clear(self) -> None:
        """Clear all image canvases back to their placeholder text."""
        self.sem_canvas.clear()
        self.fib_canvas.clear()
        self._widget.fm_widget.clear()

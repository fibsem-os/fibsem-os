"""A picker for the FM's emission filters, shown by name and band.

The items are today's emission values (``None``, a label, or a band's bottom edge in
nm), so ``ChannelSettings.emission_wavelength`` and saved channels keep their values.
Each item shows the filter that value names: its name, its band drawn on a 400 to
700 nm strip, and the band's edges and centre in the tooltip.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple, Union

from PyQt5.QtCore import QPointF, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QStyledItemDelegate

from fibsem.fm.structures import REFLECTION, EmissionFilter, emission_filter_for
from fibsem.ui.widgets.custom_widgets import ValueComboBox

EmissionValue = Union[None, float, str]
FilterLookup = Callable[[EmissionValue], EmissionFilter]

STRIP_NM: Tuple[float, float] = (400.0, 700.0)
_STRIP_SIZE = QSize(36, 10)
_SWATCH_SIZE = QSize(10, 10)
_STRIP_COLOR = QColor("#555a66")
_UNKNOWN_BANDS_COLOR = QColor("#9aa0a8")


def plain_emission_filter(value: EmissionValue) -> EmissionFilter:
    """The filter a value names when no filter set is connected: no known bands."""
    return emission_filter_for(value, {})


def emission_lookup_for(fm: Any) -> Optional[FilterLookup]:
    """Name emission values through the FM's filter set, which knows their bands. A
    filter set that doesn't subclass ``FilterSet`` gets the plain names."""
    filter_set = getattr(fm, "filter_set", None)
    return getattr(filter_set, "emission_filter", None)


def band_color(nm: float) -> QColor:
    """An approximate display colour for light of wavelength ``nm``."""
    stops = (
        (400.0, (130, 0, 200)),
        (440.0, (60, 80, 255)),
        (490.0, (0, 200, 255)),
        (520.0, (60, 200, 60)),
        (570.0, (230, 220, 0)),
        (600.0, (255, 140, 0)),
        (650.0, (240, 40, 20)),
        (700.0, (180, 0, 0)),
    )
    nm = min(max(nm, stops[0][0]), stops[-1][0])
    for (lo, lo_rgb), (hi, hi_rgb) in zip(stops, stops[1:]):
        if nm <= hi:
            t = (nm - lo) / (hi - lo)
            return QColor(*(round(a + (b - a) * t) for a, b in zip(lo_rgb, hi_rgb)))
    return QColor(*stops[-1][1])


def _paint_stripes(painter: QPainter, rect: QRectF) -> None:
    """Diagonal stripes: several bands, none of them known."""
    painter.save()
    painter.setClipRect(rect)
    painter.setPen(QPen(_UNKNOWN_BANDS_COLOR, 2))
    step = 5.0
    x = rect.left() - rect.height()
    while x < rect.right():
        painter.drawLine(
            QPointF(x, rect.bottom()), QPointF(x + rect.height(), rect.top())
        )
        x += step
    painter.restore()


def _band_strip(emission_filter: EmissionFilter) -> QPixmap:
    """The filter's bands on a 400 to 700 nm strip; stripes for a multi-band filter
    whose bands aren't known; an empty strip for reflection."""
    pixmap = QPixmap(_STRIP_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(_STRIP_COLOR)
        width, height = _STRIP_SIZE.width(), _STRIP_SIZE.height()
        painter.drawRoundedRect(QRectF(0, 0, width, height), 2, 2)
        if emission_filter.multi_band and not emission_filter.bands:
            _paint_stripes(painter, QRectF(0, 0, width, height))
        start, end = STRIP_NM
        for low, high in emission_filter.bands:
            # A driver that knows only the bottom edge gets a thin mark there.
            high = high if high is not None else low + 5
            x0 = width * (max(low, start) - start) / (end - start)
            x1 = width * (min(high, end) - start) / (end - start)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(band_color((low + high) / 2))
            painter.drawRect(QRectF(x0, 0, max(x1 - x0, 2.0), height))
    finally:
        painter.end()
    return pixmap


def _band_swatch(emission_filter: EmissionFilter) -> QPixmap:
    """A square split between the bands' colours; stripes for unknown bands;
    transparent for reflection."""
    pixmap = QPixmap(_SWATCH_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    bands = emission_filter.bands
    if not bands and not emission_filter.multi_band:
        return pixmap
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = float(_SWATCH_SIZE.width())
        painter.setClipPath(_rounded(QRectF(0, 0, size, size)))
        if not bands:
            painter.fillRect(QRectF(0, 0, size, size), _STRIP_COLOR)
            _paint_stripes(painter, QRectF(0, 0, size, size))
        for i, (low, high) in enumerate(bands):
            high = high if high is not None else low
            part = size / len(bands)
            painter.fillRect(
                QRectF(i * part, 0, part, size), band_color((low + high) / 2)
            )
    finally:
        painter.end()
    return pixmap


def _rounded(rect: QRectF) -> QPainterPath:
    path = QPainterPath()
    path.addRoundedRect(rect, 2, 2)
    return path


def band_icon(emission_filter: EmissionFilter) -> QIcon:
    """The band as a strip in the open list and as a colour swatch in the closed box:
    Qt picks whichever pixmap matches the size it asks for."""
    icon = QIcon(_band_strip(emission_filter))
    icon.addPixmap(_band_swatch(emission_filter))
    return icon


def band_tooltip(emission_filter: EmissionFilter) -> str:
    """The filter's name and what it passes."""
    label = emission_filter.label
    if emission_filter.centre is not None:
        return (
            f"{label}: band {emission_filter.low:.0f} to {emission_filter.high:.0f} "
            f"nm, centre {emission_filter.centre:.0f} nm"
        )
    if len(emission_filter.bands) > 1:
        return f"{label}: {len(emission_filter.bands)} bands"
    if emission_filter.bands:
        return f"{label}: band from {emission_filter.low:.0f} nm"
    if emission_filter.multi_band:
        return f"{label}: several bands, not reported by the microscope"
    if emission_filter == REFLECTION:
        return "Reflection: no emission filter"
    return label


class EmissionFilterComboBox(ValueComboBox):
    """Picks an emission value; shows the filter it names, by name and band."""

    def __init__(
        self,
        items: Optional[Sequence[EmissionValue]] = None,
        lookup: Optional[FilterLookup] = None,
        parent=None,
    ) -> None:
        self._lookup: FilterLookup = lookup or plain_emission_filter
        super().__init__(
            items=list(items or []),
            format_fn=lambda value: self._lookup(value).label,
            parent=parent,
        )
        self.setIconSize(_SWATCH_SIZE)
        # The default popup delegate sizes icons by the combo box; this one asks the
        # list for its own size, so the open list shows each band as a strip.
        self.setItemDelegate(QStyledItemDelegate(self))
        self.view().setIconSize(_STRIP_SIZE)
        self.view().setStyleSheet("QListView::item { min-height: 24px; }")

    def set_lookup(self, lookup: Optional[FilterLookup]) -> None:
        """Name the items through another filter set; keeps the selection."""
        self._lookup = lookup or plain_emission_filter
        items = [self.itemData(i) for i in range(self.count())]
        self.set_values(items)

    def emission_filter(self) -> EmissionFilter:
        """The filter the current item names."""
        return self._lookup(self.value())

    def add_value(self, item) -> None:
        emission_filter = self._lookup(item)
        self.addItem(band_icon(emission_filter), emission_filter.label, item)
        tooltip = band_tooltip(emission_filter)
        self.setItemData(self.count() - 1, tooltip, Qt.ItemDataRole.ToolTipRole)

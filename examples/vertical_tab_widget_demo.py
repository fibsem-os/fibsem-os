"""Demo: a modern vertical tab widget that shows icons only, and expands on hover.

Run it::

    python examples/vertical_tab_widget_demo.py            # rail pushes the content across
    python examples/vertical_tab_widget_demo.py --overlay  # rail floats over the content
    python examples/vertical_tab_widget_demo.py --dual     # + a contextual rail on the right

The rail sits collapsed at ``COLLAPSED_WIDTH`` showing just icons. Hovering
anywhere over it animates it out to ``EXPANDED_WIDTH``, fading the labels in on
the back half of the animation. The icons never move: they are painted at a
fixed x derived from the collapsed width, so expanding only ever reveals text
alongside them.

A rail can sit on either edge (``side=Qt.LeftEdge`` / ``Qt.RightEdge``). A right
rail is a true mirror, not a rail shoved rightwards: it grows leftwards, keeps
its icon column pinned to the window edge, right-aligns its labels and puts the
selection bar on the far right.

Everything is drawn in ``_TabButton.paintEvent`` rather than assembled from
QToolButton + QLabel. That buys three things worth having:
  * no ellipsised text ("Millin…") while the rail is mid-animation - the label is
    simply clipped by the widget edge and faded;
  * the icon can be pinned to a fixed x independent of the widget width, on
    either edge;
  * the selection chip and accent bar are one paint pass, no stylesheet juggling.

The constants immediately below are the knobs - widths, timings, palette.
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional, Sequence, Tuple

from qtpy.QtCore import (
    Property,
    QEasingCurve,
    QPropertyAnimation,
    QRect,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from qtpy.QtGui import QColor, QCursor, QFont, QIcon, QPainter
from qtpy.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

try:  # keep the demo runnable outside a fibsem checkout
    from fibsem.ui.icon import fibsem_icon
except ImportError:  # pragma: no cover - convenience fallback
    import qtawesome as qta

    def fibsem_icon(key: str, color: Optional[str] = None, **kwargs) -> QIcon:
        name = key.split(":", 1)[1] if ":" in key else key
        if color is not None:
            kwargs["color"] = color
        return qta.icon("mdi6.{}".format(name), **kwargs)


# ── Geometry / timing ────────────────────────────────────────────────────────
COLLAPSED_WIDTH = 56
EXPANDED_WIDTH = 208
ITEM_HEIGHT = 44
ICON_SIZE = 22
ANIM_DURATION = 180   # ms, width animation
EXPAND_DELAY = 100    # ms of hover before expanding (kills flicker on a fly-by)
COLLAPSE_DELAY = 220  # ms of grace before collapsing again

# ── Palette (napari-dark leaning) ────────────────────────────────────────────
COLOR_RAIL_BG = "#22252b"
COLOR_CONTENT_BG = "#2b2f37"
COLOR_CARD_BG = "#23262d"
COLOR_BORDER = "#181a1f"
COLOR_TEXT = "#e4e7ec"
COLOR_TEXT_MUTED = "#8b929c"
COLOR_ACCENT = "#68a0dd"
COLOR_HOVER = "#31353e"


def _label_opacity(width: int) -> float:
    """Fade the labels in over the back half of the expansion, not linearly.

    Text that starts fading the instant the rail twitches looks smeared; holding
    it at zero until the rail is a third of the way out reads as a reveal.
    """
    span = EXPANDED_WIDTH - COLLAPSED_WIDTH
    t = (width - COLLAPSED_WIDTH) / span if span > 0 else 1.0
    return max(0.0, min(1.0, (t - 0.35) / 0.5))


def _label_rect(width: int, height: int, mirrored: bool) -> Tuple[QRect, int]:
    """Where the label goes, and how it aligns, for a row of the given width.

    The label always hugs the icon column and is clipped away from it, so on a
    left rail it grows rightwards and on a right rail leftwards. Both leave the
    same 13px gutter between glyph and text.
    """
    if mirrored:
        right = width - COLLAPSED_WIDTH + 4
        return QRect(10, 0, max(0, right - 10), height), Qt.AlignRight | Qt.AlignVCenter
    left = COLLAPSED_WIDTH - 4
    return QRect(left, 0, max(0, width - left - 10), height), Qt.AlignLeft | Qt.AlignVCenter


def _icon_x(width: int, mirrored: bool) -> int:
    """Fixed icon column - anchored to whichever edge the rail lives on."""
    inset = (COLLAPSED_WIDTH - ICON_SIZE) // 2
    return width - COLLAPSED_WIDTH + inset if mirrored else inset


class _TabButton(QAbstractButton):
    """One rail entry: fixed-position icon, label that fades in as the rail opens."""

    def __init__(
        self,
        icon_key: str,
        text: str,
        side: int = Qt.LeftEdge,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(ITEM_HEIGHT)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.TabFocus)
        self._mirrored = side == Qt.RightEdge
        self._icon_idle = fibsem_icon(icon_key, color=COLOR_TEXT_MUTED)
        self._icon_active = fibsem_icon(icon_key, color=COLOR_ACCENT)
        self._hover = False

    # QAbstractButton would size itself from the text; we want the rail to drive width.
    def sizeHint(self) -> QSize:
        return QSize(EXPANDED_WIDTH, ITEM_HEIGHT)

    def minimumSizeHint(self) -> QSize:
        return QSize(0, ITEM_HEIGHT)

    def enterEvent(self, event) -> None:
        self._hover = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self.update()
        super().leaveEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Only useful while collapsed - once the label is legible the tooltip is noise.
        self.setToolTip("" if _label_opacity(self.width()) > 0.9 else self.text())

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()
        checked = self.isChecked()

        chip = QRect(6, 4, max(0, width - 12), height - 8)
        painter.setPen(Qt.NoPen)
        if checked:
            fill = QColor(COLOR_ACCENT)
            fill.setAlpha(48)
            painter.setBrush(fill)
            painter.drawRoundedRect(chip, 8, 8)
        elif self._hover:
            painter.setBrush(QColor(COLOR_HOVER))
            painter.drawRoundedRect(chip, 8, 8)

        if checked:  # accent bar hard against the rail's outer edge
            bar_x = width - 3 if self._mirrored else 0
            painter.setBrush(QColor(COLOR_ACCENT))
            painter.drawRoundedRect(QRect(bar_x, height // 2 - 9, 3, 18), 2, 2)

        icon = self._icon_active if checked else self._icon_idle
        icon_x = _icon_x(width, self._mirrored)
        icon.paint(painter, QRect(icon_x, (height - ICON_SIZE) // 2, ICON_SIZE, ICON_SIZE))

        opacity = _label_opacity(width)
        if opacity > 0.01:
            painter.setOpacity(opacity)
            painter.setPen(QColor(COLOR_TEXT if checked else COLOR_TEXT_MUTED))
            font = painter.font()
            font.setWeight(QFont.DemiBold if checked else QFont.Normal)
            painter.setFont(font)
            rect, align = _label_rect(width, height, self._mirrored)
            painter.drawText(rect, align, self.text())


class _RailHeader(QWidget):
    """Top row of a rail: an optional brand badge plus a label that fades in.

    With ``badge=False`` it is a pure caption - used on the right rail to name
    the section whose tools are currently listed. It still reserves its height
    while collapsed so both rails' rows line up.
    """

    def __init__(
        self,
        title: str,
        side: int = Qt.LeftEdge,
        badge: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._title = title
        self._badge = badge
        self._mirrored = side == Qt.RightEdge
        self.setFixedHeight(52)

    def setTitle(self, title: str) -> None:
        self._title = title
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        width, height = self.width(), self.height()

        if self._badge:
            badge_x = _icon_x(width, self._mirrored) - 2
            badge = QRect(badge_x, (height - 26) // 2, 26, 26)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(COLOR_ACCENT))
            painter.drawRoundedRect(badge, 7, 7)
            painter.setPen(QColor("#12141a"))
            font = painter.font()
            font.setWeight(QFont.Bold)
            painter.setFont(font)
            painter.drawText(badge, Qt.AlignCenter, self._title[:1].upper())

        opacity = _label_opacity(width)
        if opacity > 0.01:
            painter.setOpacity(opacity)
            font = painter.font()
            if self._badge:
                painter.setPen(QColor(COLOR_TEXT))
                font.setWeight(QFont.DemiBold)
            else:
                painter.setPen(QColor(COLOR_TEXT_MUTED))
                font.setPointSize(max(1, font.pointSize() - 1))
            painter.setFont(font)
            rect, align = _label_rect(width, height, self._mirrored)
            painter.drawText(rect, align, self._title)


class VerticalTabBar(QWidget):
    """The rail itself: collapsed icon strip that animates open while hovered."""

    currentChanged = Signal(int)
    railWidthChanged = Signal(int)

    def __init__(
        self,
        title: str = "fibsem",
        side: int = Qt.LeftEdge,
        badge: bool = True,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.side = side
        self._mirrored = side == Qt.RightEdge
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet(
            "VerticalTabBar {{ background-color: {}; {}: 1px solid {}; }}".format(
                COLOR_RAIL_BG, "border-left" if self._mirrored else "border-right", COLOR_BORDER
            )
        )
        self.setFixedWidth(COLLAPSED_WIDTH)

        self._buttons: List[_TabButton] = []
        self._current = -1
        self._pinned = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(2)
        self.header = _RailHeader(title, side=side, badge=badge)
        layout.addWidget(self.header)
        layout.addSpacing(6)
        self._items_layout = QVBoxLayout()
        self._items_layout.setContentsMargins(0, 0, 0, 0)
        self._items_layout.setSpacing(2)
        layout.addLayout(self._items_layout)
        layout.addStretch(1)

        # Pin lives outside the exclusive group - it toggles "stay open", not a page.
        self._pin = _TabButton("mdi:pin-outline", "Keep expanded", side=side)
        self._pin.toggled.connect(self.set_pinned)
        layout.addWidget(self._pin)

        self._anim = QPropertyAnimation(self, b"railWidth", self)
        self._anim.setDuration(ANIM_DURATION)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)

        self._expand_timer = QTimer(self)
        self._expand_timer.setSingleShot(True)
        self._expand_timer.timeout.connect(lambda: self._animate_to(EXPANDED_WIDTH))
        self._collapse_timer = QTimer(self)
        self._collapse_timer.setSingleShot(True)
        self._collapse_timer.timeout.connect(self._maybe_collapse)

    # ── tabs ────────────────────────────────────────────────────────────────
    def addTab(self, icon_key: str, text: str) -> int:
        index = len(self._buttons)
        button = _TabButton(icon_key, text, side=self.side)
        button.clicked.connect(lambda _checked=False, i=index: self.setCurrentIndex(i))
        self._items_layout.addWidget(button)
        self._buttons.append(button)
        if self._current < 0:
            self.setCurrentIndex(0)
        return index

    def clear(self) -> None:
        for button in self._buttons:
            self._items_layout.removeWidget(button)
            button.setParent(None)  # removeWidget alone leaves it drawn as a child
            button.deleteLater()
        self._buttons = []
        self._current = -1

    def setTabs(self, tabs: Sequence[Tuple[str, str]]) -> None:
        """Swap the whole tab set out. Selects the first entry and re-emits."""
        self.clear()
        for icon_key, text in tabs:
            self.addTab(icon_key, text)

    def setCurrentIndex(self, index: int) -> None:
        if not 0 <= index < len(self._buttons):
            return
        for i, button in enumerate(self._buttons):
            button.setChecked(i == index)
        if index != self._current:
            self._current = index
            self.currentChanged.emit(index)

    def currentIndex(self) -> int:
        return self._current

    # ── expand / collapse ───────────────────────────────────────────────────
    def set_pinned(self, pinned: bool) -> None:
        self._pinned = pinned
        self._pin.setChecked(pinned)
        self._expand_timer.stop()
        self._collapse_timer.stop()
        self._animate_to(EXPANDED_WIDTH if pinned else COLLAPSED_WIDTH)

    def enterEvent(self, event) -> None:
        self._collapse_timer.stop()
        if not self._pinned:
            self._expand_timer.start(EXPAND_DELAY)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._expand_timer.stop()
        if not self._pinned:
            self._collapse_timer.start(COLLAPSE_DELAY)
        super().leaveEvent(event)

    def _maybe_collapse(self) -> None:
        # Qt hands the rail a Leave whenever the pointer exits its rect, including
        # some child-widget transitions - re-check where the cursor actually is.
        if self._pinned or self.rect().contains(self.mapFromGlobal(QCursor.pos())):
            return
        self._animate_to(COLLAPSED_WIDTH)

    def _animate_to(self, target: int) -> None:
        if self.width() == target:
            return
        self._anim.stop()
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(target)
        self._anim.start()

    def _get_rail_width(self) -> int:
        return self.width()

    def _set_rail_width(self, width: int) -> None:
        self.setFixedWidth(int(width))
        self.railWidthChanged.emit(int(width))

    railWidth = Property(int, _get_rail_width, _set_rail_width)


def _float_rail(host: QWidget, bar: VerticalTabBar) -> QWidget:
    """Lift ``bar`` out of the layout so it floats over the content.

    Returns the fixed-width gutter that holds its collapsed slot. The caller is
    responsible for calling :func:`_place_rail` on resize and width change.
    """
    gutter = QWidget()
    gutter.setFixedWidth(COLLAPSED_WIDTH)
    bar.setParent(host)
    shadow = QGraphicsDropShadowEffect(bar)
    shadow.setBlurRadius(24)
    shadow.setXOffset(-6 if bar.side == Qt.RightEdge else 6)
    shadow.setYOffset(0)
    shadow.setColor(QColor(0, 0, 0, 140))
    bar.setGraphicsEffect(shadow)
    return gutter


def _place_rail(host: QWidget, bar: VerticalTabBar) -> None:
    x = host.width() - bar.width() if bar.side == Qt.RightEdge else 0
    bar.setGeometry(x, 0, bar.width(), host.height())
    bar.raise_()


class VerticalTabWidget(QWidget):
    """QTabWidget-ish container: hover rail on one edge, stacked pages beside it.

    ``overlay=True`` floats the rail above the pages (the expansion covers content
    instead of shoving it), which keeps the page layout perfectly still.
    """

    currentChanged = Signal(int)

    def __init__(
        self,
        title: str = "fibsem",
        overlay: bool = False,
        side: int = Qt.LeftEdge,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._overlay = overlay
        self._bar = VerticalTabBar(title, side=side)
        self._stack = QStackedWidget()
        self._bar.currentChanged.connect(self._stack.setCurrentIndex)
        self._bar.currentChanged.connect(self.currentChanged)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        rail_widget = _float_rail(self, self._bar) if overlay else self._bar
        if overlay:
            self._bar.railWidthChanged.connect(lambda _w: _place_rail(self, self._bar))
        if side == Qt.RightEdge:
            layout.addWidget(self._stack)
            layout.addWidget(rail_widget)
        else:
            layout.addWidget(rail_widget)
            layout.addWidget(self._stack)

    def addTab(self, page: QWidget, icon_key: str, text: str) -> int:
        index = self._stack.addWidget(page)
        self._bar.addTab(icon_key, text)
        return index

    def setCurrentIndex(self, index: int) -> None:
        self._bar.setCurrentIndex(index)

    def currentIndex(self) -> int:
        return self._bar.currentIndex()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._overlay:
            _place_rail(self, self._bar)


# ── demo ─────────────────────────────────────────────────────────────────────

# (icon, section, blurb, [(icon, tool, blurb), ...]) - the trailing list is what
# the right-hand rail swaps in when that section is selected.
TABS = [
    (
        "mdi:microscope", "Imaging", "Beam settings, detectors and live acquisition.",
        [
            ("mdi:tune-vertical", "Beam", "Voltage, current and working distance."),
            ("mdi:camera-iris", "Detector", "Detector type, brightness and contrast."),
            ("mdi:image-filter-center-focus", "Autofocus", "Focus sweep and beam alignment."),
        ],
    ),
    (
        "mdi:target", "Milling", "Pattern definition, stages and mill supervision.",
        [
            ("mdi:shape-rectangle-plus", "Patterns", "Trench, undercut and microexpansion shapes."),
            ("mdi:layers-triple", "Stages", "Ordering, currents and per-stage overrides."),
            ("mdi:eye-check-outline", "Supervision", "Pause points and operator confirmation."),
            ("mdi:crosshairs", "Alignment", "Drift correction between milling stages."),
        ],
    ),
    (
        "mdi:axis-arrow", "Movement", "Stage moves, compucentric rotation and tilts.",
        [
            ("mdi:stove", "Stage", "Absolute and relative stage positioning."),
            ("mdi:robot-industrial", "Manipulator", "Needle insert, retract and saved positions."),
            ("mdi:rotate-3d-variant", "Compucentric", "Rotation about the coincidence point."),
        ],
    ),
    (
        "mdi:image-multiple", "Correlation", "Fluorescence overlay and point picking.",
        [
            ("mdi:vector-point", "Points", "Pick and pair fiducials across modalities."),
            ("mdi:function-variant", "Transform", "Fit, residuals and RMS reporting."),
            ("mdi:layers-outline", "Overlay", "Channel blending and opacity."),
        ],
    ),
    (
        "mdi:cog-outline", "System", "Hardware connection, calibration and logs.",
        [
            ("mdi:lan-connect", "Connection", "Microscope address and handshake."),
            ("mdi:ruler-square", "Calibration", "Beam shift, stage and detector calibration."),
            ("mdi:text-box-outline", "Logs", "Session log stream and export."),
        ],
    ),
]


class _SectionPage(QWidget):
    """A primary page. The card at the bottom is driven by the right-hand rail."""

    def __init__(self, title: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background-color: {};".format(COLOR_CONTENT_BG))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 28, 32, 28)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setStyleSheet(
            "color: {}; font-size: 22px; font-weight: 600;".format(COLOR_TEXT)
        )
        caption = QLabel(subtitle)
        caption.setStyleSheet("color: {}; font-size: 13px;".format(COLOR_TEXT_MUTED))
        layout.addWidget(heading)
        layout.addWidget(caption)

        rule = QFrame()
        rule.setFrameShape(QFrame.HLine)
        rule.setStyleSheet("color: {};".format(COLOR_BORDER))
        layout.addWidget(rule)

        card = QFrame()
        card.setStyleSheet(
            "QFrame {{ background-color: {}; border: 1px solid {}; border-radius: 10px; }}".format(
                COLOR_CARD_BG, COLOR_BORDER
            )
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(6)
        eyebrow = QLabel("SELECTED TOOL")
        eyebrow.setStyleSheet(
            "color: {}; font-size: 10px; font-weight: 600; border: none;".format(COLOR_ACCENT)
        )
        self._tool_name = QLabel("-")
        self._tool_name.setStyleSheet(
            "color: {}; font-size: 16px; font-weight: 600; border: none;".format(COLOR_TEXT)
        )
        self._tool_blurb = QLabel("")
        self._tool_blurb.setStyleSheet(
            "color: {}; font-size: 12px; border: none;".format(COLOR_TEXT_MUTED)
        )
        card_layout.addWidget(eyebrow)
        card_layout.addWidget(self._tool_name)
        card_layout.addWidget(self._tool_blurb)
        layout.addWidget(card)

        hint = QLabel(
            "Hover either rail to expand it. The right rail lists tools for this\n"
            "section only - switch sections and its whole tab set is swapped out."
        )
        hint.setStyleSheet("color: {}; font-size: 12px;".format(COLOR_TEXT_MUTED))
        layout.addWidget(hint)
        layout.addStretch(1)

    def set_tool(self, name: str, blurb: str) -> None:
        self._tool_name.setText(name)
        self._tool_blurb.setText(blurb)


class DualRailDemo(QWidget):
    """Primary rail left, contextual rail right, pages in between.

    The right rail is rebuilt from scratch on every section change - see
    ``_on_section_changed``. That is the point of the demo: its tab set is not
    fixed, it is a function of the left rail's selection.
    """

    def __init__(self, overlay: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._overlay = overlay
        self._left = VerticalTabBar("fibsem", side=Qt.LeftEdge)
        self._right = VerticalTabBar("", side=Qt.RightEdge, badge=False)
        self._stack = QStackedWidget()

        for icon_key, name, blurb, _tools in TABS:
            self._stack.addWidget(_SectionPage(name, blurb))
            self._left.addTab(icon_key, name)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        if overlay:
            left_slot = _float_rail(self, self._left)
            right_slot = _float_rail(self, self._right)
            self._left.railWidthChanged.connect(lambda _w: _place_rail(self, self._left))
            self._right.railWidthChanged.connect(lambda _w: _place_rail(self, self._right))
        else:
            left_slot, right_slot = self._left, self._right
        layout.addWidget(left_slot)
        layout.addWidget(self._stack)
        layout.addWidget(right_slot)

        # Wire up only once both rails exist, then prime the initial selection by
        # hand - addTab already set the left rail to 0, so it will not re-emit.
        self._left.currentChanged.connect(self._on_section_changed)
        self._right.currentChanged.connect(self._on_tool_changed)
        self._on_section_changed(0)

    def _on_section_changed(self, index: int) -> None:
        self._stack.setCurrentIndex(index)
        _icon, name, _blurb, tools = TABS[index]
        self._right.header.setTitle("{} tools".format(name))
        # Emits currentChanged(0) -> _on_tool_changed fills in the card.
        self._right.setTabs([(key, label) for key, label, _ in tools])

    def _on_tool_changed(self, index: int) -> None:
        section = self._left.currentIndex()
        tools = TABS[section][3]
        if 0 <= index < len(tools):
            _key, name, blurb = tools[index]
            self._stack.widget(section).set_tool(name, blurb)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._overlay:
            _place_rail(self, self._left)
            _place_rail(self, self._right)


def build_window(overlay: bool = False, dual: bool = False) -> QWidget:
    window = QWidget()
    window.setWindowTitle(
        "Vertical tab widget - {}hover to expand".format("dual rail, " if dual else "")
    )
    window.resize(1040 if dual else 880, 560)
    window.setStyleSheet("background-color: {};".format(COLOR_CONTENT_BG))

    if dual:
        content = DualRailDemo(overlay=overlay)  # type: QWidget
    else:
        content = VerticalTabWidget(title="fibsem", overlay=overlay)
        for icon_key, name, blurb, tools in TABS:
            page = _SectionPage(name, blurb)
            page.set_tool(*tools[0][1:])  # no right rail here, so pin the first tool
            content.addTab(page, icon_key, name)

    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(content)
    return window


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="float the rail(s) over the content instead of pushing it across",
    )
    parser.add_argument(
        "--dual",
        action="store_true",
        help="add a right-hand rail whose tabs change with the selected section",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = build_window(overlay=args.overlay, dual=args.dual)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

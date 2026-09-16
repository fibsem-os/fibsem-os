from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt5.QtCore import QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from fibsem.applications.autolamella.structures import Lamella
from fibsem.structures import MicroscopeState
from fibsem.ui import stylesheets
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    NEUTRAL_550,
    SURFACE_COLOR,
    TEXT_COLOR,
)
from fibsem.ui.widgets.canvas.overlay_controls import CanvasOverlayControls
from fibsem.ui.widgets.custom_widgets import IconToolButton
from fibsem.ui.widgets.microscope_state_widget import MicroscopeStateWidget
from fibsem.utils import (
    NOT_AVAILABLE,
    format_current,
    format_distance,
    format_stage_position,
    format_voltage,
)

_NAME_WIDTH = 110
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 3 + 16  # 3 buttons + 2 gaps

# The pose the milling patterns belong to. They are a property of the lamella rather
# than of any pose, but the milling pose is where they would be cut, so that is the
# row the control sits on -- and the only one, so a lamella with a fluorescence pose
# does not grow a second button that does the same thing.
MILLING_POSE_NAME = "MILLING"

# (key, label, checked) per task, as CanvasOverlayControls takes them.
PatternEntry = Tuple[str, str, bool]

_POPUP_WIDTH = 400

# The position control reads as text until it is approached. Flat, transparent and in
# the same muted colour the label used, so a row at rest looks exactly as it did; the
# hover state is the whole of the affordance, which is why it has to be visible.
_POSITION_BUTTON_STYLE = f"""
QPushButton {{
    background: transparent;
    border: none;
    padding: 0px;
    text-align: left;
    color: {NEUTRAL_550};
}}
QPushButton:hover {{
    color: {TEXT_COLOR};
    text-decoration: underline;
}}
QPushButton:disabled {{
    color: {NEUTRAL_550};
    text-decoration: none;
}}
"""

# Preferred display order; poses not listed keep their insertion order after these.
_POSE_ORDER = ["MILLING", "FLUORESCENCE"]


class LamellaPoseRowWidget(QWidget):
    """A single pose row: name, position, update and move-to buttons.

    The position is a flat button rather than a label. A pose is a whole
    ``MicroscopeState`` -- both beams, both detectors, a timestamp -- and the row has
    space for one line of it, so the rest needs somewhere to go. Making the position
    itself the control avoids a third icon in a 40-pixel row that already carries two,
    and puts the affordance on the thing it describes.

    A button rather than a ``QLabel`` with a ``mousePressEvent``: it is focusable and
    in the tab order, which a label is not, and it brings hover and pressed states
    with it. Otherwise Details would be the one action in the row unreachable from the
    keyboard while Move To and Update stayed reachable.
    """

    update_clicked = pyqtSignal(str)  # pose name
    move_to_clicked = pyqtSignal(str)  # pose name
    pattern_overlays_changed = pyqtSignal(list)  # checked task names

    def __init__(
        self,
        pose_name: str,
        state: Optional[MicroscopeState],
        parent: Optional[QWidget] = None,
        pattern_entries: Sequence[PatternEntry] = (),
    ) -> None:
        super().__init__(parent)
        self.pose_name = pose_name
        self._state = state
        self._popup: Optional[_PoseDetailPopup] = None
        self._overlay_controls: Optional[CanvasOverlayControls] = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.name_label = QLabel(pose_name.capitalize())
        self.name_label.setFixedWidth(_NAME_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.position_button = QPushButton()
        self.position_button.setFlat(True)
        self.position_button.setCursor(Qt.PointingHandCursor)
        self.position_button.setStyleSheet(_POSITION_BUTTON_STYLE)
        self.position_button.clicked.connect(self._show_details)
        layout.addWidget(self.position_button, 1)

        self.btn_move_to = IconToolButton(
            icon=ICON_MOVE_TO_POSITION,
            tooltip="Move to Position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_move_to)

        self.btn_update = IconToolButton(
            icon=ICON_UPDATE_POSITION,
            tooltip="Update Position",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_update)

        self.btn_overlay: Optional[IconToolButton] = None
        if pose_name == MILLING_POSE_NAME and pattern_entries:
            self.btn_overlay = self._build_overlay_button(pattern_entries)
            layout.addWidget(self.btn_overlay)
        else:
            # Hold the column open. A hidden widget takes no space in a box layout, so
            # without this the rows without the button would sit their Move To and
            # Update one button-width right of the milling row's.
            spacer = QWidget()
            spacer.setFixedWidth(_BTN_SIZE.width())
            spacer.setStyleSheet("background: transparent;")
            layout.addWidget(spacer)

        self.btn_update.clicked.connect(
            lambda: self.update_clicked.emit(self.pose_name)
        )
        self.btn_move_to.clicked.connect(
            lambda: self.move_to_clicked.emit(self.pose_name)
        )

        self.set_state(state)

    def _build_overlay_button(self, entries: Sequence[PatternEntry]) -> IconToolButton:
        """The overlay button, and the popup of checkboxes behind it.

        A ``CanvasOverlayControls`` in a ``QWidgetAction`` rather than a menu of
        checkable ``QAction``s: it is the same control the Lamella tab opens from its
        canvas, so the two read as one interface, and it stays open while several
        steps are ticked -- a menu of checkable actions closes on the first click,
        which makes turning on three patterns three trips.
        """
        controls = CanvasOverlayControls(entries)
        controls.toggled.connect(self._on_overlay_toggled)

        menu = QMenu(self)
        action = QWidgetAction(menu)
        action.setDefaultWidget(controls)
        menu.addAction(action)

        button = IconToolButton(
            icon="mdi:eye-outline",
            tooltip="Overlay Pattern",
            size=_BTN_SIZE.width(),
        )
        button.setMenu(menu)
        button.setPopupMode(IconToolButton.InstantPopup)
        # Appended to the button's own stylesheet, not set over it: replacing it drops
        # the hover and pressed states along with the menu arrow this hides.
        button.setStyleSheet(
            stylesheets.ICON_TOOLBUTTON_STYLESHEET
            + "QToolButton::menu-indicator { image: none; }"
        )
        self._overlay_controls = controls
        return button

    def _on_overlay_toggled(self, _key: str, _checked: bool) -> None:
        """Report the whole selection, not the one box that changed.

        The canvas draws a set of patterns, so the set is what a consumer needs; the
        per-key signal would make every one of them reconstruct it.
        """
        self.pattern_overlays_changed.emit(self.checked_pattern_tasks())

    def checked_pattern_tasks(self) -> List[str]:
        """Task names currently ticked in the overlay popup."""
        controls = self._overlay_controls
        if controls is None:
            return []
        return [key for key in controls.keys() if controls.is_visible(key)]

    def set_state(self, state: Optional[MicroscopeState]) -> None:
        """Re-render the row from a pose."""
        self._state = state
        position = state.stage_position if state is not None else None
        self.position_button.setText(
            format_stage_position(position) if position is not None else "Unknown"
        )
        self.position_button.setToolTip(_summary(self.pose_name, state))
        # Nothing to open when there is no record behind the row.
        self.position_button.setEnabled(state is not None)
        if self._popup is not None and self._popup.isVisible():
            self._popup.set_state(self.pose_name, state)

    def _show_details(self) -> None:
        if self._state is None:
            return
        if self._popup is None:
            self._popup = _PoseDetailPopup(self)
        self._popup.set_state(self.pose_name, self._state)
        self._popup.show_under(self.position_button)


class _PoseDetailPopup(QFrame):
    """The full state, in a popup that closes when you click away.

    ``Qt.Popup`` rather than a dialog: it is non-modal and self-dismissing, so a pose
    can be read without losing the canvas behind it, and comparing two poses is two
    clicks rather than four.

    The stylesheet is scoped to the object name. An unscoped ``QFrame { border }``
    cascades onto every descendant, which draws a box around every value in the table
    inside.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent, Qt.Popup)
        self.setObjectName("PoseDetailPopup")
        self.setStyleSheet(
            f"QFrame#PoseDetailPopup {{ background: {SURFACE_COLOR};"
            f" border: 1px solid {BORDER_COLOR}; border-radius: 4px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(9, 9, 9, 9)
        self.state_widget = MicroscopeStateWidget()
        layout.addWidget(self.state_widget)
        self.setFixedWidth(_POPUP_WIDTH)

    def set_state(self, pose_name: str, state: Optional[MicroscopeState]) -> None:
        self.state_widget.set_state(state, title=pose_name)

    def show_under(self, anchor: QWidget) -> None:
        """Open below the control that summarises the same thing."""
        self.adjustSize()
        self.move(anchor.mapToGlobal(anchor.rect().bottomLeft()))
        self.show()


def _summary(pose_name: str, state: Optional[MicroscopeState]) -> str:
    """The tooltip: enough to answer "is this the pose I want" without a click."""
    if state is None:
        return "No position recorded"
    lines = [f"{pose_name} \u00b7 {format_stage_position(state.stage_position)}"]
    # SEM and FIB, matching the canvases and the popup this tooltip previews.
    for label, beam in (("SEM", state.electron_beam), ("FIB", state.ion_beam)):
        if beam is None:
            lines.append(f"{label}   {NOT_AVAILABLE}")
            continue
        lines.append(
            f"{label}   {format_voltage(beam.voltage)}"
            f" \u00b7 {format_current(beam.beam_current)}"
            f" \u00b7 {format_distance(beam.hfw)}"
        )
    return "\n".join(lines)


class _LamellaPoseListHeader(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        name_header = QLabel("Pose")
        name_header.setFixedWidth(_NAME_WIDTH)
        name_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(name_header)

        position_header = QLabel("Position")
        position_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(position_header, 1)

        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)


class LamellaPoseListWidget(QWidget):
    """List widget displaying a Lamella's poses with name, position and actions.

    Self-contained: holds only the pose names of the lamella set via
    :meth:`set_lamella`. Emits :attr:`update_requested` / :attr:`move_to_requested`
    with the pose name when the row buttons are clicked.
    """

    update_requested = pyqtSignal(str)  # pose name
    move_to_requested = pyqtSignal(str)  # pose name
    pattern_overlays_changed = pyqtSignal(list)  # checked task names

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # The user's choice of patterns, kept whole across lamella changes rather than
        # narrowed to whatever the current one happens to have: selecting a lamella
        # without Trench Milling and coming back should not have silently unticked it.
        # What is *emitted* is this narrowed to the current lamella -- see set_lamella.
        self._checked_pattern_tasks: List[str] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _LamellaPoseListHeader()
        layout.addWidget(self._header)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #3a3d42;")
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.setSpacing(0)
        self._list.setStyleSheet(stylesheets.LIST_WIDGET_STYLESHEET)
        self._list.setAlternatingRowColors(False)
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self._list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Rebuild the rows from the lamella's existing poses."""
        self._list.clear()
        entries = self._pattern_entries(lamella)
        if lamella is not None and lamella.poses:
            for pose_name in self._sorted_pose_names(lamella.poses):
                self._add_row(pose_name, lamella.poses[pose_name], entries)
        # The rows have just been replaced, so anything drawn from them describes the
        # previous lamella. Say what this one has ticked -- emitted even when that is
        # nothing, which is what clears the overlay on a lamella with no patterns.
        self.pattern_overlays_changed.emit(
            [key for key, _, checked in entries if checked]
        )

    def refresh_pose(self, pose_name: str, state: Optional[MicroscopeState]) -> None:
        """Update an existing pose row in place, from the record itself.

        Takes the ``MicroscopeState`` rather than a rendered string: the row now shows
        more of it than one line, and every caller already had the state in hand and
        was reaching into it for ``stage_position.pretty`` at the call site.

        Avoids rebuilding the list so row selection/scroll state is preserved. No-op if
        no row matches *pose_name*.
        """
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, LamellaPoseRowWidget) and row.pose_name == pose_name:
                row.set_state(state)
                return

    def clear(self) -> None:
        self._list.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _pattern_entries(self, lamella: Optional[Lamella]) -> List[PatternEntry]:
        """One entry per task of *lamella* that mills something, in protocol order.

        Read off the lamella rather than the experiment's workflow: this widget is
        handed a Lamella and nothing else, and its two hosts (the Experiment tab and
        the coincidence viewer) do not hold the same things besides. ``task_config``
        is seeded from the protocol when the lamella is added, so its order is the
        protocol's declaration order.
        """
        if lamella is None:
            return []
        return [
            (task_name, task_name, task_name in self._checked_pattern_tasks)
            for task_name, config in lamella.task_config.items()
            if getattr(config, "milling", None)
        ]

    def _on_row_overlays_changed(self, task_names: list) -> None:
        self._checked_pattern_tasks = list(task_names)
        self.pattern_overlays_changed.emit(list(task_names))

    @staticmethod
    def _sorted_pose_names(poses) -> list:
        """Order poses by _POSE_ORDER first; remaining keep insertion order after."""

        def key(name: str):
            try:
                return (0, _POSE_ORDER.index(name))
            except ValueError:
                return (1, 0)

        return sorted(poses.keys(), key=key)

    def _add_row(
        self,
        pose_name: str,
        state: Optional[MicroscopeState],
        pattern_entries: Sequence[PatternEntry] = (),
    ) -> LamellaPoseRowWidget:
        row = LamellaPoseRowWidget(pose_name, state, pattern_entries=pattern_entries)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        row.update_clicked.connect(self.update_requested)
        row.move_to_clicked.connect(self.move_to_requested)
        row.pattern_overlays_changed.connect(self._on_row_overlays_changed)
        return row

from __future__ import annotations

from typing import List, Optional

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
)

from fibsem.applications.autolamella.structures import Lamella, PoseProvenance
from fibsem.structures import MicroscopeState
from fibsem.ui import stylesheets
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION
from fibsem.ui.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    NEUTRAL_550,
    SEMANTIC_WARNING_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
)
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

ICON_DERIVE_POSE = "mdi:link-variant"  # derive this pose from the other one

# What a pose's provenance looks like on its row: a small word after the position,
# only when there is something to say. An observed pose says nothing -- it is the
# normal state, and a chip on every row is a chip on none.
_PROVENANCE_CHIP = {
    PoseProvenance.DERIVED: (
        "derived",
        TEXT_MUTED_COLOR,
        "Worked out from the other pose, not yet centred by hand here.",
    ),
    PoseProvenance.STALE: (
        "stale",
        SEMANTIC_WARNING_COLOR,
        "Set by hand, but the other pose has moved since; it may no longer "
        "describe where the lamella is.",
    ),
}

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
    derive_clicked = pyqtSignal(str, object)  # pose name, orientation or None

    def __init__(
        self,
        pose_name: str,
        state: Optional[MicroscopeState],
        parent: Optional[QWidget] = None,
        provenance: PoseProvenance = PoseProvenance.OBSERVED,
        derive_orientations: Optional[List[str]] = None,
    ) -> None:
        """*derive_orientations* are the orientations this pose may be derived into;
        more than one and the derive button offers a menu, otherwise it derives
        straight away (into the one orientation, or into whatever the derivation
        decides)."""
        super().__init__(parent)
        self.pose_name = pose_name
        self._state = state
        self._popup: Optional[_PoseDetailPopup] = None
        self.derive_orientations = list(derive_orientations or [])
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

        self.provenance_label = QLabel()
        self.provenance_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.provenance_label)
        self.set_provenance(provenance)

        self.btn_derive = IconToolButton(
            icon=ICON_DERIVE_POSE,
            tooltip="Derive from the other pose",
            size=_BTN_SIZE.width(),
        )
        layout.addWidget(self.btn_derive)
        self.btn_derive.clicked.connect(self._on_derive_clicked)

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

        self.btn_update.clicked.connect(
            lambda: self.update_clicked.emit(self.pose_name)
        )
        self.btn_move_to.clicked.connect(
            lambda: self.move_to_clicked.emit(self.pose_name)
        )

        self.set_state(state)

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

    def set_provenance(self, provenance: PoseProvenance) -> None:
        chip = _PROVENANCE_CHIP.get(PoseProvenance(provenance))
        if chip is None:
            self.provenance_label.setText("")
            self.provenance_label.setToolTip("")
            self.provenance_label.setVisible(False)
            return
        text, colour, tooltip = chip
        self.provenance_label.setText(text)
        self.provenance_label.setToolTip(tooltip)
        self.provenance_label.setStyleSheet(
            f"background: transparent; color: {colour}; font-size: 11px;"
        )
        self.provenance_label.setVisible(True)

    def _build_derive_menu(self) -> QMenu:
        """One entry per orientation this pose may be derived into."""
        menu = QMenu(self)
        for orientation in self.derive_orientations:
            action = menu.addAction(f"Derive into the {orientation} orientation")
            action.triggered.connect(
                lambda _checked=False, o=orientation: self.derive_clicked.emit(
                    self.pose_name, o
                )
            )
        return menu

    def _on_derive_clicked(self) -> None:
        if len(self.derive_orientations) > 1:
            menu = self._build_derive_menu()
            menu.exec_(self.btn_derive.mapToGlobal(self.btn_derive.rect().bottomLeft()))
            return
        orientation = self.derive_orientations[0] if self.derive_orientations else None
        self.derive_clicked.emit(self.pose_name, orientation)

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
    derive_requested = pyqtSignal(str, object)  # pose name, orientation or None

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        # The orientations the fluorescence pose may be derived into: what the FM
        # declares it images from. Handed in by the host, which has the microscope.
        self._fluorescence_orientations: List[str] = []

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
        if lamella is None or not lamella.poses:
            return
        for pose_name in self._sorted_pose_names(lamella.poses):
            self._add_row(
                pose_name,
                lamella.poses[pose_name],
                provenance=_provenance_of(lamella, pose_name),
                derive_orientations=(
                    self._fluorescence_orientations
                    if pose_name == "FLUORESCENCE"
                    else []
                ),
            )

    def set_fluorescence_orientations(self, orientations: List[str]) -> None:
        """The orientations the fluorescence pose may be derived into."""
        self._fluorescence_orientations = list(orientations)

    def refresh_pose(
        self,
        pose_name: str,
        state: Optional[MicroscopeState],
        provenance: Optional[PoseProvenance] = None,
    ) -> None:
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
                if provenance is not None:
                    row.set_provenance(provenance)
                return

    def clear(self) -> None:
        self._list.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

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
        provenance: PoseProvenance = PoseProvenance.OBSERVED,
        derive_orientations: Optional[List[str]] = None,
    ) -> LamellaPoseRowWidget:
        row = LamellaPoseRowWidget(
            pose_name,
            state,
            provenance=provenance,
            derive_orientations=derive_orientations,
        )
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        row.update_clicked.connect(self.update_requested)
        row.move_to_clicked.connect(self.move_to_requested)
        row.derive_clicked.connect(self.derive_requested)
        return row


def _provenance_of(lamella, pose_name: str) -> PoseProvenance:
    """`Lamella.provenance_of`, tolerating the stand-ins the harnesses use."""
    provenance_of = getattr(lamella, "provenance_of", None)
    if provenance_of is None:
        return PoseProvenance.OBSERVED
    return provenance_of(pose_name)

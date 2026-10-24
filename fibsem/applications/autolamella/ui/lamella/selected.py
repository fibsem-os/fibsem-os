from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

from PyQt5.QtCore import QObject, QSize, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.poses import (
    current_objective_position,
    set_lamella_pose,
    set_objective_position,
    set_objective_position_on_all,
    stored_objective_position,
)
from fibsem.applications.autolamella.structures import Experiment, Lamella
from fibsem.constants import METRE_TO_MICRON, MICRON_TO_METRE
from fibsem.devices.fm import Objective
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import FibsemStagePosition, MicroscopeState
from fibsem.theme.tokens import (
    BORDER_COLOR,
    CANVAS_BG,
    NEUTRAL_550,
    SURFACE_COLOR,
    TEXT_COLOR,
)
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.control import control_of
from fibsem.ui.icon import ICON_MOVE_TO_POSITION, ICON_UPDATE_POSITION
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueSpinBox,
)
from fibsem.ui.widgets.form_builder import configure_spinbox, parameter_spec
from fibsem.ui.widgets.microscope_state_widget import MicroscopeStateWidget
from fibsem.utils import (
    NOT_AVAILABLE,
    format_current,
    format_distance,
    format_stage_position,
    format_voltage,
)


class SelectedLamellaWidget(QWidget):
    """Presentational panel for the currently selected lamella.

    Shows the objective-position control and the lamella's poses. This is a
    view: it reads display state from the lamella passed to :meth:`set_lamella`
    and emits signals for user actions; :class:`SelectedLamellaController`
    acts on them.
    """

    objective_position_changed = pyqtSignal(float)  # value in µm
    use_current_objective_requested = pyqtSignal()
    apply_objective_to_all_requested = pyqtSignal()
    move_objective_requested = pyqtSignal()  # move objective to stored position
    pose_update_requested = pyqtSignal(str)  # pose name
    pose_move_to_requested = pyqtSignal(str)  # pose name

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        # --- description row (read-only mirror; edited in the Lamella editor) ---
        self._lamella: Optional[Lamella] = None
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet(
            f"color: {NEUTRAL_550}; font-style: italic; background: transparent;"
        )
        self.description_label.setToolTip("Free-text note (edit in the Lamella editor)")

        # --- objective position row ---
        self.label_objective_position = QLabel("Objective Position")
        # Shown as the objective's position is (µm, its step and decimals); the range
        # is wide because a lamella's stored position is not checked against a device.
        self.spinbox_objective_position = ValueSpinBox()
        configure_spinbox(
            self.spinbox_objective_position, parameter_spec(Objective.position)
        )
        self.spinbox_objective_position.setRange(-20000.0, 20000.0)
        self.btn_objective_actions = IconToolButton(
            "mdi:dots-horizontal", tooltip="Actions"
        )
        self.btn_objective_actions.setPopupMode(IconToolButton.InstantPopup)
        self.btn_objective_actions.setStyleSheet(
            "QToolButton::menu-indicator { image: none; }"
        )
        obj_menu = QMenu(self)
        self._action_move_to_obj_pos = obj_menu.addAction("Move Objective to Position")
        obj_menu.addSeparator()
        self._action_use_current_obj_pos = obj_menu.addAction(
            "Use Current Objective Position"
        )
        self._action_apply_obj_to_all = obj_menu.addAction("Apply to All Lamella")
        self.btn_objective_actions.setMenu(obj_menu)

        # --- pose list ---
        self.pose_list = LamellaPoseListWidget()

        # --- layout ---
        content = QWidget()
        content_layout = QGridLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        obj_row = QWidget()
        obj_row_layout = QHBoxLayout(obj_row)
        obj_row_layout.setContentsMargins(0, 0, 0, 0)
        obj_row_layout.setSpacing(2)
        obj_row_layout.addWidget(self.spinbox_objective_position)
        obj_row_layout.addWidget(self.btn_objective_actions)
        content_layout.addWidget(self.description_label, 0, 0, 1, 2)
        content_layout.addWidget(self.label_objective_position, 1, 0)
        content_layout.addWidget(obj_row, 1, 1)
        content_layout.addWidget(self.pose_list, 2, 0, 1, 2)

        self._panel = TitledPanel(
            "Selected Lamella", content=content, collapsible=False
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._panel)

        # --- wire internal widgets to public signals ---
        self.spinbox_objective_position.valueChanged.connect(
            self.objective_position_changed
        )
        self._action_move_to_obj_pos.triggered.connect(self.move_objective_requested)
        self._action_use_current_obj_pos.triggered.connect(
            self.use_current_objective_requested
        )
        self._action_apply_obj_to_all.triggered.connect(
            self.apply_objective_to_all_requested
        )
        self.pose_list.update_requested.connect(self.pose_update_requested)
        self.pose_list.move_to_requested.connect(self.pose_move_to_requested)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Refresh the panel display from *lamella* (or clear it if None)."""
        # re-point the live description mirror at the newly selected lamella
        if self._lamella is not None:
            try:
                self._lamella.events.description.disconnect(self._refresh_description)  # type: ignore[union-attr]
            except (TypeError, ValueError, RuntimeError):
                pass
        self._lamella = lamella

        if lamella is None:
            self._panel.set_title("Selected Lamella")
            self.description_label.setVisible(False)
            self.label_objective_position.setVisible(False)
            self.spinbox_objective_position.setVisible(False)
            self.btn_objective_actions.setVisible(False)
            self.pose_list.set_lamella(None)
            self.pose_list.setVisible(False)
            return

        # show which lamella is selected in the panel title
        self._panel.set_title(f"Selected Lamella — {lamella.name}")

        # read-only description mirror; live-updates when edited in the Lamella editor
        self.description_label.setVisible(True)
        self._refresh_description()
        lamella.events.description.connect(self._refresh_description)  # type: ignore[union-attr]

        # objective position (shown in µm). The controls are shown whenever the
        # lamella has a fluorescence pose, so the objective can still be set/restored
        # (e.g. via "Use Current Objective Position") even when the position is unset.
        # If it were gated on obj_pos being set, a wiped (None) objective would hide
        # the only UI to recover it.
        has_fluorescence_pose = lamella.fluorescence_pose is not None
        obj_pos = (
            lamella.fluorescence_pose.objective_position
            if has_fluorescence_pose
            else None
        )
        self.set_objective_value_um(
            obj_pos * METRE_TO_MICRON if obj_pos is not None else 0.0
        )
        self.label_objective_position.setVisible(has_fluorescence_pose)
        self.spinbox_objective_position.setVisible(has_fluorescence_pose)
        self.btn_objective_actions.setVisible(has_fluorescence_pose)
        # "Apply to All" copies the current value to every lamella; disable it when the
        # objective is unset so the 0.0 placeholder cannot clobber other lamellae's focus.
        # "Use Current Objective Position" stays enabled as the recovery path.
        self._action_apply_obj_to_all.setEnabled(obj_pos is not None)

        # poses
        self.pose_list.set_lamella(lamella)
        self.pose_list.setVisible(bool(lamella.poses))

    def _refresh_description(self) -> None:
        """Mirror the current lamella's description into the read-only label."""
        if self._lamella is None:
            return
        text = self._lamella.description
        self.description_label.setText(text or "No description")
        self.description_label.setToolTip(
            text or "Free-text note (edit in the Lamella editor)"
        )

    def refresh_pose(self, pose_name: str, state) -> None:
        """Update one pose row in place, without rebuilding the list."""
        self.pose_list.refresh_pose(pose_name, state)

    def set_instrument_actions_enabled(self, enabled: bool) -> None:
        """Whether the actions that move or read the stage and objective are on:
        off while a run, which owns them, is on."""
        self._action_move_to_obj_pos.setEnabled(enabled)
        self._action_use_current_obj_pos.setEnabled(enabled)
        self.pose_list.set_stage_actions_enabled(enabled)

    def objective_value_um(self) -> float:
        """Current objective spinbox value, in µm."""
        return self.spinbox_objective_position.value()

    def set_objective_value_um(self, um: float) -> None:
        """Set the objective spinbox value (µm) without emitting a change."""
        self.spinbox_objective_position.blockSignals(True)
        self.spinbox_objective_position.setValue(um)
        self.spinbox_objective_position.blockSignals(False)


class SelectedLamellaController(QObject):
    """Acts on a :class:`SelectedLamellaWidget`'s requests through ``poses``,
    which saves: the Experiment tab's and the coincidence viewer's one copy.

    ``lamella`` gives the selected lamella; ``move_stage`` moves the stage to a
    pose's position the way its host moves the stage; ``busy`` names the
    lamellae a run is working on, which Apply to All leaves alone.
    """

    def __init__(
        self,
        widget: SelectedLamellaWidget,
        experiment: Callable[[], Optional[Experiment]],
        microscope: Callable[[], Optional[FibsemMicroscope]],
        lamella: Callable[[], Optional[Lamella]],
        move_stage: Callable[[FibsemStagePosition], None],
        busy: Callable[[], Sequence[str]] = tuple,
    ) -> None:
        super().__init__(widget)
        self.widget = widget
        self._experiment = experiment
        self._microscope = microscope
        self._lamella = lamella
        self._move_stage = move_stage
        self._busy = busy
        self._worker: Optional[FunctionWorker] = None

        widget.objective_position_changed.connect(self.store_objective_position)
        widget.use_current_objective_requested.connect(self.use_current_objective)
        widget.apply_objective_to_all_requested.connect(self.apply_objective_to_all)
        widget.move_objective_requested.connect(self.move_objective)
        widget.pose_update_requested.connect(self.set_pose)
        widget.pose_move_to_requested.connect(self.move_to_pose)

    def _ask(self, title: str, text: str) -> bool:
        ret = QMessageBox.question(
            self.widget, title, text, QMessageBox.Yes | QMessageBox.No
        )
        return ret == QMessageBox.Yes

    def _fm_lamella(self) -> Optional[Lamella]:
        """The selected lamella with a fluorescence pose, on a microscope with
        FM; None, said, otherwise."""
        microscope = self._microscope()
        if microscope is None or microscope.fm is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return None
        lamella = self._lamella()
        if lamella is None or lamella.fluorescence_pose is None:
            notification_service.show_toast("No lamella selected.", "warning")
            return None
        return lamella

    def store_objective_position(self, value_um: float) -> None:
        lamella = self._lamella()
        if lamella is None or lamella.fluorescence_pose is None:
            return
        set_objective_position(self._experiment(), lamella, value_um * MICRON_TO_METRE)

    def use_current_objective(self) -> None:
        lamella = self._fm_lamella()
        if lamella is None:
            return
        value_m = current_objective_position(self._microscope().fm)
        if value_m is None:
            notification_service.show_toast(
                "Objective position unavailable.", "warning"
            )
            return
        set_objective_position(self._experiment(), lamella, value_m)
        # full refresh so the objective value shows and "Apply to All" re-enables
        self.widget.set_lamella(lamella)
        notification_service.show_toast(
            f"Set objective position to {value_m * METRE_TO_MICRON:.1f} µm "
            f"for {lamella.name}.",
            "info",
        )

    def apply_objective_to_all(self) -> None:
        experiment = self._experiment()
        if experiment is None:
            return
        value_um = self.widget.objective_value_um()
        count = set_objective_position_on_all(
            experiment, value_um * MICRON_TO_METRE, skip=self._busy()
        )
        if count:
            notification_service.show_toast(
                f"Applied objective position ({value_um:.1f} µm) to {count} lamella.",
                "info",
            )

    def move_objective(self) -> None:
        """Move the FM objective to the lamella's stored position, off the GUI
        thread; independent of the stage."""
        lamella = self._fm_lamella()
        if lamella is None:
            return
        objective = self._microscope().fm.objective
        try:
            position = stored_objective_position(self._microscope().fm, lamella)
        except ValueError as e:
            notification_service.show_toast(str(e), "warning")
            return
        said = f"{position * METRE_TO_MICRON:.1f} µm for {lamella.name}"
        if not self._ask("Move Objective", f"Move objective to {said}?"):
            return
        logging.info(f"Moving objective to {said}.")
        self._worker = FunctionWorker(objective.move_absolute, position)
        self._worker.returned.connect(
            lambda _: notification_service.show_toast(
                f"Moved objective to {said}.", "info"
            )
        )
        self._worker.errored.connect(
            lambda e: notification_service.show_toast(
                f"Failed to move objective: {e}", "warning"
            )
        )
        self._worker.start(
            control=control_of(self._microscope()), action="move objective"
        )

    def set_pose(self, pose_name: str) -> None:
        """The current microscope state as the lamella's pose *pose_name*."""
        microscope = self._microscope()
        if microscope is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self._lamella()
        if lamella is None or not pose_name:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        state = microscope.get_microscope_state()
        if state is None or state.stage_position is None:
            notification_service.show_toast(
                "Failed to get microscope state.", "warning"
            )
            return
        if not self._ask(
            "Set Pose Confirmation",
            f"Set current position as pose '{pose_name}' for {lamella.name}?\n"
            f"{state.stage_position.pretty}",
        ):
            return
        if set_lamella_pose(self._experiment(), microscope, lamella, pose_name, state):
            self.widget.refresh_pose("FLUORESCENCE", lamella.fluorescence_pose)
        self.widget.refresh_pose(pose_name, state)
        notification_service.show_toast(
            f"Set current position as pose '{pose_name}' for {lamella.name}.", "info"
        )

    def move_to_pose(self, pose_name: str) -> None:
        if self._microscope() is None:
            notification_service.show_toast("No microscope connected.", "warning")
            return
        lamella = self._lamella()
        if lamella is None or not pose_name:
            notification_service.show_toast("No lamella selected.", "warning")
            return
        pose = lamella.poses.get(pose_name)
        if pose is None or pose.stage_position is None:
            notification_service.show_toast(
                f"Pose '{pose_name}' has no stage position.", "warning"
            )
            return
        if not self._ask(
            "Move to Pose Confirmation",
            f"Move to pose '{pose_name}' for {lamella.name}?\n"
            f"{pose.stage_position.pretty}",
        ):
            return
        logging.info(f"Moving to pose '{pose_name}' for {lamella.name}.")
        self._move_stage(pose.stage_position)


_NAME_WIDTH = 110
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 2 + 8  # 2 buttons + 1 gap

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

    def __init__(
        self,
        pose_name: str,
        state: Optional[MicroscopeState],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.pose_name = pose_name
        self._state = state
        self._popup: Optional[_PoseDetailPopup] = None
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

    def set_stage_actions_enabled(self, enabled: bool) -> None:
        self.btn_move_to.setEnabled(enabled)
        self.btn_update.setEnabled(enabled)

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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

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
        self._stage_actions = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_lamella(self, lamella: Optional[Lamella]) -> None:
        """Rebuild the rows from the lamella's existing poses."""
        self._list.clear()
        if lamella is None or not lamella.poses:
            return
        for pose_name in self._sorted_pose_names(lamella.poses):
            self._add_row(pose_name, lamella.poses[pose_name])

    def set_stage_actions_enabled(self, enabled: bool) -> None:
        """Whether each row's Move to and Update are on."""
        self._stage_actions = enabled
        for i in range(self._list.count()):
            row = self._list.itemWidget(self._list.item(i))
            if isinstance(row, LamellaPoseRowWidget):
                row.set_stage_actions_enabled(enabled)

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
        self, pose_name: str, state: Optional[MicroscopeState]
    ) -> LamellaPoseRowWidget:
        row = LamellaPoseRowWidget(pose_name, state)
        row.set_stage_actions_enabled(self._stage_actions)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)
        row.update_clicked.connect(self.update_requested)
        row.move_to_clicked.connect(self.move_to_requested)
        return row

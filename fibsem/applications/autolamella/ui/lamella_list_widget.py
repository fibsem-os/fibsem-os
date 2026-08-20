from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt5.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QCursor, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskStatus,
    DefectState,
    DefectType,
    Lamella,
)
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import (
    CANVAS_BG,
    NEUTRAL_550,
)
from fibsem.ui.widgets.custom_widgets import IconToolButton

_NAME_MIN_WIDTH = 160
_BTN_SIZE = QSize(32, 32)
_ROW_HEIGHT = 40
_BTN_SPACER_WIDTH = _BTN_SIZE.width() * 3 + 8 * 2  # 3 buttons + 2 gaps



class _LamellaTooltip(QWidget):
    """Frameless popup showing a thumbnail image when hovering a lamella name."""

    def __init__(self) -> None:
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)  # type: ignore[call-overload]
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(f"background: {CANVAS_BG}; border: 1px solid #3a3d42;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self._img_label = QLabel()
        self._img_label.setFixedSize(256, 170)
        layout.addWidget(self._img_label)

    def set_image(self, arr: np.ndarray) -> None:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=2)
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        h, w, c = arr.shape
        qimg = QImage(arr.data, w, h, w * c, QImage.Format_RGB888)
        self._img_label.setPixmap(
            QPixmap.fromImage(qimg).scaled(256, 170, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def show_near_cursor(self) -> None:
        from PyQt5.QtWidgets import QApplication
        cursor = QCursor.pos()
        self.adjustSize()
        w, h = self.width(), self.height()
        x = cursor.x() + 14
        y = cursor.y() + 14  # below the cursor by default
        screen = QApplication.screenAt(cursor)
        if screen is not None:
            geom = screen.availableGeometry()
            x = max(geom.left(), min(x, geom.right() - w))
            y = max(geom.top(), min(y, geom.bottom() - h))
        self.move(x, y)
        self.show()
        self.raise_()


def _status_text(lamella: Lamella) -> tuple[str, str]:
    """Return (text, stylesheet) for the status column."""
    ts = lamella.task_state
    if ts and ts.status == AutoLamellaTaskStatus.InProgress:
        return f"{ts.name}", f"color: {stylesheets.PRIMARY_COLOR}; background: transparent;"
    last = lamella.last_completed_task
    if last:
        return last.completed, f"color: {NEUTRAL_550}; background: transparent;"
    return "", "background: transparent;"


def _defect_icon(lamella: Lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator button."""
    d = lamella.defect
    if d.state == DefectType.NONE:
        return "mdi:check-circle", stylesheets.GREEN_COLOR, "No defect"
    if d.state == DefectType.REWORK:
        return "mdi:refresh-circle", stylesheets.DEFECT_ORANGE_COLOR, f"Rework required{': ' + d.description if d.description else ''}"
    return "mdi:close-circle", stylesheets.DEFECT_RED_COLOR, f"Failure{': ' + d.description if d.description else ''}"


class LamellaRowWidget(QWidget):
    move_to_clicked = pyqtSignal(object)       # Lamella
    edit_clicked = pyqtSignal(object)          # Lamella
    remove_clicked = pyqtSignal(object)        # Lamella
    defect_changed = pyqtSignal(object)        # Lamella
    selection_changed = pyqtSignal(object, bool)  # Lamella, checked

    def __init__(
        self,
        lamella: Lamella,
        checked: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.lamella = lamella
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._popup: Optional[_LamellaTooltip] = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.timeout.connect(self._show_popup)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(8)

        self.checkbox = QCheckBox()
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("background: transparent;")
        layout.addWidget(self.checkbox)

        self.name_label = QLabel()
        self.name_label.setMinimumWidth(_NAME_MIN_WIDTH)
        self.name_label.setStyleSheet("background: transparent;")
        layout.addWidget(self.name_label)

        self.status_label = QLabel()
        layout.addWidget(self.status_label, 1)

        self.btn_defect = QToolButton()
        self.btn_defect.setIcon(fibsem_icon("mdi:circle", color=stylesheets.GREEN_COLOR))
        self.btn_defect.setFixedSize(_BTN_SIZE)
        self.btn_defect.setStyleSheet(stylesheets.TOOLBUTTON_ICON_STYLESHEET)
        layout.addWidget(self.btn_defect)

        self.btn_edit = IconToolButton(icon="mdi:pencil", tooltip="Edit Lamella", size=_BTN_SIZE.width())
        layout.addWidget(self.btn_edit)

        self.btn_remove = IconToolButton(icon="mdi:trash-can-outline", tooltip="Remove", size=_BTN_SIZE.width())
        layout.addWidget(self.btn_remove)

        self.checkbox.stateChanged.connect(
            lambda s: self.selection_changed.emit(self.lamella, bool(s))
        )
        self.btn_defect.installEventFilter(self)
        self.btn_defect.clicked.connect(self._on_defect_clicked)
        self.btn_edit.clicked.connect(lambda: self.edit_clicked.emit(self.lamella))
        self.btn_remove.clicked.connect(self._on_remove_clicked)

        # Connect to evented fields so only this row updates when its data changes.
        # task_history is a plain List so append won't fire; task_state.events covers
        # in-progress and completion transitions.
        #
        # `task_state.name`/`.status` are written by the running workflow
        # (`workflows/tasks/base.py:234,239`) on the FunctionWorker thread, and psygnal
        # delivers synchronously on the emitting thread -- so `refresh` carries
        # @ensure_main_thread (FIB-565). `defect` and `description` are only ever
        # written by GUI widgets; marshalling is a no-op for them, because
        # ensure_main_thread runs inline when already on the GUI thread.
        #
        # The two task_state connections are DISABLED, not deleted, and the difference
        # matters -- see FIB-604. Marshalling makes the refresh arrive *later*, by which
        # point `clear()` may have destroyed this widget, and an exception escaping a Qt
        # slot is not a logged traceback: PyQt5 calls qFatal and the process aborts
        # (FIB-329, no excepthook installed). That is what happened on 2026-08-12 --
        # the app died mid-workflow, and the abort killed a thumbnail write partway
        # through, leaving a 0-byte file that then aborted every later load (FIB-602).
        #
        # `_on_workflow_update` refreshes this row by name, so the display still follows
        # a run. What is lost is determinism: its InProgress update is emitted *before*
        # `pre_task` writes the new name and status, and only arrives after because
        # delivery is queued. A row can therefore show the previous task for the length
        # of the current one. Slightly stale beats a lost run -- Patrick's call, and the
        # right one while the crash is fatal.
        #
        # Restore these only once FIB-604 gives the refresh a trigger that fires after
        # the write; at that point they should be deleted rather than re-enabled.
        #
        # type: ignore because @evented adds .events dynamically and pyright can't see it.
        # lamella.task_state.events.name.connect(self.refresh)    # DISABLED: FIB-604
        # lamella.task_state.events.status.connect(self.refresh)  # DISABLED: FIB-604
        lamella.events.defect.connect(self.refresh)             # type: ignore[union-attr]
        lamella.events.description.connect(self.refresh)        # type: ignore[union-attr]

        self.refresh()

    def _on_defect_clicked(self) -> None:
        menu = QMenu(self)
        action_none = menu.addAction(
            fibsem_icon("mdi:check-circle", color=stylesheets.GREEN_COLOR), "No defect"
        )
        action_rework = menu.addAction(
            fibsem_icon("mdi:refresh-circle", color=stylesheets.DEFECT_ORANGE_COLOR), "Rework required"
        )
        action_failure = menu.addAction(
            fibsem_icon("mdi:close-circle", color=stylesheets.DEFECT_RED_COLOR), "Failure"
        )

        chosen = menu.exec_(self.btn_defect.mapToGlobal(
            self.btn_defect.rect().bottomLeft()
        ))

        if chosen == action_none:
            self.lamella.defect = DefectState(state=DefectType.NONE)
        elif chosen == action_rework:
            self.lamella.defect = DefectState(state=DefectType.REWORK)
        elif chosen == action_failure:
            self.lamella.defect = DefectState(state=DefectType.FAILURE)
        else:
            return

        self.refresh()
        self.defect_changed.emit(self.lamella)

    def _on_remove_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Remove Lamella",
            f"Remove <b>{self.lamella.name}</b>?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.remove_clicked.emit(self.lamella)

    def eventFilter(self, obj, event) -> bool:
        if obj is self.btn_defect:
            if event.type() == QEvent.Enter:
                self._hover_timer.start(400)
            elif event.type() == QEvent.Leave:
                self._hover_timer.stop()
                if self._popup:
                    self._popup.hide()
        return super().eventFilter(obj, event)

    def _show_popup(self) -> None:
        if self._popup is None:
            self._popup = _LamellaTooltip()
        self._popup.set_image(self.lamella.get_thumbnail())
        self._popup.show_near_cursor()

    @ensure_main_thread
    def refresh(self) -> None:
        """Re-read all display fields from the stored Lamella."""
        self.name_label.setText(self.lamella.name)
        self.setToolTip(self.lamella.description or "")

        icon_name, icon_color, tooltip = _defect_icon(self.lamella)
        self.btn_defect.setIcon(fibsem_icon(icon_name, color=icon_color))
        self.btn_defect.setToolTip(tooltip)

        status_text, status_style = _status_text(self.lamella)
        self.status_label.setText(status_text)
        self.status_label.setStyleSheet(status_style)


class _LamellaListHeader(QWidget):
    select_all_changed = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background: {CANVAS_BG};")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(8)

        # spans checkbox indicator + spacing + name column
        self.checkbox_all = QCheckBox("Select All")
        self.checkbox_all.setChecked(True)
        self.checkbox_all.setStyleSheet("font-weight: bold; background: transparent;")
        self.checkbox_all.setMinimumWidth(24 + 8 + _NAME_MIN_WIDTH)
        layout.addWidget(self.checkbox_all)

        status_header = QLabel("Status")
        status_header.setStyleSheet("font-weight: bold; background: transparent;")
        layout.addWidget(status_header, 1)

        spacer = QWidget()
        spacer.setFixedWidth(_BTN_SPACER_WIDTH)
        spacer.setStyleSheet("background: transparent;")
        layout.addWidget(spacer)

        self.checkbox_all.stateChanged.connect(
            lambda s: self.select_all_changed.emit(bool(s))
        )


class LamellaListWidget(QWidget):
    """List widget displaying Lamella objects with name, defect, status and actions."""

    move_to_requested = pyqtSignal(object)   # Lamella
    edit_requested = pyqtSignal(object)      # Lamella
    remove_requested = pyqtSignal(object)    # Lamella
    defect_changed = pyqtSignal(object)      # Lamella
    selection_changed = pyqtSignal(list)     # List[Lamella]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._header = _LamellaListHeader()
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

        self._header.select_all_changed.connect(self.set_all_selected)

        self._btn_visible = {
            "edit": True,
            "remove": True,
            "defect": True,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_lamella(self, lamella: Lamella, checked: bool = False) -> LamellaRowWidget:
        row = LamellaRowWidget(lamella, checked)
        item = QListWidgetItem()
        item.setSizeHint(QSize(0, _ROW_HEIGHT))
        self._list.addItem(item)
        self._list.setItemWidget(item, row)

        row.move_to_clicked.connect(self.move_to_requested)
        row.edit_clicked.connect(self.edit_requested)
        row.remove_clicked.connect(self._on_remove_clicked)
        row.defect_changed.connect(self.defect_changed)
        row.selection_changed.connect(self._on_row_selection_changed)

        self._apply_btn_visibility(row)
        self._sync_select_all()
        return row

    def enable_actions_button(self, visible: bool) -> None:
        self.enable_edit_action(visible)

    def enable_move_to_action(self, visible: bool) -> None:
        pass  # not available

    def enable_edit_action(self, visible: bool) -> None:
        self._btn_visible["edit"] = visible
        for i in range(self._list.count()):
            row = self._row(i)
            if row is not None:
                row.btn_edit.setVisible(visible)

    def enable_remove_button(self, visible: bool) -> None:
        self._btn_visible["remove"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_remove.setVisible(visible)

    def enable_defect_button(self, visible: bool) -> None:
        self._btn_visible["defect"] = visible
        for i in range(self._list.count()):
            self._row(i).btn_defect.setVisible(visible)

    def remove_lamella(self, lamella: Lamella) -> None:
        for i in range(self._list.count()):
            if self._row(i).lamella is lamella:
                self._list.takeItem(i)
                break
        self._sync_select_all()

    def refresh_lamella(self, lamella: Lamella) -> None:
        """Refresh the display of a single lamella row."""
        for i in range(self._list.count()):
            row = self._row(i)
            if row.lamella is lamella:
                row.refresh()
                break

    def refresh_all(self) -> None:
        """Refresh every row from its current Lamella state."""
        for i in range(self._list.count()):
            self._row(i).refresh()

    def get_selected(self) -> List[Lamella]:
        return [
            self._row(i).lamella
            for i in range(self._list.count())
            if self._row(i).checkbox.isChecked()
        ]

    def clear(self) -> None:
        self._list.clear()

    def set_all_selected(self, checked: bool) -> None:
        """Tick or untick every row, and bring the header checkbox with them.

        Public because the header is not the only thing that clears the selection --
        starting a run and adding to the queue both do. Those callers used to invoke
        the header's slot directly, which left the box reading "Select All" over a
        list where nothing was ticked (FIB-577).
        """
        for i in range(self._list.count()):
            row = self._row(i)
            row.checkbox.blockSignals(True)
            row.checkbox.setChecked(checked)
            row.checkbox.blockSignals(False)
        # Redundant when the header emitted into here (it is already right), and
        # load-bearing for every other caller. Cheap enough not to branch on.
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _row(self, i: int) -> LamellaRowWidget:
        return self._list.itemWidget(self._list.item(i))  # type: ignore[return-value]

    def _apply_btn_visibility(self, row: LamellaRowWidget) -> None:
        row.btn_edit.setVisible(self._btn_visible["edit"])
        row.btn_remove.setVisible(self._btn_visible["remove"])
        row.btn_defect.setVisible(self._btn_visible["defect"])

    def _on_remove_clicked(self, lamella: Lamella) -> None:
        self.remove_lamella(lamella)
        self.remove_requested.emit(lamella)

    def _on_row_selection_changed(self, *_) -> None:
        self._sync_select_all()
        self.selection_changed.emit(self.get_selected())

    def _sync_select_all(self) -> None:
        count = self._list.count()
        if count == 0:
            return
        n_checked = sum(
            self._row(i).checkbox.isChecked()
            for i in range(count)
        )
        cb = self._header.checkbox_all
        cb.blockSignals(True)
        if n_checked == 0:
            cb.setCheckState(Qt.Unchecked)
        elif n_checked == count:
            cb.setCheckState(Qt.Checked)
        else:
            cb.setTristate(True)
            cb.setCheckState(Qt.PartiallyChecked)
        cb.blockSignals(False)

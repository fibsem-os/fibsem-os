"""The microscope configuration, shown in one window.

What the connected instrument is and how it was set up, by kind of value -- the same
kinds the configuration file is divided into:

- **Instrument**: which microscope this is, the file it was started from, and which
  subsystems are fitted (reported by the instrument, or the backend's own answer where
  it cannot be asked -- the file does not state them).
- **Geometry**: the columns, the stage and where each device sees the sample. Set when
  the system is installed, by the guided setup; shown here, not edited.
- **Calibration**: what was measured at this instrument -- the holder's slots and the
  fluorescence objective. Never typed in: the holder's slots come from its guided
  calibration, opened from here.
- **Defaults**: what the columns and the acquire tab start at -- read off the
  instrument, edited, and saved by the window's Save; Apply to Microscope sets them.
- **Session**: what the application remembers for this configuration between
  sessions (`fibsem.session_state`) -- the file it is kept in and everything each
  section holds. Shown, not edited: each section is changed where it is used.

Only while connected: everything here is read from the live session. The Defaults tab
is the only one with anything to save; the window says when it has unsaved changes.
"""

import logging
import math
import os
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import ImageSettings
from fibsem.ui import notification_service, stylesheets
from fibsem.ui.tokens import (
    OK_COLOR,
    PANEL_COLOR,
    ROW_ALT_COLOR,
    SURFACE_COLOR,
    TEXT_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
    WARN_COLOR,
)
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.microscope_defaults_widget import MicroscopeDefaultsWidget

NOT_STATED = "—"


# ---- small pieces ---------------------------------------------------------------------


def _muted(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; background: transparent;")
    label.setWordWrap(True)
    return label


def _status(text: str, colour: str) -> QLabel:
    """A state as coloured text."""
    label = QLabel(text)
    label.setStyleSheet(f"color: {colour}; background: transparent;")
    return label


def _value(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet(f"color: {TEXT_COLOR}; background: transparent;")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _form(rows: Iterable[Tuple[str, str]]) -> QWidget:
    widget = QWidget()
    form = QFormLayout(widget)
    form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.DontWrapRows)
    form.setContentsMargins(8, 6, 8, 6)
    form.setHorizontalSpacing(18)
    for name, text in rows:
        label = QLabel(name)  # not wrapped: a wrapping label is the one Qt squeezes
        label.setStyleSheet(f"color: {TEXT_MUTED_COLOR}; background: transparent;")
        form.addRow(label, _value(text))
    return widget


def _table(headers: Sequence[str], rows: List[Sequence], stretch: int) -> QTableWidget:
    """A read-only table; a cell may be a string or a widget."""
    table = QTableWidget(len(rows), len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setShowGrid(False)
    table.setAlternatingRowColors(True)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.NoSelection)
    table.setFocusPolicy(Qt.NoFocus)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    table.setStyleSheet(
        f"QTableWidget {{ background: {SURFACE_COLOR}; alternate-background-color: "
        f"{ROW_ALT_COLOR}; color: {TEXT_COLOR}; border: none; }}"
        f"QHeaderView::section {{ background: {PANEL_COLOR}; color: {TEXT_MUTED_COLOR};"
        f" border: none; padding: 4px 6px; }}"
    )
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if isinstance(cell, QWidget):
                holder = QWidget()
                layout = QHBoxLayout(holder)
                layout.setContentsMargins(6, 0, 6, 0)
                layout.addWidget(cell)
                layout.addStretch()
                table.setCellWidget(r, c, holder)
            else:
                table.setItem(r, c, QTableWidgetItem(str(cell)))
    header = table.horizontalHeader()
    header.setDefaultAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    for c in range(len(headers)):
        header.setSectionResizeMode(
            c, QHeaderView.Stretch if c == stretch else QHeaderView.ResizeToContents
        )
    table.verticalHeader().setDefaultSectionSize(26)
    table.setFixedHeight(26 * len(rows) + 30)
    return table


def _panel(title: str, content: QWidget) -> TitledPanel:
    panel = TitledPanel(title, content=content, collapsible=False)
    policy = panel.sizePolicy()
    policy.setVerticalPolicy(policy.Maximum)
    panel.setSizePolicy(policy)
    return panel


def _mm(metres: Optional[float]) -> str:
    return NOT_STATED if metres is None else f"{metres * 1e3:.2f} mm"


def _degrees(value: Optional[float]) -> str:
    return NOT_STATED if value is None else f"{value:g}°"


# ---- the tabs -------------------------------------------------------------------------

SOURCE_LABELS = {"instrument": "Instrument", "backend": "Backend default"}


def devices_rows(microscope: FibsemMicroscope) -> List[Tuple[str, bool, str, str]]:
    """(device, fitted, reported by, detail) for each subsystem the application uses."""
    sources = getattr(microscope, "capability_sources", {}) or {}
    ion = microscope.system.ion
    loader = getattr(getattr(microscope, "_stage", None), "loader", None)

    def probed(key: str) -> Tuple[bool, str]:
        return (
            bool(microscope.is_available(key)),
            SOURCE_LABELS.get(sources.get(key, ""), NOT_STATED),
        )

    return [
        (
            "Electron column",
            bool(microscope.is_available("electron_beam")),
            "Instrument",
            "",
        ),
        (
            "Ion column",
            bool(microscope.is_available("ion_beam")),
            "Instrument",
            f"Plasma · {ion.plasma_gas}" if ion.plasma_gas else "",
        ),
        (
            "Stage",
            bool(microscope.is_available("stage")),
            "Instrument",
            "CompuStage" if microscope.stage_is_compustage else "",
        ),
        ("Manipulator", *probed("manipulator"), ""),
        ("GIS", *probed("gis"), ""),
        ("Multichem", *probed("gis_multichem"), ""),
        ("Sputter coater", *probed("gis_sputter_coater"), ""),
        ("Fluorescence", microscope.fm is not None, "Configuration", ""),
        ("Grid loader", loader is not None, "Instrument", ""),
    ]


def instrument_tab(microscope: FibsemMicroscope) -> QWidget:
    info = microscope.system.info
    path = getattr(microscope, "configuration_path", None)
    identity = _form(
        [
            ("Name", info.name),
            ("Manufacturer", f"{info.manufacturer}  ·  {info.model}"),
            ("Serial number", str(info.serial_number)),
            ("Address", str(info.ip_address)),
            (
                "Software",
                f"{info.software_version}  (hardware {info.hardware_version})",
            ),
            ("fibsemOS", str(info.fibsem_version)),
            ("Revision", str(info.fibsem_revision)),
        ]
    )
    configuration = _form(
        [
            ("File", os.path.basename(path) if path else "Not started from a file"),
            (
                "Session state",
                f"config/session/{os.path.basename(path)}" if path else NOT_STATED,
            ),
        ]
    )
    devices = _table(
        ["Device", "", "Reported by", "Detail"],
        [
            (
                name,
                _status(
                    "Fitted" if fitted else "Not fitted",
                    OK_COLOR if fitted else TEXT_MUTED_COLOR,
                ),
                source,
                detail,
            )
            for name, fitted, source, detail in devices_rows(microscope)
        ],
        stretch=3,
    )

    left = QVBoxLayout()
    left.addWidget(_panel("Instrument", identity))
    left.addWidget(_panel("Configuration file", configuration))
    left.addStretch()
    right = QVBoxLayout()
    right.addWidget(_panel("Devices", devices))
    right.addWidget(
        _muted(
            "Reported by the instrument when it can be asked; otherwise the "
            "backend's own answer. Not stated in the configuration file."
        )
    )
    right.addStretch()
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.addLayout(left, 1)
    layout.addLayout(right, 1)
    return widget


def geometry_tab(microscope: FibsemMicroscope) -> QWidget:
    system = microscope.system
    stage = system.stage
    columns = _table(
        ["Column", "Tilt", "Eucentric height", "Plasma gas"],
        [
            (
                "Electron",
                _degrees(system.electron.column_tilt),
                _mm(system.electron.eucentric_height),
                NOT_STATED,
            ),
            (
                "Ion",
                _degrees(system.ion.column_tilt),
                _mm(system.ion.eucentric_height),
                system.ion.plasma_gas or NOT_STATED,
            ),
        ],
        stretch=3,
    )
    stage_form = _form(
        [
            ("Rotation reference", _degrees(stage.rotation_reference)),
            ("Facing the ion beam", f"{_degrees(stage.rotation_180)}  (derived)"),
            (
                "Rotation axis",
                ("Yes" if stage.rotation else "No") + "  (reported by the stage)",
            ),
            ("Milling angle", _degrees(stage.milling_angle)),
            ("Device range", f"± {_mm(stage.device_range.x)} in x"),
        ]
    )
    devices = _table(
        ["Device", "Origin (x)", "Acquires at"],
        [
            (
                name,
                _mm(device.origin.x),
                ", ".join(device.acquisition_orientations) or NOT_STATED,
            )
            for name, device in stage.devices.items()
        ],
        stretch=2,
    )

    left = QVBoxLayout()
    left.addWidget(_panel("Columns", columns))
    left.addWidget(_panel("Stage", stage_form))
    left.addStretch()
    right = QVBoxLayout()
    right.addWidget(_panel("Device positions", devices))
    right.addWidget(
        _muted("Where the stage travels for each instrument to see the sample.")
    )
    right.addStretch()
    grid = QHBoxLayout()
    grid.addLayout(left, 1)
    grid.addLayout(right, 1)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addWidget(
        _muted(
            "What the instrument is: set when the system is installed, by the guided "
            "setup. Every stage move and image projection is built on these numbers."
        )
    )
    layout.addLayout(grid)
    return widget


def slot_rows(holder) -> List[Tuple[str, str, str, str, str]]:
    """(slot, state, grid, position, captured) for each slot of *holder*."""
    rows = []
    for name, slot in holder.slots.items():
        calibration = slot.calibration
        if not slot.is_calibrated:
            state = "Not calibrated"
        elif calibration is not None and calibration.is_builtin:
            state = "Built in"
        else:
            state = "Calibrated"
        position = slot.position
        rows.append(
            (
                name,
                state,
                slot.loaded_grid.name if slot.loaded_grid is not None else NOT_STATED,
                f"x {position.x * 1e3:.2f}  y {position.y * 1e3:.2f} mm"
                if position is not None and slot.is_calibrated
                else NOT_STATED,
                calibration.captured_at.replace("T", " ")
                if calibration is not None and calibration.captured_at
                else NOT_STATED,
            )
        )
    return rows


STATE_COLOURS = {"Calibrated": OK_COLOR, "Built in": OK_COLOR}


def calibration_tab(microscope: FibsemMicroscope, calibrate_slots) -> QWidget:
    """*calibrate_slots* opens the holder's guided calibration, or is None."""
    holder = microscope._stage.holder
    rows = slot_rows(holder)
    calibrated = sum(1 for _, state, *_ in rows if state != "Not calibrated")
    holder_box = QWidget()
    box = QVBoxLayout(holder_box)
    box.setContentsMargins(0, 0, 0, 0)
    box.addWidget(
        _form(
            [
                ("Active holder", holder.name),
                ("Pre-tilt", _degrees(holder.pre_tilt)),
                ("Slots", f"{calibrated} of {len(rows)} calibrated"),
            ]
        )
    )
    box.addWidget(
        _table(
            ["Slot", "", "Grid", "Position", "Captured"],
            [
                (
                    name,
                    _status(state, STATE_COLOURS.get(state, WARN_COLOR)),
                    grid,
                    where,
                    when,
                )
                for name, state, grid, where, when in rows
            ],
            stretch=3,
        )
    )

    left = QVBoxLayout()
    left.addWidget(_panel("Sample holder", holder_box))
    if calibrate_slots is not None:
        button = QPushButton("Calibrate Slots…")
        button.setObjectName("calibrate_slots")
        button.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        button.clicked.connect(calibrate_slots)
        left.addWidget(button, alignment=Qt.AlignRight)
    left.addStretch()

    right = QVBoxLayout()
    fm = microscope.system.fm
    if microscope.fm is not None:
        right.addWidget(
            _panel(
                "Fluorescence objective",
                _form(
                    [
                        (
                            "Focus position",
                            _mm(fm.focus_position)
                            if fm.focus_position is not None
                            else "Not calibrated",
                        ),
                        (
                            "Travel limit",
                            _mm(fm.limit_position)
                            if fm.limit_position is not None
                            else "Not calibrated",
                        ),
                    ]
                ),
            )
        )
        right.addWidget(
            _muted(
                "Set on the Fluorescence tab's objective controls, and saved with "
                "Save calibration there."
            )
        )
    right.addWidget(
        _muted(
            "Measured at this instrument, never typed in. Saved into the "
            "configuration, and copied into every experiment."
        )
    )
    right.addStretch()

    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.addLayout(left, 3)
    layout.addLayout(right, 2)
    return widget


def _number(value, scale: float = 1.0, digits: int = 2, unit: str = "") -> str:
    try:
        number = f"{float(value) * scale:.{digits}f}"
    except (TypeError, ValueError):
        return NOT_STATED
    return f"{number} {unit}" if unit else number


def _angle(radians) -> str:
    try:
        return f"{math.degrees(float(radians)):.1f}°"
    except (TypeError, ValueError):
        return NOT_STATED


def _emission(value) -> str:
    if value is None:
        return "Reflection"
    if isinstance(value, str):
        return value
    return _number(value, digits=0, unit="nm")


def _channel_row(channel: dict) -> Tuple[str, str, str, str, str]:
    return (
        str(channel.get("name", NOT_STATED)),
        _number(channel.get("excitation_wavelength"), digits=0, unit="nm"),
        _emission(channel.get("emission_wavelength")),
        _number(channel.get("power"), scale=100, digits=1, unit="%"),
        _number(channel.get("exposure_time"), scale=1e3, digits=0, unit="ms"),
    )


CHANNEL_HEADERS = ("Channel", "Excitation", "Emission", "Power", "Exposure")


def _dicts(value) -> List[dict]:
    return [v for v in value if isinstance(v, dict)] if isinstance(value, list) else []


def session_sections(sections: dict) -> List[Tuple[str, Sequence[str], List[tuple]]]:
    """(title, headers, rows) for each section a session state holds, known ones
    first. A section with nothing in it has no rows."""
    positions = _dicts(sections.get("saved_positions"))
    occupancy = sections.get("holder_occupancy")
    occupancy = occupancy if isinstance(occupancy, dict) else {}
    fm = sections.get("fm") if isinstance(sections.get("fm"), dict) else {}
    working = fm.get("working") if isinstance(fm.get("working"), dict) else {}

    result = [
        (
            "Saved positions",
            ("Name", "x (mm)", "y (mm)", "z (mm)", "Rotation", "Tilt"),
            [
                (
                    str(p.get("name", NOT_STATED)),
                    _number(p.get("x"), 1e3),
                    _number(p.get("y"), 1e3),
                    _number(p.get("z"), 1e3),
                    _angle(p.get("r")),
                    _angle(p.get("t")),
                )
                for p in positions
            ],
        ),
        (
            "Grids in the holder",
            ("Slot", "Grid"),
            [
                (str(slot), str(grid.get("name", NOT_STATED)))
                for slot, grid in occupancy.items()
                if isinstance(grid, dict)
            ],
        ),
        (
            "Fluorescence channels",
            CHANNEL_HEADERS,
            [_channel_row(c) for c in _dicts(working.get("channel_settings"))],
        ),
        (
            "Recent channels",
            CHANNEL_HEADERS,
            [_channel_row(c) for c in _dicts(fm.get("recent_channels"))],
        ),
    ]
    known = {"saved_positions", "holder_occupancy", "fm"}
    unknown = [
        (name, type(value).__name__)
        for name, value in sections.items()
        if name not in known
    ]
    if unknown:
        result.append(("Not used by this version", ("Section", "Holds"), unknown))
    return result


def _ago(seconds: float) -> str:
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n > 1 else ''} ago"
    return "just now"


def _show_in_folder(path) -> None:
    from PyQt5.QtCore import QUrl
    from PyQt5.QtGui import QDesktopServices

    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def session_tab(microscope: FibsemMicroscope) -> QWidget:
    import time

    from fibsem.session_state import session_state_for

    state = session_state_for(microscope)
    path = state.path

    head = QHBoxLayout()
    if path is None:
        head.addWidget(_value("None: this session was not started from a file"))
    else:
        head.addWidget(_value(path.name))
        head.addWidget(
            _muted(
                f"written {_ago(time.time() - path.stat().st_mtime)}"
                if path.exists()
                else "nothing saved yet"
            )
        )
        head.addStretch()
        show = QPushButton("Show in Folder")
        show.setObjectName("show_session_state")
        show.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        show.setToolTip(str(path))
        show.clicked.connect(lambda: _show_in_folder(path.parent))
        head.addWidget(show)

    # Positions and grids on the left, fluorescence on the right.
    columns = [QVBoxLayout(), QVBoxLayout()]
    for i, (title, headers, rows) in enumerate(session_sections(state.sections())):
        if rows:
            content = _table(headers, rows, stretch=len(headers) - 1)
            title = f"{title}  ·  {len(rows)}"
        else:
            content = _muted("Nothing saved")
            content.setContentsMargins(8, 6, 8, 6)
        columns[0 if i < 2 else 1].addWidget(_panel(title, content))
    grid = QHBoxLayout()
    for column in columns:
        column.addStretch()
        grid.addLayout(column, 1)

    widget = QWidget()
    layout = QVBoxLayout(widget)
    layout.addLayout(head)
    layout.addWidget(
        _muted(
            "What the application remembers for this configuration between sessions. "
            "Each section is changed where it is used -- the saved positions panel, "
            "the sample holder, the fluorescence tab."
        )
    )
    layout.addLayout(grid, 1)
    return widget


# ---- the window -----------------------------------------------------------------------


DEFAULTS = "Defaults"


class MicroscopeConfigurationWindow(QDialog):
    """The connected microscope's configuration, by kind. Non-modal.

    Everything is read-only except the Defaults tab, which the window's one Save
    writes into the configuration file.
    """

    def __init__(
        self,
        microscope: FibsemMicroscope,
        parent: Optional[QWidget] = None,
        image_settings: Optional[ImageSettings] = None,
        current_imaging: Optional[Callable[[], ImageSettings]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Microscope Configuration")
        self.setModal(False)
        self.microscope = microscope
        self._closing_without_asking = False

        info = microscope.system.info
        title = QLabel("Microscope Configuration")
        title.setStyleSheet(
            f"color: {TEXT_STRONG_COLOR}; font-size: 16px; font-weight: bold;"
        )
        texts = QVBoxLayout()
        texts.addWidget(title)
        texts.addWidget(
            _muted(
                f"{info.name}  ·  {info.manufacturer} {info.model}  ·  {info.ip_address}"
            )
        )
        head = QHBoxLayout()
        head.addLayout(texts, 1)
        head.addWidget(_status("Connected", OK_COLOR), alignment=Qt.AlignTop)

        self.defaults = MicroscopeDefaultsWidget()
        self.defaults.set_microscope(microscope, image_settings=image_settings)
        self.defaults.set_current_imaging(current_imaging)

        self.tabs = QTabWidget()
        self.tabs.addTab(instrument_tab(microscope), "Instrument")
        self.tabs.addTab(geometry_tab(microscope), "Geometry")
        self.tabs.addTab(self._calibration_tab(), "Calibration")
        self.tabs.addTab(self.defaults, DEFAULTS)
        self.tabs.addTab(session_tab(microscope), "Session")
        self._calibration_dialog = None

        path = getattr(microscope, "configuration_path", None)
        self._file_name = os.path.basename(path) if path else ""
        self.label_unsaved = _status("", WARN_COLOR)
        self.pushButton_apply = QPushButton("Apply to Microscope")
        self.pushButton_apply.setToolTip(
            "Set the columns and detectors to the defaults on the Defaults tab, "
            "saved or not. The beams should be on."
        )
        self.pushButton_apply.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        self.pushButton_apply.setEnabled(cfg.APPLY_CONFIGURATION_ENABLED)
        self.pushButton_save = QPushButton("Save")
        self.pushButton_save.setToolTip(
            "Write the defaults into the configuration file this session was started "
            "from. Nothing else in the file is changed."
        )
        self.pushButton_save.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.pushButton_close = QPushButton("Close")
        self.pushButton_close.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        foot = QHBoxLayout()
        foot.addWidget(self.label_unsaved)
        foot.addStretch()
        for button in (
            self.pushButton_apply,
            self.pushButton_save,
            self.pushButton_close,
        ):
            foot.addWidget(button)

        layout = QVBoxLayout(self)
        layout.addLayout(head)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(foot)
        self.setStyleSheet(f"QDialog {{ background: {SURFACE_COLOR}; }}")
        self.resize(1040, 560)

        self.defaults.changed.connect(self._show_unsaved)
        self.pushButton_apply.clicked.connect(self.apply_to_microscope)
        self.pushButton_save.clicked.connect(self.save)
        self.pushButton_close.clicked.connect(self.close)
        self._show_unsaved()

    # -- saving -----------------------------------------------------------------------

    def has_unsaved_changes(self) -> bool:
        return self.defaults.is_modified()

    def _show_unsaved(self) -> None:
        unsaved = self.has_unsaved_changes()
        where = f"  ·  {self._file_name}" if self._file_name else ""
        self.label_unsaved.setText(
            f"Unsaved changes  ·  {DEFAULTS}{where}" if unsaved else ""
        )
        self.pushButton_save.setEnabled(unsaved)
        index = self.tabs.indexOf(self.defaults)
        self.tabs.setTabText(index, f"{DEFAULTS} •" if unsaved else DEFAULTS)

    def save(self) -> bool:
        """Write the window's changes into the configuration. Returns whether saved."""
        return self.defaults.save_to_configuration()

    def apply_to_microscope(self) -> None:
        """Set the instrument to the defaults the Defaults tab shows."""
        self.defaults.write_form_into_system()
        try:
            self.microscope.apply_configuration()
        except Exception as e:  # a slot: nothing may reach Qt
            logging.error(f"Could not apply the defaults: {e}")
            notification_service.show_toast(
                f"Could not apply the defaults: {e}", "error"
            )
            return
        notification_service.show_toast("Defaults applied to the microscope.", "info")

    def reject(self) -> None:
        """Close, Escape and the title bar all come here: offer to save first."""
        if self._closing_without_asking or not self.has_unsaved_changes():
            super().reject()
            return
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            "The defaults have changes that are not saved to the configuration.",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            return
        if answer == QMessageBox.Save and not self.save():
            return  # the save said why; stay open
        super().reject()

    def close_without_asking(self) -> None:
        """Close now, dropping unsaved changes: the microscope went away."""
        self._closing_without_asking = True
        self.close()

    # -- calibration ------------------------------------------------------------------

    def _calibration_tab(self) -> QWidget:
        # A holder the loader fills is calibrated by construction; there are no
        # slots to capture.
        loader = getattr(self.microscope._stage, "loader", None)
        return calibration_tab(
            self.microscope, None if loader is not None else self.calibrate_slots
        )

    def calibrate_slots(self) -> None:
        """Open the holder's guided calibration; this tab follows when it saves."""
        from fibsem.ui.widgets.holder_calibration_dialog import (
            HolderCalibrationDialog,
        )

        dialog = HolderCalibrationDialog(
            self.microscope, self.microscope._stage.holder, parent=self
        )
        dialog.holder_saved.connect(self.refresh_calibration)
        self._calibration_dialog = dialog  # non-modal; kept alive here
        dialog.show()

    def refresh_calibration(self, *_) -> None:
        index = next(
            i for i in range(self.tabs.count()) if self.tabs.tabText(i) == "Calibration"
        )
        current = self.tabs.currentIndex()
        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self.tabs.insertTab(index, self._calibration_tab(), "Calibration")
        self.tabs.setCurrentIndex(current)
        old.deleteLater()

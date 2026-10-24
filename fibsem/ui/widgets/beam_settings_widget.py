import numpy as np
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QWidget,
)
from superqt.utils import qdebounced

from fibsem import utils
from fibsem.devices.beam import STANDARD_RESOLUTIONS, Beam
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.params import Parameter
from fibsem.structures.settings import BeamSettings, BeamType, Point, Resolution
from fibsem.ui import notification_service
from fibsem.ui.control import control_of, operator_acts
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.utils import find_data, install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import _create_combobox_control
from fibsem.ui.widgets.form_builder import (
    configure_spinbox,
    parameter_spec,
)
from fibsem.workflows.actions import record_operator_action

# Working-distance step per Shift+scroll notch (mm). 1 um — fine focus control.
WD_WHEEL_STEP_MM = 0.001

# The spin boxes, by beam parameter. Label, unit, scale, step, decimals and range all
# come from the parameter: its declared display hint, and the limits the beam reports.
SPINBOX_PARAMETERS = ("hfw", "dwell_time", "working_distance", "scan_rotation")
POINT_PARAMETERS = ("shift", "stigmation")


class FibsemBeamSettingsWidget(QWidget):
    settings_changed = pyqtSignal(BeamSettings)

    def __init__(
        self,
        microscope: FibsemMicroscope,
        beam_type: BeamType,
        parent=None,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.beam_type = beam_type
        self._advanced_visible = False
        self._setup_ui()
        self._connect_signals()
        self._update_visibility()

        # Shift+scroll working-distance nudge (mirrors the FM objective wheel): the spinbox
        # updates immediately for feedback, the hardware move is debounced so a burst of
        # scroll notches coalesces into a single move.
        self._wd_wheel_target_mm: float = self.working_distance_spinbox.value()
        self._execute_wd_wheel_move = qdebounced(
            self._execute_wd_wheel_move_impl, timeout=150
        )

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        def _make_spinbox() -> QDoubleSpinBox:
            sb = QDoubleSpinBox()
            sb.setKeyboardTracking(False)
            install_wheel_blocker(sb)
            return sb

        def _make_label(name: str, suffix: str = "") -> QLabel:
            return QLabel(self._field(name).label + suffix)

        def _make_point_row(name: str):
            x, y = _make_spinbox(), _make_spinbox()
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(x)
            row_layout.addWidget(y)
            return x, y, row, _make_label(name, " X / Y")

        # --- Field of View ---
        self.hfw_spinbox = _make_spinbox()
        self.hfw_label = _make_label("hfw")
        layout.addRow(self.hfw_label, self.hfw_spinbox)

        # --- Dwell Time ---
        self.dwell_time_spinbox = _make_spinbox()
        self.dwell_time_label = _make_label("dwell_time")
        layout.addRow(self.dwell_time_label, self.dwell_time_spinbox)

        # --- Resolution ---
        self.resolution_combo = QComboBox()
        for width, height in STANDARD_RESOLUTIONS:
            self.resolution_combo.addItem(
                str(Resolution(width, height)), (width, height)
            )
        self.resolution_combo.setCurrentIndex(
            find_data(self.resolution_combo, (1536, 1024))
        )
        install_wheel_blocker(self.resolution_combo)
        self.resolution_label = _make_label("resolution")
        layout.addRow(self.resolution_label, self.resolution_combo)

        # --- Beam Current ---
        self.beam_current_combo = QComboBox()
        install_wheel_blocker(self.beam_current_combo)
        self.beam_current_label = _make_label("current")
        layout.addRow(self.beam_current_label, self.beam_current_combo)

        # --- Beam Voltage ---
        self.beam_voltage_combo = QComboBox()
        install_wheel_blocker(self.beam_voltage_combo)
        self.beam_voltage_label = _make_label("voltage")
        layout.addRow(self.beam_voltage_label, self.beam_voltage_combo)

        # --- Preset ---
        self.preset_combo = QComboBox()
        install_wheel_blocker(self.preset_combo)
        self.preset_label = _make_label("preset")
        layout.addRow(self.preset_label, self.preset_combo)

        # --- Working Distance ---
        self.working_distance_spinbox = _make_spinbox()
        self.working_distance_label = _make_label("working_distance")
        layout.addRow(self.working_distance_label, self.working_distance_spinbox)

        # --- Scan Rotation ---
        self.scan_rotation_spinbox = _make_spinbox()
        self.scan_rotation_label = _make_label("scan_rotation")
        layout.addRow(self.scan_rotation_label, self.scan_rotation_spinbox)

        # --- Shift X / Y ---
        (
            self.shift_x_spinbox,
            self.shift_y_spinbox,
            self.shift_row,
            self.shift_label,
        ) = _make_point_row("shift")
        layout.addRow(self.shift_label, self.shift_row)

        # --- Stigmation X / Y ---
        (
            self.stigmation_x_spinbox,
            self.stigmation_y_spinbox,
            self.stigmation_row,
            self.stigmation_label,
        ) = _make_point_row("stigmation")
        layout.addRow(self.stigmation_label, self.stigmation_row)

        # All widgets that are shown only when advanced mode is active
        self._adv_widgets = [
            self.scan_rotation_label,
            self.scan_rotation_spinbox,
            self.shift_label,
            self.shift_row,
            self.stigmation_label,
            self.stigmation_row,
            self.beam_voltage_label,
            self.beam_voltage_combo,
        ]
        self._apply_display()

    # ------------------------------------------------------------------
    # Signal connections
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self.hfw_spinbox.valueChanged.connect(self._on_hfw_changed)
        self.dwell_time_spinbox.valueChanged.connect(self._on_dwell_time_changed)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.beam_current_combo.currentIndexChanged.connect(
            self._on_beam_current_changed
        )
        self.beam_voltage_combo.currentIndexChanged.connect(
            self._on_beam_voltage_changed
        )
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        self.working_distance_spinbox.valueChanged.connect(
            self._on_working_distance_changed
        )
        self.scan_rotation_spinbox.valueChanged.connect(self._on_scan_rotation_changed)
        self.shift_x_spinbox.valueChanged.connect(self._on_shift_changed)
        self.shift_y_spinbox.valueChanged.connect(self._on_shift_changed)
        self.stigmation_x_spinbox.valueChanged.connect(self._on_stigmation_changed)
        self.stigmation_y_spinbox.valueChanged.connect(self._on_stigmation_changed)

    # ------------------------------------------------------------------
    # Live-update handlers
    # ------------------------------------------------------------------

    def _on_hfw_changed(self, value: float):
        hfw = self._to_si("hfw", value)
        if not self._set("field width", self.microscope.set_field_of_view, hfw):
            return
        self._record("field width", "hfw", hfw)
        self.settings_changed.emit(self.get_settings())

    def _on_dwell_time_changed(self, value: float):
        dwell_time = self._to_si("dwell_time", value)
        if not self._set("dwell time", self.microscope.set_dwell_time, dwell_time):
            return
        self._record("dwell time", "dwell_time", dwell_time)
        self.settings_changed.emit(self.get_settings())

    def _on_resolution_changed(self, index: int):
        resolution = self.resolution_combo.itemData(index)
        if resolution is not None:
            if not self._set("resolution", self.microscope.set_resolution, resolution):
                return
            self._record("resolution", "resolution", list(resolution))
            self.settings_changed.emit(self.get_settings())

    def _on_beam_current_changed(self, index: int):
        current = self.beam_current_combo.itemData(index)
        if current is not None:
            if not self._set("beam current", self.microscope.set_beam_current, current):
                return
            self._record("beam current", "current", current)
            self.settings_changed.emit(self.get_settings())

    def _on_beam_voltage_changed(self, index: int):
        voltage = self.beam_voltage_combo.itemData(index)
        if voltage is not None:
            if not self._set("beam voltage", self.microscope.set_beam_voltage, voltage):
                return
            self._populate_currents()  # the current choices follow the voltage
            self._record("beam voltage", "voltage", voltage)
            self.settings_changed.emit(self.get_settings())

    def _on_preset_changed(self, index: int):
        preset = self.preset_combo.itemData(index)
        if preset is None:
            return
        # Preset activation is a multi-second SharkSEM round trip (with the settle-wait
        # from #82), so it must not run on the GUI thread -- and it can fail: the
        # simulator enumerates SEM presets that Activate then refuses (PresetNotFound),
        # which must surface as a toast rather than an exception in a Qt slot.
        self.preset_combo.setEnabled(False)
        worker = FunctionWorker(self.microscope.set_preset, preset, self.beam_type)
        worker.returned.connect(lambda _result, p=preset: self._on_preset_applied(p))
        worker.errored.connect(lambda exc, p=preset: self._on_preset_failed(p, exc))
        worker.finished.connect(lambda: self.preset_combo.setEnabled(True))
        worker.start(control=control_of(self.microscope), action="set beam preset")

    def _on_preset_applied(self, preset: str) -> None:
        self._record("preset", "preset", preset)
        self.settings_changed.emit(self.get_settings())

    def _on_preset_failed(self, preset: str, exc: Exception) -> None:
        """Toast the failure and put the combo back on the preset the microscope reports.

        On Tescan ``get_preset`` is served from the cached beam parameters, so the
        revert makes no SDK call. If nothing was ever activated, clear the selection --
        leaving the refused preset displayed would show state the microscope is not in.
        """
        notification_service.show_toast(
            f"Failed to activate preset {preset}: {exc}", "error"
        )
        current = self.microscope.get_preset(self.beam_type)
        self.preset_combo.blockSignals(True)
        idx = self.preset_combo.findData(current) if current is not None else -1
        self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

    def _on_working_distance_changed(self, value: float):
        wd = self._to_si("working_distance", value)
        if not self._set("working distance", self.microscope.set_working_distance, wd):
            return
        self._record("working distance", "working_distance", wd)
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Working-distance mouse-wheel nudge (Shift + scroll on the beam canvas)
    # ------------------------------------------------------------------

    def _on_canvas_scroll(self, x: float, y: float, direction: int, modifiers) -> None:
        """Shift+scroll on this beam's canvas nudges the working distance by one spinbox step;
        plain scroll is left to the canvas (zoom). The hardware move is debounced so a burst of
        notches coalesces into one move. Unlike the FM objective, WD (beam focus) can be adjusted
        live while scanning — that's how you focus — so there is deliberately no acquisition lockout."""
        if "Shift" not in modifiers:
            return
        sb = self.working_distance_spinbox
        old_val = sb.value()
        new_val = float(
            np.clip(old_val + WD_WHEEL_STEP_MM * direction, sb.minimum(), sb.maximum())
        )
        # immediate visual feedback without a hardware call; debounce the actual move
        sb.blockSignals(True)
        sb.setValue(new_val)
        sb.blockSignals(False)
        self._wd_wheel_target_mm = new_val
        # transient flash on the canvas that emitted the scroll (fades after scrolling
        # stops): the target and the step, in the bar's words. Three decimals,
        # where the bar shows two: a notch is 1 µm, and each one should be seen.
        canvas = self.sender()
        if canvas is not None and hasattr(canvas, "flash_message"):
            step_um = (new_val - old_val) * 1e3
            canvas.flash_message(f"WD {new_val:.3f} mm  {step_um:+.0f} µm")
        self._execute_wd_wheel_move()

    def _execute_wd_wheel_move_impl(self) -> None:
        """Apply the settled Shift+scroll target to hardware. No large-change confirmation —
        WD is beam focus (lens), not a physical objective, so a big move isn't a collision risk."""
        target_m = self._to_si("working_distance", self._wd_wheel_target_mm)
        if not self._set(
            "working distance", self.microscope.set_working_distance, target_m
        ):
            return
        self._record(
            "working distance", "working_distance", target_m, via="Shift+scroll"
        )
        self._set_working_distance_spinbox(
            self.microscope.get_working_distance(self.beam_type)
        )
        self.settings_changed.emit(self.get_settings())

    def _set(self, label: str, setter, value) -> bool:
        """Write ``value`` holding control. Refused, the controls show
        the beam's values again."""
        action = f"set the {self.beam_type.name} {label}"
        if operator_acts(
            control_of(self.microscope), action, setter, value, self.beam_type
        ):
            return True
        self.update_from_settings(self.microscope.get_beam_settings(self.beam_type))
        return False

    def _record(self, label: str, setting: str, value, via: str = "Beam settings"):
        record_operator_action(
            self.microscope,
            f"Set the {self.beam_type.name} {label}",
            via=via,
            beam=self.beam_type.name,
            setting=setting,
            value=value,
        )

    def _set_working_distance_spinbox(self, wd_m: float) -> None:
        """Set the WD spinbox from a value in metres without triggering a hardware move."""
        self.working_distance_spinbox.blockSignals(True)
        self.working_distance_spinbox.setValue(self._shown("working_distance", wd_m))
        self.working_distance_spinbox.blockSignals(False)

    def _on_scan_rotation_changed(self, value: float):
        rotation = self._to_si("scan_rotation", value)
        if not self._set("scan rotation", self.microscope.set_scan_rotation, rotation):
            return
        self._record("scan rotation", "scan_rotation", rotation)
        self.settings_changed.emit(self.get_settings())

    def _on_shift_changed(self):
        shift = self._read_point("shift")
        if not self._set("beam shift", self.microscope.set_beam_shift, shift):
            return
        self._record("beam shift", "shift", shift.to_dict())
        self.settings_changed.emit(self.get_settings())

    def _on_stigmation_changed(self):
        stigmation = self._read_point("stigmation")
        if not self._set("stigmation", self.microscope.set_stigmation, stigmation):
            return
        self._record("stigmation", "stigmation", stigmation.to_dict())
        self.settings_changed.emit(self.get_settings())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def populate_beam_combos(self):
        """Populate beam current and voltage comboboxes from the microscope.

        Call this once after construction (or whenever the beam type changes)
        so the comboboxes contain the correct available values. A beam the
        microscope has not got has no values to list.
        """
        if self._beam_device() is None:
            return
        self._populate_resolutions()
        self._apply_display()

        self._populate_currents()

        self.beam_voltage_combo.blockSignals(True)
        self.beam_voltage_combo.clear()
        voltage = self.microscope.get_beam_voltage(self.beam_type)
        _create_combobox_control(
            value=voltage,
            items=self._combo_items("voltage", voltage),
            units="V",
            format_fn=utils.format_value,
            control=self.beam_voltage_combo,
        )
        self.beam_voltage_combo.blockSignals(False)

        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        presets = self._choices("preset")
        if presets:
            for preset in presets:
                self.preset_combo.addItem(str(preset), str(preset))
            current = self.microscope.get_preset(self.beam_type)
            if current is not None:
                idx = self.preset_combo.findData(current)
                if idx != -1:
                    self.preset_combo.setCurrentIndex(idx)
        self.preset_combo.blockSignals(False)

        # whether this beam has presets is only known once they are populated
        # (Tescan exposes them on the FIB but not the SEM), so re-apply visibility
        self._update_visibility()

    def _populate_currents(self):
        """The current combo from the beam's current choices, at its current value."""
        self.beam_current_combo.blockSignals(True)
        self.beam_current_combo.clear()
        current = self.microscope.get_beam_current(self.beam_type)
        _create_combobox_control(
            value=current,
            items=self._combo_items("current", current),
            units="A",
            format_fn=utils.format_value,
            control=self.beam_current_combo,
        )
        self.beam_current_combo.blockSignals(False)

    def _populate_resolutions(self) -> None:
        """List the beam's resolutions, keeping the selection; a beam that lists none
        keeps the standard ones."""
        choices = self._choices("resolution")
        if not choices:
            return
        selected = self.resolution_combo.currentData()
        self.resolution_combo.blockSignals(True)
        self.resolution_combo.clear()
        for width, height in choices:
            self.resolution_combo.addItem(
                str(Resolution(width, height)), (width, height)
            )
        idx = find_data(self.resolution_combo, selected)
        if idx != -1:
            self.resolution_combo.setCurrentIndex(idx)
        self.resolution_combo.blockSignals(False)

    def _apply_display(self) -> None:
        """Set each box's suffix, step, decimals and range from its beam parameter,
        with the limits the beam reports where it reports them."""
        for name in SPINBOX_PARAMETERS:
            spinbox = self._spinbox(name)
            spinbox.blockSignals(True)
            configure_spinbox(spinbox, self._field(name))
            spinbox.blockSignals(False)
        for name in POINT_PARAMETERS:
            for axis in ("x", "y"):
                spinbox = self._spinbox(name, axis)
                spinbox.blockSignals(True)
                configure_spinbox(spinbox, self._field(name, axis))
                spinbox.blockSignals(False)

    def get_settings(self) -> BeamSettings:
        """Return a BeamSettings built from the current widget values."""
        return BeamSettings(
            beam_type=self.beam_type,
            working_distance=self._to_si(
                "working_distance", self.working_distance_spinbox.value()
            ),
            hfw=self._to_si("hfw", self.hfw_spinbox.value()),
            dwell_time=self._to_si("dwell_time", self.dwell_time_spinbox.value()),
            resolution=self.resolution_combo.currentData(),
            beam_current=self.beam_current_combo.currentData(),
            voltage=self.beam_voltage_combo.currentData(),
            preset=self.preset_combo.currentData(),
            scan_rotation=self._to_si(
                "scan_rotation", self.scan_rotation_spinbox.value()
            ),
            shift=self._read_point("shift"),
            stigmation=self._read_point("stigmation"),
        )

    def update_from_settings(self, settings: BeamSettings):
        """Populate all controls from a BeamSettings object without triggering live updates."""
        widgets = [
            self.working_distance_spinbox,
            self.hfw_spinbox,
            self.dwell_time_spinbox,
            self.resolution_combo,
            self.beam_current_combo,
            self.beam_voltage_combo,
            self.preset_combo,
            self.scan_rotation_spinbox,
            self.shift_x_spinbox,
            self.shift_y_spinbox,
            self.stigmation_x_spinbox,
            self.stigmation_y_spinbox,
        ]
        for w in widgets:
            w.blockSignals(True)

        if settings.working_distance is not None:
            self.working_distance_spinbox.setValue(
                self._shown("working_distance", settings.working_distance)
            )
        if settings.hfw is not None:
            self.hfw_spinbox.setValue(self._shown("hfw", settings.hfw))
        if settings.dwell_time is not None:
            self.dwell_time_spinbox.setValue(
                self._shown("dwell_time", settings.dwell_time)
            )
        if settings.resolution is not None:
            idx = find_data(self.resolution_combo, tuple(settings.resolution))
            if idx != -1:
                self.resolution_combo.setCurrentIndex(idx)
        if settings.beam_current is not None:
            self._set_combo_closest(self.beam_current_combo, settings.beam_current)
        if settings.voltage is not None:
            self._set_combo_closest(self.beam_voltage_combo, settings.voltage)
        if settings.preset is not None:
            idx = self.preset_combo.findData(settings.preset)
            if idx != -1:
                self.preset_combo.setCurrentIndex(idx)
        if settings.scan_rotation is not None:
            self.scan_rotation_spinbox.setValue(
                self._shown("scan_rotation", settings.scan_rotation)
            )
        if settings.shift is not None:
            self._write_point("shift", settings.shift)
        if settings.stigmation is not None:
            self._write_point("stigmation", settings.stigmation)

        for w in widgets:
            w.blockSignals(False)

    def set_advanced_visible(self, show: bool):
        """Show or hide advanced controls."""
        self._advanced_visible = show
        self._update_visibility()

    def _beam_device(self):
        """This beam's device, or None when the microscope has no such beam."""
        return self.microscope.beams.get(self.beam_type)

    def _beam_parameter(self, name: str):
        """The beam device's parameter, or None when the beam does not have it."""
        beam = self._beam_device()
        return None if beam is None else beam.parameters.get(name)

    def _combo_items(self, key: str, value) -> list:
        """The choices for a combo: the available values when the beam can set
        ``key``, and only the value it reads when the beam reports it read-only, so
        the control shows what the beam is at rather than snapping to a choice."""
        parameter = self._beam_parameter(key)
        if parameter is not None and not parameter.settable:
            return [] if value is None else [value]
        return self._choices(key)

    def _field(self, name: str, field=None) -> Parameter:
        """How to show the beam parameter ``name`` (one ``field`` of a point): its
        display hint, with the limits and choices this beam reports. With no such
        beam, or a beam without the parameter, what every beam declares."""
        parameter = self._beam_parameter(name) or getattr(Beam, name)
        return parameter_spec(parameter, field)

    def _choices(self, name: str) -> list:
        return self._field(name).choices or []

    def _scale(self, name: str, field=None) -> float:
        return self._field(name, field).scale or 1.0

    def _to_si(self, name: str, shown: float, field=None) -> float:
        """A value as the box shows it, in the parameter's SI unit."""
        return shown / self._scale(name, field)

    def _shown(self, name: str, value: float, field=None) -> float:
        """An SI value as the box shows it."""
        return value * self._scale(name, field)

    def _spinbox(self, name: str, axis: str = "") -> QDoubleSpinBox:
        return getattr(self, f"{name}_{axis}_spinbox" if axis else f"{name}_spinbox")

    def _read_point(self, name: str) -> Point:
        return Point(
            x=self._to_si(name, self._spinbox(name, "x").value(), "x"),
            y=self._to_si(name, self._spinbox(name, "y").value(), "y"),
        )

    def _write_point(self, name: str, point: Point) -> None:
        self._spinbox(name, "x").setValue(self._shown(name, point.x, "x"))
        self._spinbox(name, "y").setValue(self._shown(name, point.y, "y"))

    def _update_visibility(self):
        """Apply visibility from the beam device's parameters and advanced mode.

        A control is hidden when the beam has no such parameter, and shown read-only
        when the beam reports it not settable. A beam the microscope has not got (a
        disabled column) has none of them."""
        adv = self._advanced_visible

        for w in self._adv_widgets:
            w.setVisible(adv)

        def apply(name, widgets, advanced=False):
            parameter = self._beam_parameter(name)
            shown = parameter is not None and (adv or not advanced)
            settable = parameter is not None and parameter.settable
            for w in widgets:
                w.setVisible(shown)
                w.setEnabled(settable)
            return parameter

        apply("stigmation", [self.stigmation_label, self.stigmation_row], advanced=True)
        apply(
            "voltage", [self.beam_voltage_label, self.beam_voltage_combo], advanced=True
        )
        apply("current", [self.beam_current_label, self.beam_current_combo])

        # An empty preset combo also hides (it reads as a control the user failed to set).
        preset = self._beam_parameter("preset")
        show_preset = preset is not None and self.preset_combo.count() > 0
        for w in [self.preset_label, self.preset_combo]:
            w.setVisible(show_preset)

        wd = apply(
            "working_distance",
            [self.working_distance_label, self.working_distance_spinbox],
        )
        self.working_distance_spinbox.setToolTip(
            "" if wd is None or wd.settable else "Not settable on this beam"
        )

    @staticmethod
    def _set_combo_closest(combo, value: float):
        """Set the combobox to the item whose stored data is closest to value."""
        idx = combo.findData(value)
        if idx == -1:
            items = [combo.itemData(i) for i in range(combo.count())]
            if items:
                idx = combo.findData(min(items, key=lambda x: abs(x - value)))
        if idx != -1:
            combo.setCurrentIndex(idx)

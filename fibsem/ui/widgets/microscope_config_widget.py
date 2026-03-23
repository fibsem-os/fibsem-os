"""Microscope configuration widget — pure PyQt5, no napari dependency."""

from __future__ import annotations

import logging
import os
from typing import Optional

from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from fibsem import config as cfg
from fibsem import constants, utils
from fibsem.structures import BeamType


class MicroscopeConfigWidget(QWidget):
    """Tabbed widget to view and edit a microscope configuration YAML file.

    Tabs: Info | Stage | Electron | Ion | Imaging | Milling | Subsystems | Sim

    Public API:
        set_config(config: dict)   — populate all fields from a config dict
        get_config() -> dict       — read all fields back to a config dict
        load_from_file(path: str)  — load YAML file and call set_config
        save_to_file(path: str)    — call get_config and write YAML file
    """

    def __init__(self, path: Optional[str] = None, parent=None) -> None:
        super().__init__(parent)
        self._path: Optional[str] = path
        self._config: dict = {}
        self._setup_ui()
        if path is not None:
            self.load_from_file(path)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        self._tabs.addTab(self._build_info_tab(), "Info")
        self._tabs.addTab(self._build_stage_tab(), "Stage")
        self._tabs.addTab(self._build_beam_tab("electron"), "Electron")
        self._tabs.addTab(self._build_beam_tab("ion"), "Ion")
        self._tabs.addTab(self._build_imaging_tab(), "Imaging")
        self._tabs.addTab(self._build_milling_tab(), "Milling")
        self._tabs.addTab(self._build_subsystems_tab(), "Subsystems")
        self._tabs.addTab(self._build_sim_tab(), "Sim")

        # Bottom toolbar
        toolbar = QHBoxLayout()
        btn_load = QPushButton("Load YAML")
        btn_save = QPushButton("Save YAML")
        btn_load.clicked.connect(self._on_load)
        btn_save.clicked.connect(self._on_save)
        toolbar.addWidget(btn_load)
        toolbar.addWidget(btn_save)
        toolbar.addStretch(1)
        outer.addLayout(toolbar)

        # manufacturer change drives milling field visibility
        self._cb_manufacturer.currentTextChanged.connect(self._on_manufacturer_changed)

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _build_info_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._le_name = QLineEdit()
        self._le_ip = QLineEdit()
        self._cb_manufacturer = QComboBox()
        self._cb_manufacturer.addItems(cfg.AVAILABLE_MANUFACTURERS)
        form.addRow("Name", self._le_name)
        form.addRow("IP Address", self._le_ip)
        form.addRow("Manufacturer", self._cb_manufacturer)
        return w

    def _build_stage_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._sb_stage_rot_ref = QSpinBox(); self._sb_stage_rot_ref.setRange(-360, 360)
        self._sb_stage_rot_180 = QSpinBox(); self._sb_stage_rot_180.setRange(-360, 360)
        self._sb_stage_pretilt = QSpinBox(); self._sb_stage_pretilt.setRange(0, 90)
        self._dsb_stage_manip_limit = _dsb(0, 100, 4, suffix=" mm")
        form.addRow("Rotation Reference (deg)", self._sb_stage_rot_ref)
        form.addRow("Rotation 180 (deg)", self._sb_stage_rot_180)
        form.addRow("Shuttle Pre-tilt (deg)", self._sb_stage_pretilt)
        form.addRow("Manipulator Height Limit (mm)", self._dsb_stage_manip_limit)
        return w

    def _build_beam_tab(self, beam: str) -> QWidget:
        """Build the Electron or Ion tab. `beam` is 'electron' or 'ion'."""
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 0, 0, 0)

        inner = QWidget()
        form = QFormLayout(inner)

        sb_col_tilt = QSpinBox(); sb_col_tilt.setRange(-90, 90)
        dsb_euc = _dsb(0, 100, 4, suffix=" mm")
        dsb_voltage = _dsb(0, 100000, 0, suffix=" V")
        dsb_current = _dsb(0, 1e6, 4, suffix=" nA")
        sb_res_x = QSpinBox(); sb_res_x.setRange(64, 16384)
        sb_res_y = QSpinBox(); sb_res_y.setRange(64, 16384)
        dsb_hfw = _dsb(0, 100000, 2, suffix=" µm")
        dsb_dwell = _dsb(0, 10000, 4, suffix=" µs")
        le_det_mode = QLineEdit()
        le_det_type = QLineEdit()

        form.addRow("Column Tilt (deg)", sb_col_tilt)
        form.addRow("Eucentric Height (mm)", dsb_euc)
        form.addRow("Voltage (V)", dsb_voltage)
        form.addRow("Current (nA)", dsb_current)
        form.addRow("Resolution X (px)", sb_res_x)
        form.addRow("Resolution Y (px)", sb_res_y)
        form.addRow("HFW (µm)", dsb_hfw)
        form.addRow("Dwell Time (µs)", dsb_dwell)
        form.addRow("Detector Mode", le_det_mode)
        form.addRow("Detector Type", le_det_type)

        fields = {
            "col_tilt": sb_col_tilt, "euc": dsb_euc,
            "voltage": dsb_voltage, "current": dsb_current,
            "res_x": sb_res_x, "res_y": sb_res_y,
            "hfw": dsb_hfw, "dwell": dsb_dwell,
            "det_mode": le_det_mode, "det_type": le_det_type,
        }

        if beam == "ion":
            cb_plasma = QCheckBox()
            le_plasma_gas = QLineEdit()
            form.addRow("Plasma FIB", cb_plasma)
            form.addRow("Plasma Gas", le_plasma_gas)
            fields["plasma"] = cb_plasma
            fields["plasma_gas"] = le_plasma_gas

        vbox.addWidget(self._scrollable(inner))
        setattr(self, f"_beam_{beam}", fields)
        return container

    def _build_imaging_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self._cb_imaging_beam_type = QComboBox()
        self._cb_imaging_beam_type.addItems([b.name for b in BeamType])
        self._sb_imaging_res_x = QSpinBox(); self._sb_imaging_res_x.setRange(64, 16384)
        self._sb_imaging_res_y = QSpinBox(); self._sb_imaging_res_y.setRange(64, 16384)
        self._dsb_imaging_hfw = _dsb(0, 100000, 2, suffix=" µm")
        self._dsb_imaging_dwell = _dsb(0, 10000, 4, suffix=" µs")
        self._dsb_imaging_current = _dsb(0, 1e6, 4, suffix=" nA")
        self._chk_autocontrast = QCheckBox()
        self._chk_autogamma = QCheckBox()
        self._chk_autosave = QCheckBox()
        form.addRow("Beam Type", self._cb_imaging_beam_type)
        form.addRow("Resolution X (px)", self._sb_imaging_res_x)
        form.addRow("Resolution Y (px)", self._sb_imaging_res_y)
        form.addRow("HFW (µm)", self._dsb_imaging_hfw)
        form.addRow("Dwell Time (µs)", self._dsb_imaging_dwell)
        form.addRow("Imaging Current (nA)", self._dsb_imaging_current)
        form.addRow("Autocontrast", self._chk_autocontrast)
        form.addRow("Autogamma", self._chk_autogamma)
        form.addRow("Auto-save", self._chk_autosave)
        return w

    def _build_milling_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        # Thermo fields
        self._grp_thermo = QGroupBox("Thermo Fisher")
        thermo_form = QFormLayout(self._grp_thermo)
        self._dsb_mill_voltage = _dsb(0, 100000, 0, suffix=" V")
        self._dsb_mill_current = _dsb(0, 1e6, 4, suffix=" nA")
        thermo_form.addRow("Milling Voltage (V)", self._dsb_mill_voltage)
        thermo_form.addRow("Milling Current (nA)", self._dsb_mill_current)

        # Tescan fields
        self._grp_tescan = QGroupBox("Tescan")
        tescan_form = QFormLayout(self._grp_tescan)
        self._dsb_mill_dwell = _dsb(0, 10000, 4, suffix=" µs")
        self._dsb_mill_rate = _dsb(0, 1e6, 6, suffix=" µm³/s")
        self._dsb_mill_spot = _dsb(0, 1e6, 4, suffix=" nm")
        self._le_mill_preset = QLineEdit()
        tescan_form.addRow("Dwell Time (µs)", self._dsb_mill_dwell)
        tescan_form.addRow("Rate (µm³/s)", self._dsb_mill_rate)
        tescan_form.addRow("Spot Size (nm)", self._dsb_mill_spot)
        tescan_form.addRow("Preset", self._le_mill_preset)

        vbox.addWidget(self._grp_thermo)
        vbox.addWidget(self._grp_tescan)
        vbox.addStretch(1)
        return w

    def _build_subsystems_tab(self) -> QWidget:
        w = QWidget()
        vbox = QVBoxLayout(w)

        def grp(title: str, fields: list) -> QGroupBox:
            g = QGroupBox(title)
            f = QFormLayout(g)
            widgets = {}
            for label, key in fields:
                chk = QCheckBox()
                f.addRow(label, chk)
                widgets[key] = chk
            return g, widgets

        g_elec, self._sub_electron = grp("Electron", [("Enabled", "enabled")])
        g_ion, self._sub_ion = grp("Ion", [("Enabled", "enabled"), ("Plasma", "plasma")])
        g_stage, self._sub_stage = grp("Stage", [("Enabled", "enabled"), ("Rotation", "rotation"), ("Tilt", "tilt")])
        g_manip, self._sub_manip = grp("Manipulator", [("Enabled", "enabled"), ("Rotation", "rotation"), ("Tilt", "tilt")])
        g_gis, self._sub_gis = grp("GIS", [("Enabled", "enabled"), ("Multichem", "multichem"), ("Sputter Coater", "sputter_coater")])

        for g in (g_elec, g_ion, g_stage, g_manip, g_gis):
            vbox.addWidget(g)
        vbox.addStretch(1)
        return w

    def _build_sim_tab(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)

        def path_row(label: str):
            le = QLineEdit()
            btn = QPushButton("Browse...")
            row = QWidget()
            h = QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.addWidget(le)
            h.addWidget(btn)
            btn.clicked.connect(lambda: le.setText(
                QFileDialog.getOpenFileName(self, f"Select {label}")[0] or le.text()
            ))
            return row, le

        sem_row, self._le_sim_sem = path_row("SEM image")
        fib_row, self._le_sim_fib = path_row("FIB image")
        self._chk_sim_cycle = QCheckBox()

        form.addRow("SEM Data Path", sem_row)
        form.addRow("FIB Data Path", fib_row)
        form.addRow("Use Cycle", self._chk_sim_cycle)
        return w

    # ------------------------------------------------------------------
    # Manufacturer-driven visibility
    # ------------------------------------------------------------------

    def _on_manufacturer_changed(self, manufacturer: str) -> None:
        is_thermo = manufacturer in ("Thermo", "Demo")
        is_tescan = manufacturer in ("Tescan", "Demo")
        self._grp_thermo.setVisible(is_thermo)
        self._grp_tescan.setVisible(is_tescan)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_config(self, config: dict) -> None:
        self._config = config
        try:
            self._populate(config)
        except Exception as e:
            logging.warning(f"MicroscopeConfigWidget: error populating fields: {e}")

    def get_config(self) -> dict:
        try:
            return self._read()
        except Exception as e:
            logging.warning(f"MicroscopeConfigWidget: error reading fields: {e}")
            return self._config

    def load_from_file(self, path: str) -> None:
        self._path = path
        config = utils.load_yaml(path)
        self.set_config(config)

    def save_to_file(self, path: str) -> None:
        utils.save_yaml(path, self.get_config())
        self._path = path
        logging.info(f"MicroscopeConfigWidget: saved to {path}")

    # ------------------------------------------------------------------
    # Populate (config dict → widgets)
    # ------------------------------------------------------------------

    def _populate(self, c: dict) -> None:
        info = c.get("info", {})
        self._le_name.setText(str(info.get("name", "")))
        self._le_ip.setText(str(info.get("ip_address", "")))
        self._cb_manufacturer.setCurrentText(str(info.get("manufacturer", "Demo")))

        stage = c.get("stage", {})
        self._sb_stage_rot_ref.setValue(int(stage.get("rotation_reference", 0)))
        self._sb_stage_rot_180.setValue(int(stage.get("rotation_180", 180)))
        self._sb_stage_pretilt.setValue(int(stage.get("shuttle_pre_tilt", 0)))
        self._dsb_stage_manip_limit.setValue(float(stage.get("manipulator_height_limit", 0)) * constants.SI_TO_MILLI)

        self._populate_beam("electron", c.get("electron", {}))
        self._populate_beam("ion", c.get("ion", {}))

        imaging = c.get("imaging", {})
        self._cb_imaging_beam_type.setCurrentText(str(imaging.get("beam_type", "ELECTRON")))
        res = imaging.get("resolution", [1536, 1024])
        self._sb_imaging_res_x.setValue(int(res[0]))
        self._sb_imaging_res_y.setValue(int(res[1]))
        self._dsb_imaging_hfw.setValue(float(imaging.get("hfw", 150e-6)) * constants.SI_TO_MICRO)
        self._dsb_imaging_dwell.setValue(float(imaging.get("dwell_time", 1e-6)) * constants.SI_TO_MICRO)
        self._dsb_imaging_current.setValue(float(imaging.get("imaging_current", 0)) * constants.SI_TO_NANO)
        self._chk_autocontrast.setChecked(bool(imaging.get("autocontrast", True)))
        self._chk_autogamma.setChecked(bool(imaging.get("autogamma", False)))
        self._chk_autosave.setChecked(bool(imaging.get("save", False)))

        milling = c.get("milling", {})
        self._dsb_mill_voltage.setValue(float(milling.get("milling_voltage", 30000)))
        self._dsb_mill_current.setValue(float(milling.get("milling_current", 0)) * constants.SI_TO_NANO)
        self._dsb_mill_dwell.setValue(float(milling.get("dwell_time", 1e-6)) * constants.SI_TO_MICRO)
        self._dsb_mill_rate.setValue(float(milling.get("rate", 0)))
        self._dsb_mill_spot.setValue(float(milling.get("spot_size", 0)) * constants.SI_TO_NANO)
        self._le_mill_preset.setText(str(milling.get("preset", "")))

        for key, chk in self._sub_electron.items():
            chk.setChecked(bool(c.get("electron", {}).get(key, False)))
        for key, chk in self._sub_ion.items():
            chk.setChecked(bool(c.get("ion", {}).get(key, False)))
        for key, chk in self._sub_stage.items():
            chk.setChecked(bool(c.get("stage", {}).get(key, False)))
        for key, chk in self._sub_manip.items():
            chk.setChecked(bool(c.get("manipulator", {}).get(key, False)))
        for key, chk in self._sub_gis.items():
            chk.setChecked(bool(c.get("gis", {}).get(key, False)))

        sim = c.get("sim", {})
        self._le_sim_sem.setText(str(sim.get("sem", "") or ""))
        self._le_sim_fib.setText(str(sim.get("fib", "") or ""))
        self._chk_sim_cycle.setChecked(bool(sim.get("use_cycle", True)))

    def _populate_beam(self, beam: str, b: dict) -> None:
        f = getattr(self, f"_beam_{beam}")
        f["col_tilt"].setValue(int(b.get("column_tilt", 0)))
        f["euc"].setValue(float(b.get("eucentric_height", 0)) * constants.SI_TO_MILLI)
        f["voltage"].setValue(float(b.get("voltage", 0)))
        f["current"].setValue(float(b.get("current", 0)) * constants.SI_TO_NANO)
        res = b.get("resolution", [1536, 1024])
        f["res_x"].setValue(int(res[0]))
        f["res_y"].setValue(int(res[1]))
        f["hfw"].setValue(float(b.get("hfw", 150e-6)) * constants.SI_TO_MICRO)
        f["dwell"].setValue(float(b.get("dwell_time", 1e-6)) * constants.SI_TO_MICRO)
        f["det_mode"].setText(str(b.get("detector_mode", "")))
        f["det_type"].setText(str(b.get("detector_type", "")))
        if beam == "ion":
            f["plasma"].setChecked(bool(b.get("plasma", False)))
            f["plasma_gas"].setText(str(b.get("plasma_gas", "") or ""))

    # ------------------------------------------------------------------
    # Read (widgets → config dict)
    # ------------------------------------------------------------------

    def _read(self) -> dict:
        import copy
        c = copy.deepcopy(self._config)

        c.setdefault("info", {})
        c["info"]["name"] = self._le_name.text()
        c["info"]["ip_address"] = self._le_ip.text()
        c["info"]["manufacturer"] = self._cb_manufacturer.currentText()

        c.setdefault("stage", {})
        c["stage"]["rotation_reference"] = self._sb_stage_rot_ref.value()
        c["stage"]["rotation_180"] = self._sb_stage_rot_180.value()
        c["stage"]["shuttle_pre_tilt"] = self._sb_stage_pretilt.value()
        c["stage"]["manipulator_height_limit"] = self._dsb_stage_manip_limit.value() * constants.MILLI_TO_SI

        c["electron"] = self._read_beam("electron", c.get("electron", {}))
        c["ion"] = self._read_beam("ion", c.get("ion", {}))

        c.setdefault("imaging", {})
        c["imaging"]["beam_type"] = self._cb_imaging_beam_type.currentText()
        c["imaging"]["resolution"] = [self._sb_imaging_res_x.value(), self._sb_imaging_res_y.value()]
        c["imaging"]["hfw"] = self._dsb_imaging_hfw.value() * constants.MICRO_TO_SI
        c["imaging"]["dwell_time"] = self._dsb_imaging_dwell.value() * constants.MICRO_TO_SI
        c["imaging"]["imaging_current"] = self._dsb_imaging_current.value() * constants.NANO_TO_SI
        c["imaging"]["autocontrast"] = self._chk_autocontrast.isChecked()
        c["imaging"]["autogamma"] = self._chk_autogamma.isChecked()
        c["imaging"]["save"] = self._chk_autosave.isChecked()

        c.setdefault("milling", {})
        c["milling"]["milling_voltage"] = self._dsb_mill_voltage.value()
        c["milling"]["milling_current"] = self._dsb_mill_current.value() * constants.NANO_TO_SI
        c["milling"]["dwell_time"] = self._dsb_mill_dwell.value() * constants.MICRO_TO_SI
        c["milling"]["rate"] = self._dsb_mill_rate.value()
        c["milling"]["spot_size"] = self._dsb_mill_spot.value() * constants.NANO_TO_SI
        c["milling"]["preset"] = self._le_mill_preset.text()

        for key, chk in self._sub_electron.items():
            c.setdefault("electron", {})[key] = chk.isChecked()
        for key, chk in self._sub_ion.items():
            c.setdefault("ion", {})[key] = chk.isChecked()
        for key, chk in self._sub_stage.items():
            c.setdefault("stage", {})[key] = chk.isChecked()
        for key, chk in self._sub_manip.items():
            c.setdefault("manipulator", {})[key] = chk.isChecked()
        for key, chk in self._sub_gis.items():
            c.setdefault("gis", {})[key] = chk.isChecked()

        c.setdefault("sim", {})
        c["sim"]["sem"] = self._le_sim_sem.text() or None
        c["sim"]["fib"] = self._le_sim_fib.text() or None
        c["sim"]["use_cycle"] = self._chk_sim_cycle.isChecked()

        return c

    def _read_beam(self, beam: str, existing: dict) -> dict:
        import copy
        b = copy.deepcopy(existing)
        f = getattr(self, f"_beam_{beam}")
        b["column_tilt"] = f["col_tilt"].value()
        b["eucentric_height"] = f["euc"].value() * constants.MILLI_TO_SI
        b["voltage"] = f["voltage"].value()
        b["current"] = f["current"].value() * constants.NANO_TO_SI
        b["resolution"] = [f["res_x"].value(), f["res_y"].value()]
        b["hfw"] = f["hfw"].value() * constants.MICRO_TO_SI
        b["dwell_time"] = f["dwell"].value() * constants.MICRO_TO_SI
        b["detector_mode"] = f["det_mode"].text()
        b["detector_type"] = f["det_type"].text()
        if beam == "ion":
            b["plasma"] = f["plasma"].isChecked()
            b["plasma_gas"] = f["plasma_gas"].text() or None
        return b

    # ------------------------------------------------------------------
    # Load / Save slots
    # ------------------------------------------------------------------

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Microscope Configuration", self._path or "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.load_from_file(path)

    def _on_save(self) -> None:
        default = self._path or os.path.join(os.path.expanduser("~"), "microscope-configuration.yaml")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Microscope Configuration", default, "YAML (*.yaml *.yml)"
        )
        if path:
            self.save_to_file(path)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _dsb(min_val: float, max_val: float, decimals: int, suffix: str = "") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(min_val, max_val)
    w.setDecimals(decimals)
    w.setSingleStep(10 ** (-decimals) * 10)
    if suffix:
        w.setSuffix(suffix)
    return w


# ---------------------------------------------------------------------------
# Dialog wrapper (for embedding in AutoLamellaMainUI)
# ---------------------------------------------------------------------------

def open_microscope_config_dialog(path: Optional[str] = None, parent=None) -> None:
    """Open MicroscopeConfigWidget in a modal dialog."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Microscope Configuration")
    dlg.resize(640, 720)
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(0, 0, 0, 0)

    widget = MicroscopeConfigWidget(path=path or cfg.DEFAULT_CONFIGURATION_PATH)
    layout.addWidget(widget)

    btns = QDialogButtonBox(QDialogButtonBox.Close)
    btns.rejected.connect(dlg.reject)
    layout.addWidget(btns)

    dlg.exec_()

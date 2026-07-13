"""A compact preset picker for FM configurations.

Owns the registry + dialog UI (list / load / save-as / overwrite / delete) but is
agnostic about *where* the configuration comes from or goes to: the host supplies
a ``config_provider`` callable that returns the current FluorescenceConfiguration,
and connects to ``configuration_loaded`` to apply a loaded one. This keeps the
widget-specific read/apply logic in each host (which has different sub-widgets)
rather than forcing a shared, lowest-common-denominator helper.
"""

import logging
from typing import Callable, Optional

from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)
from PyQt5.QtCore import pyqtSignal

from fibsem.fm import config as fm_config
from fibsem.fm.structures import FluorescenceConfiguration
from fibsem.ui.stylesheets import PRIMARY_BUTTON_STYLESHEET, SECONDARY_BUTTON_STYLESHEET

_PLACEHOLDER = "— presets —"


class FMPresetBar(QWidget):
    """Preset picker: pick a named FM configuration, or save the current one."""

    configuration_loaded = pyqtSignal(object)  # FluorescenceConfiguration

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config_provider: Optional[Callable[[], FluorescenceConfiguration]] = None
        self._build_ui()
        self.refresh_presets()

    def set_config_provider(
        self, provider: Callable[[], FluorescenceConfiguration]
    ) -> None:
        """Register a callable returning the current FluorescenceConfiguration."""
        self._config_provider = provider

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(QLabel("Preset"))

        self.combo = QComboBox()
        self.combo.setMinimumWidth(140)
        layout.addWidget(self.combo, stretch=1)

        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save")
        self.btn_save_as = QPushButton("Save As…")
        self.btn_delete = QPushButton("Delete")
        self.btn_load.setStyleSheet(PRIMARY_BUTTON_STYLESHEET)
        for b in (self.btn_save, self.btn_save_as, self.btn_delete):
            b.setStyleSheet(SECONDARY_BUTTON_STYLESHEET)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_save_as)
        layout.addWidget(self.btn_delete)

        self.btn_load.clicked.connect(self._on_load)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as.clicked.connect(self._on_save_as)
        self.btn_delete.clicked.connect(self._on_delete)
        self.combo.currentTextChanged.connect(self._update_button_state)

    def refresh_presets(self, select: Optional[str] = None) -> None:
        """Reload the preset list; optionally select a given name."""
        names = fm_config.list_fm_presets()
        default = fm_config.get_default_fm_preset_name()
        self.combo.blockSignals(True)
        self.combo.clear()
        if names:
            self.combo.addItems(names)
            target = select or default or names[0]
            idx = self.combo.findText(target)
            self.combo.setCurrentIndex(max(0, idx))
        else:
            self.combo.addItem(_PLACEHOLDER)
        self.combo.blockSignals(False)
        self._update_button_state()

    def _selected_name(self) -> Optional[str]:
        text = self.combo.currentText()
        if not text or text == _PLACEHOLDER:
            return None
        return text

    def _update_button_state(self, *_) -> None:
        has_selection = self._selected_name() is not None
        self.btn_load.setEnabled(has_selection)
        self.btn_save.setEnabled(has_selection)
        self.btn_delete.setEnabled(has_selection)

    # --- actions ---------------------------------------------------------

    def _on_load(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        try:
            config = fm_config.load_fm_preset(name)
        except Exception as e:
            logging.error(f"Failed to load FM preset '{name}': {e}")
            QMessageBox.warning(self, "Load Preset", f"Failed to load '{name}':\n{e}")
            return
        fm_config.set_default_fm_preset(name)
        self.configuration_loaded.emit(config)

    def _current_config(self) -> Optional[FluorescenceConfiguration]:
        if self._config_provider is None:
            return None
        try:
            return self._config_provider()
        except Exception as e:
            logging.error(f"Failed to read current FM configuration: {e}")
            QMessageBox.warning(self, "Save Preset", f"Could not read current settings:\n{e}")
            return None

    def _on_save(self) -> None:
        """Overwrite the currently selected preset with the current settings."""
        name = self._selected_name()
        if name is None:
            return
        config = self._current_config()
        if config is None:
            return
        fm_config.save_fm_preset(name, config, set_default=True)
        self.refresh_presets(select=name)

    def _on_save_as(self) -> None:
        config = self._current_config()
        if config is None:
            return
        name, ok = QInputDialog.getText(self, "Save FM Preset As", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in fm_config.list_fm_presets():
            ret = QMessageBox.question(
                self, "Overwrite Preset",
                f"A preset named '{name}' already exists. Overwrite it?",
            )
            if ret != QMessageBox.Yes:
                return
        fm_config.save_fm_preset(name, config, set_default=True)
        self.refresh_presets(select=name)

    def _on_delete(self) -> None:
        name = self._selected_name()
        if name is None:
            return
        ret = QMessageBox.question(
            self, "Delete Preset", f"Delete the preset '{name}'?"
        )
        if ret != QMessageBox.Yes:
            return
        try:
            fm_config.remove_fm_preset(name)
        except Exception as e:
            logging.error(f"Failed to delete FM preset '{name}': {e}")
        self.refresh_presets()

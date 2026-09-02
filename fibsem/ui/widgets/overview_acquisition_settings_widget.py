from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.config import (
    AVAILABLE_RESOLUTIONS_ZIP,
    DEFAULT_STANDARD_RESOLUTION,
    DEFAULT_STANDARD_RESOLUTION_LIST,
)
from fibsem.imaging.tiled import stamped_overview_name as _stamped_overview_name
from fibsem.structures import (
    AutoFocusMode,
    BeamType,
    FocusStackSettings,
    ImageSettings,
    OverviewAcquisitionSettings,
    TileOrderStrategy,
)
from fibsem.ui import stylesheets
from fibsem.ui.widgets.custom_widgets import (
    IconToolButton,
    TitledPanel,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.image_settings_widget import ImageSettingsWidget

# What an overview run looks like before anyone touches it.
#
# House defaults rather than physics: a 500 um tile and a 1 us dwell are what the napari
# minimap has always opened with. They lived in that module, which is scheduled for
# deletion, and nothing on the rebuilt path referenced them -- so the tab opened at
# `ImageSettings`'s generic values instead: a 150 um tile, autocontrast off, and a run
# called "default_image".
#
# The name is the one that bites. `TiledAcquisitionRunner._setup` makes the tile
# sub-folder from it, so it decides where every overview of every experiment is written.
OVERVIEW_TILE_HFW = 500e-6  # m
OVERVIEW_TILE_DWELL_TIME = 1e-6  # s
DEFAULT_OVERVIEW_FILENAME = "overview-image"


# Re-exported: it lives with the runners that make the folder from it, and both
# overview tabs and the grid overview tasks import the same one.
stamped_overview_name = _stamped_overview_name


def default_overview_acquisition_settings() -> OverviewAcquisitionSettings:
    """A fresh set of overview settings, seeded with the house defaults.

    A function, not the module-level constant it replaces. The minimap held one of those
    and assigned an experiment path straight into it
    (`DEFAULT_OVERVIEW_ACQUISITION_SETTINGS.image_settings.path = path`), which edits the
    default for everything built afterwards. `ImageSettingsWidget` compounds it:
    `update_from_settings` keeps the object it is handed and `get_settings` mutates and
    returns that same one, so a shared default would be rewritten by every keystroke in
    the tab.

    The grid size is not named here -- `OverviewAcquisitionSettings` already defaults to
    3 x 3, and stating it twice is how the two drift apart.

    The resolution is the standard 1536x1024, not the square one the rebuilt tab used to
    open with. It is what the shipped tab has always acquired at, and the difference is
    not cosmetic: `hfw` is the *horizontal* field, so the same 500 um tile at 1024x1024
    is 0.49 um/px against 0.33 and half again as tall. Square was picked before there was
    a shipped tab to compare against; there is one now, and overviews that can be put
    side by side across the swap are worth more than the tidier number.
    """
    return OverviewAcquisitionSettings(
        image_settings=ImageSettings(
            # `tuple`, not the config list itself: `resolution` is annotated as one, and
            # handing every settings object the same mutable list is the sharing this
            # factory exists to avoid.
            resolution=tuple(DEFAULT_STANDARD_RESOLUTION_LIST),
            hfw=OVERVIEW_TILE_HFW,
            dwell_time=OVERVIEW_TILE_DWELL_TIME,
            autocontrast=True,
            beam_type=BeamType.ELECTRON,
            save=True,
            path=None,  # whoever owns the experiment fills this in
            filename=DEFAULT_OVERVIEW_FILENAME,
        ),
    )


class OverviewAcquisitionSettingsWidget(QWidget):
    """Widget for editing OverviewAcquisitionSettings.

    Composes a tile-grid section (beam type, rows, cols, total FOV) on top of
    an embedded ImageSettingsWidget (resolution, dwell time, tile FOV,
    autocontrast, save, path, filename).
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._show_advanced = False
        # No control for this yet -- see the `tile_mask` property.
        self._tile_mask: Optional[List[List[bool]]] = None
        self._setup_ui()
        # Before `_connect_signals`, so seeding the boxes does not read as a user edit.
        self.update_from_settings(default_overview_acquisition_settings())
        self._connect_signals()
        self._update_total_fov_label()
        self._update_advanced_visibility()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Overview Acquisition panel ────────────────────────────────
        grid_content = QWidget()
        grid_layout = QGridLayout(grid_content)
        grid_layout.setContentsMargins(4, 4, 4, 4)

        # Beam type
        grid_layout.addWidget(QLabel("Beam Type"), 0, 0)
        self.beam_type_combo = ValueComboBox(
            items=list(BeamType), format_fn=lambda bt: bt.name
        )
        grid_layout.addWidget(self.beam_type_combo, 0, 1, 1, 2)

        # Rows / cols
        grid_layout.addWidget(QLabel("Tiles (rows x cols)"), 1, 0)
        # Values come from the seed in `__init__`, not from here.
        self.nrows_spinbox = ValueSpinBox(minimum=1, maximum=15, step=1, decimals=0)
        self.ncols_spinbox = ValueSpinBox(minimum=1, maximum=15, step=1, decimals=0)
        # Two spinboxes share one grid cell here, so they are the first thing a narrow
        # column squeezes -- and below about 80px the +/- buttons take the whole
        # control and the number stops being drawn at all, leaving what looks like a
        # broken widget rather than a cramped one. Measured: at 63px nothing renders
        # while `lineEdit().text()` still reports "3", so no test notices.
        for _spinbox in (self.nrows_spinbox, self.ncols_spinbox):
            _spinbox.setMinimumWidth(80)

        _tiles_row = QWidget()
        _tiles_layout = QHBoxLayout(_tiles_row)
        _tiles_layout.setContentsMargins(0, 0, 0, 0)
        _tiles_layout.addWidget(self.nrows_spinbox)
        _label_x = QLabel("x")
        _label_x.setAlignment(Qt.AlignCenter)  # type: ignore
        _tiles_layout.addWidget(_label_x)
        _tiles_layout.addWidget(self.ncols_spinbox)
        grid_layout.addWidget(_tiles_row, 1, 1, 1, 2)

        # Overlap
        grid_layout.addWidget(QLabel("Overlap (%)"), 2, 0)
        self.overlap_spinbox = ValueSpinBox(
            suffix="%", minimum=0.0, maximum=50.0, step=5.0, decimals=0
        )
        grid_layout.addWidget(self.overlap_spinbox, 2, 1, 1, 2)

        # Total FOV label (read-only, auto-updated)
        self._label_total_fov = QLabel("Total FOV: —")
        self._label_total_fov.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # type: ignore
        grid_layout.addWidget(self._label_total_fov, 3, 0, 1, 3)

        # Focus stack (advanced)
        self._label_focus_stack = QLabel("Focus Stack")
        grid_layout.addWidget(self._label_focus_stack, 4, 0)
        self.focus_stack_enabled = QCheckBox()
        grid_layout.addWidget(self.focus_stack_enabled, 4, 1, 1, 2)

        self._label_focus_stack_steps = QLabel("Focus Stack Steps")
        grid_layout.addWidget(self._label_focus_stack_steps, 5, 0)
        self.focus_stack_steps = ValueSpinBox(minimum=1, maximum=10, step=1, decimals=0)
        self.focus_stack_steps.setValue(3)
        grid_layout.addWidget(self.focus_stack_steps, 5, 1, 1, 2)

        self._label_focus_stack_autofocus = QLabel("Focus Stack Autofocus")
        grid_layout.addWidget(self._label_focus_stack_autofocus, 6, 0)
        self.focus_stack_autofocus = QCheckBox()
        self.focus_stack_autofocus.setChecked(True)
        grid_layout.addWidget(self.focus_stack_autofocus, 6, 1, 1, 2)

        # Auto Focus mode (advanced)
        self._label_autofocus = QLabel("Auto Focus")
        grid_layout.addWidget(self._label_autofocus, 7, 0)
        self.autofocus_combo = ValueComboBox(
            items=list(AutoFocusMode),
            format_fn=lambda m: m.name.replace("_", " ").title(),
        )
        grid_layout.addWidget(self.autofocus_combo, 7, 1, 1, 2)

        # Tile order strategy (advanced)
        self._label_tile_order = QLabel("Tile Order")
        grid_layout.addWidget(self._label_tile_order, 8, 0)
        self.tile_order_combo = ValueComboBox(
            items=list(TileOrderStrategy),
            format_fn=lambda s: s.value.title(),
        )
        grid_layout.addWidget(self.tile_order_combo, 8, 1, 1, 2)

        self._adv_widgets = [
            (self._label_focus_stack, self.focus_stack_enabled),
            (self._label_focus_stack_steps, self.focus_stack_steps),
            (self._label_focus_stack_autofocus, self.focus_stack_autofocus),
            (self._label_autofocus, self.autofocus_combo),
            (self._label_tile_order, self.tile_order_combo),
        ]

        self._btn_advanced = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced settings",
            checked_tooltip="Hide advanced settings",
        )

        # ── Tile Image Settings + combined panel ─────────────────────────────────
        self.image_settings_widget = ImageSettingsWidget(
            show_advanced=False,
            always_save=True,
        )
        self.image_settings_widget.hfw_label.setText("Field of View")
        self.image_settings_widget.set_show_advanced_button(False)

        # All standard + square resolutions are supported (non-square aspect handled in acquisition)
        self.image_settings_widget.set_available_resolutions(
            AVAILABLE_RESOLUTIONS_ZIP, default=DEFAULT_STANDARD_RESOLUTION
        )

        self._btn_advanced_imaging = IconToolButton(
            icon="mdi:tune",
            checked_icon="mdi:tune-variant",
            checked_color=stylesheets.GRAY_WHITE_COLOR,
            tooltip="Show advanced image settings",
            checked_tooltip="Hide advanced image settings",
        )

        self._image_panel = TitledPanel(
            "Tile Image Settings", content=self.image_settings_widget
        )
        self._image_panel.add_header_widget(self._btn_advanced_imaging)
        self._image_panel._btn_collapse.setChecked(True)

        # combine grid controls + image settings panel into one collapsible panel
        combined = QWidget()
        combined_layout = QVBoxLayout(combined)
        combined_layout.setContentsMargins(0, 0, 0, 0)
        combined_layout.setSpacing(0)
        combined_layout.addWidget(grid_content)
        combined_layout.addWidget(self._image_panel)

        grid_panel = TitledPanel("Overview Acquisition", content=combined)
        grid_panel.add_header_widget(self._btn_advanced)
        grid_panel._btn_collapse.setChecked(True)
        outer.addWidget(grid_panel)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_advanced.toggled.connect(self.set_show_advanced)
        self.beam_type_combo.currentIndexChanged.connect(self._on_changed)
        self.nrows_spinbox.valueChanged.connect(self._on_changed)
        self._btn_advanced_imaging.toggled.connect(
            self.image_settings_widget.set_show_advanced
        )
        self.ncols_spinbox.valueChanged.connect(self._on_changed)
        self.overlap_spinbox.valueChanged.connect(self._on_changed)
        self.focus_stack_enabled.toggled.connect(self._on_changed)
        self.focus_stack_steps.valueChanged.connect(self._on_changed)
        self.focus_stack_autofocus.toggled.connect(self._on_changed)
        self.autofocus_combo.currentIndexChanged.connect(self._on_changed)
        self.tile_order_combo.currentIndexChanged.connect(self._on_changed)
        self.image_settings_widget.settings_changed.connect(self._on_changed)

    def _on_changed(self):
        self._update_total_fov_label()
        self.settings_changed.emit()

    # ------------------------------------------------------------------
    # Advanced visibility
    # ------------------------------------------------------------------

    def set_show_advanced(self, show: bool):
        self._show_advanced = show
        self._btn_advanced.blockSignals(True)
        self._btn_advanced.setChecked(show)
        self._btn_advanced.blockSignals(False)
        self._btn_advanced.set_icon_state(show)
        self._update_advanced_visibility()

    def _update_advanced_visibility(self):
        for label, widget in self._adv_widgets:
            label.setVisible(self._show_advanced)
            widget.setVisible(self._show_advanced)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_total_fov_label(self):
        settings = self.get_settings()
        total_w = settings.total_fov_x * constants.SI_TO_MICRO
        total_h = settings.total_fov_y * constants.SI_TO_MICRO
        sym = constants.MICRON_SYMBOL
        self._label_total_fov.setText(f"Total FOV: {total_w:.0f} × {total_h:.0f} {sym}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tile_mask(self) -> Optional[List[List[bool]]]:
        """Which tiles the next run would acquire, or None for all of them.

        Held here rather than drawn, for now: there is no control for it, and the one
        that will set it is the tile grid on the canvas (FIB-617). It lives here anyway
        because `get_settings` builds a *new* settings object on every call, so a mask
        set anywhere else would be silently dropped the next time anything read the
        widget -- which is on every overlay refresh.
        """
        return self._tile_mask

    @tile_mask.setter
    def tile_mask(self, mask: Optional[List[List[bool]]]) -> None:
        self._tile_mask = (
            None if mask is None else [[bool(v) for v in row] for row in mask]
        )
        self._on_changed()

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Set both grid dimensions as a single change.

        Setting the two spin boxes one after the other emits `settings_changed` twice
        and passes through a size nobody asked for -- the new row count against the old
        column count. That is invisible when a spin box is nudged by hand, and constant
        when an edge is dragged on the canvas, which emits on every motion event.
        """
        rows, cols = int(rows), int(cols)
        if (rows, cols) == (
            int(self.nrows_spinbox.value()),
            int(self.ncols_spinbox.value()),
        ):
            return
        for spinbox in (self.nrows_spinbox, self.ncols_spinbox):
            spinbox.blockSignals(True)
        try:
            self.nrows_spinbox.setValue(rows)
            self.ncols_spinbox.setValue(cols)
        finally:
            for spinbox in (self.nrows_spinbox, self.ncols_spinbox):
                spinbox.blockSignals(False)
        self._on_changed()

    def get_settings(self) -> OverviewAcquisitionSettings:
        """Read current widget values and return OverviewAcquisitionSettings."""
        image_settings: ImageSettings = self.image_settings_widget.get_settings()
        image_settings.beam_type = self.beam_type_combo.value()
        return OverviewAcquisitionSettings(
            tile_mask=self._mask_for_grid(),
            image_settings=image_settings,
            nrows=int(self.nrows_spinbox.value()),
            ncols=int(self.ncols_spinbox.value()),
            overlap=self.overlap_spinbox.value() / 100.0,
            focus_stack_settings=FocusStackSettings(
                enabled=self.focus_stack_enabled.isChecked(),
                n_steps=int(self.focus_stack_steps.value()),
                auto_focus=self.focus_stack_autofocus.isChecked(),
            ),
            autofocus_mode=self.autofocus_combo.value(),
            tile_order=self.tile_order_combo.value(),
        )

    def _mask_for_grid(self) -> Optional[List[List[bool]]]:
        """The mask, or None if it no longer describes the grid.

        Rows and columns are spin boxes and the mask is positional, so the two go out
        of step the moment either is changed. `compute_tile_grid` rejects a mismatched
        mask outright -- correctly, since silently tolerating one would acquire the
        wrong tiles -- which would turn resizing the grid into a failed run.

        Dropped rather than resized: growing it has to invent whether the new tiles are
        in or out, and shrinking it throws away a choice. Whatever sets the mask knows
        which it wants and can set it again; this only has to avoid handing on one that
        is no longer true.
        """
        mask = self._tile_mask
        if mask is None:
            return None
        rows, cols = int(self.nrows_spinbox.value()), int(self.ncols_spinbox.value())
        if len(mask) != rows or any(len(row) != cols for row in mask):
            return None
        return [list(row) for row in mask]

    def update_from_settings(self, settings: OverviewAcquisitionSettings):
        """Populate all widgets from an OverviewAcquisitionSettings object."""
        # Block tile-grid signals to prevent cascading updates
        for w in [
            self.beam_type_combo,
            self.nrows_spinbox,
            self.ncols_spinbox,
            self.overlap_spinbox,
            self.focus_stack_enabled,
            self.focus_stack_steps,
            self.focus_stack_autofocus,
            self.autofocus_combo,
            self.tile_order_combo,
        ]:
            w.blockSignals(True)

        self.beam_type_combo.set_value(settings.image_settings.beam_type)
        self.nrows_spinbox.setValue(settings.nrows)
        self.ncols_spinbox.setValue(settings.ncols)
        self.overlap_spinbox.setValue(settings.overlap * 100.0)
        self.focus_stack_enabled.setChecked(settings.focus_stack_settings.enabled)
        self.focus_stack_steps.setValue(settings.focus_stack_settings.n_steps)
        self.focus_stack_autofocus.setChecked(settings.focus_stack_settings.auto_focus)
        self.autofocus_combo.set_value(settings.autofocus_mode)
        self.tile_order_combo.set_value(settings.tile_order)

        for w in [
            self.beam_type_combo,
            self.nrows_spinbox,
            self.ncols_spinbox,
            self.overlap_spinbox,
            self.focus_stack_enabled,
            self.focus_stack_steps,
            self.focus_stack_autofocus,
            self.autofocus_combo,
            self.tile_order_combo,
        ]:
            w.blockSignals(False)

        self._tile_mask = (
            None
            if settings.tile_mask is None
            else [[bool(v) for v in row] for row in settings.tile_mask]
        )
        self.image_settings_widget.update_from_settings(settings.image_settings)
        self._update_total_fov_label()


# ---------------------------------------------------------------------------
# Quick test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    from PyQt5.QtWidgets import QApplication, QPushButton

    app = QApplication(sys.argv)
    w = QWidget()
    layout = QVBoxLayout(w)

    widget = OverviewAcquisitionSettingsWidget()
    layout.addWidget(widget)

    btn = QPushButton("Print settings")

    def _print():
        s = widget.get_settings()
        print(s)

    btn.clicked.connect(_print)
    layout.addWidget(btn)

    w.setWindowTitle("OverviewAcquisitionSettingsWidget test")
    w.show()
    sys.exit(app.exec_())

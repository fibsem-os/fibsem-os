"""The FIB/SEM overview's settings column.

Replaces `OverviewAcquisitionSettingsWidget` on the canvas tab. That one stays where it
is: the napari Overview tab still uses it, and the two go together when that tab does.

Read top to bottom it sets up one tile, then the grid of them: **Imaging** (what a tile
is), **Focus** (when to focus while walking the grid), **Stack** (whether to take a focus
stack at each tile), **Grid** (how many, where, which). That is the fluorescence column's
order, and it has a second justification here that tab does not have -- beam type is not
merely a setting, it re-poses the whole canvas (`_follow_the_acquisition_view`), so the
control that reorients everything belongs at the top rather than buried mid-column.

Focus and focus stack are two panels because they are two questions: *when to focus* and
*whether to acquire a stack*. Either can be wanted without the other, and one panel said
otherwise.

**Output** last and folded: with a run stamped by the time it started, the name is a
label rather than a destination, and the folder is the experiment's.

Two panels open, three folded, which is the fluorescence column's budget. It matters
because these sit above a `Display` panel and the pinned action row: a third expanded
panel puts the grid below the fold on a laptop-height window.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QLabel,
    QLineEdit,  # noqa: E402  (grouped with the Qt imports above)
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.config import AVAILABLE_RESOLUTIONS_ZIP
from fibsem.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    BeamType,
    FocusStackSettings,
    ImageSettings,
    OverviewAcquisitionSettings,
    TileOrderStrategy,
)
from fibsem.ui.tokens import TEXT_MUTED_COLOR
from fibsem.ui.utils import install_wheel_blocker
from fibsem.ui.widgets.custom_widgets import (
    QDirectoryLineEdit,
    TitledPanel,
    ValueComboBox,
    ValueSpinBox,
)
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    default_overview_acquisition_settings,
)
from fibsem.ui.widgets.overview_grid_settings_widget import OverviewGridSettingsWidget

_MUTED = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"
_FIELD_MIN_WIDTH = 90


class FibsemOverviewSettingsWidget(QWidget):
    """Everything a FIB/SEM overview run needs, as four titled panels.

    Same public surface as the widget it replaces -- `get_settings`,
    `update_from_settings`, `set_grid_size`, `tile_mask`, `settings_changed` -- so the
    tab swaps by changing one constructor call.
    """

    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build()
        self._connect()
        self.update_from_settings(default_overview_acquisition_settings())

    # ── construction ─────────────────────────────────────────────────────

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        outer.addWidget(self._imaging_panel())
        outer.addWidget(self._focus_panel())
        outer.addWidget(self._focus_stack_panel())

        self.grid = OverviewGridSettingsWidget()
        outer.addWidget(self.grid)

        outer.addWidget(self._output_panel())

    @staticmethod
    def _form() -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(6)
        return form

    @staticmethod
    def _field(widget: QWidget) -> QWidget:
        widget.setMinimumWidth(_FIELD_MIN_WIDTH)
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        install_wheel_blocker(widget)
        return widget

    @staticmethod
    def _panel(title: str, layout, header=None, collapsed: bool = False) -> TitledPanel:
        content = QWidget()
        content.setLayout(layout)
        panel = TitledPanel(title, content=content)
        if header is not None:
            panel.add_header_widget(header)
        if collapsed:
            panel.collapse()
        return panel

    def _imaging_panel(self) -> TitledPanel:
        """What a single tile is. Beam first, because changing it moves the canvas."""
        self.combo_beam = self._field(
            ValueComboBox(items=list(BeamType), format_fn=lambda b: b.name)
        )
        self.combo_resolution = self._field(
            ValueComboBox(
                items=[value for _, value in AVAILABLE_RESOLUTIONS_ZIP],
                format_fn=lambda r: f"{r[0]}x{r[1]}",
            )
        )
        self.spin_dwell = self._field(
            ValueSpinBox(
                suffix=f" {constants.MICROSECOND_SYMBOL}",
                minimum=0.01,
                maximum=1000.0,
                step=0.5,
                decimals=2,
            )
        )
        self.spin_hfw = self._field(
            ValueSpinBox(
                suffix=f" {constants.MICRON_SYMBOL}",
                minimum=1.0,
                maximum=10000.0,
                step=50.0,
                decimals=1,
            )
        )
        self.check_autocontrast = QCheckBox()

        # The one number that says whether the dwell just typed means seconds or hours.
        # Scan time, not run time -- see `OverviewAcquisitionSettings.scan_time`.
        self.label_scan_time = QLabel("—")
        self.label_scan_time.setStyleSheet(_MUTED)
        self.label_scan_time.setToolTip(
            "Beam-on time only. The stage has to move and settle between tiles, so a "
            "run takes longer than this."
        )

        form = self._form()
        form.addRow("Beam", self.combo_beam)
        form.addRow("Resolution", self.combo_resolution)
        form.addRow("Dwell time", self.spin_dwell)
        form.addRow("Field of view", self.spin_hfw)
        form.addRow("Auto contrast", self.check_autocontrast)
        form.addRow("Scan time", self.label_scan_time)
        self.imaging_panel = self._panel("Imaging", form)
        return self.imaging_panel

    def _focus_panel(self) -> TitledPanel:
        """*When* to focus while walking the grid. Not the same question as the focus
        stack below, which is whether to acquire a stack at each tile -- either can be
        wanted without the other, and putting them in one panel said otherwise.

        One row, because a mode is all the runner can currently act on: it calls the
        instrument's own routine, which takes a beam and a reduced area and nothing
        else. Method and sweep passes arrive with FIB-646, which connects the tiled
        runner to `run_auto_focus` like every other autofocus caller in the codebase.
        """
        self.combo_autofocus = self._field(
            ValueComboBox(
                items=list(AutoFocusMode),
                format_fn=lambda m: m.name.replace("_", " ").title(),
            )
        )

        # The runner silently promotes EACH_ROW to EACH_TILE for a spiral, because a
        # spiral has no rows -- correct, and until now invisible. The fluorescence tab
        # warns about the same combination; this tab acted on it and said nothing.
        self.label_focus_note = QLabel("")
        self.label_focus_note.setStyleSheet(_MUTED)
        self.label_focus_note.setWordWrap(True)
        self.label_focus_note.hide()

        form = self._form()
        form.addRow("Auto-focus", self.combo_autofocus)
        form.addRow("", self.label_focus_note)
        self.focus_panel = self._panel("Focus", form, collapsed=True)
        return self.focus_panel

    def _focus_stack_panel(self) -> TitledPanel:
        """Enable in the header, like the fluorescence tab's Z-Stack: it is the one piece
        of state worth seeing when the panel is folded away."""
        self.check_focus_stack = QCheckBox()
        self.check_focus_stack.setToolTip("Acquire a focus stack at each tile")
        self.spin_focus_steps = self._field(
            ValueSpinBox(minimum=1, maximum=10, step=1, decimals=0)
        )
        self.check_focus_autofocus = QCheckBox()

        form = self._form()
        form.addRow("Steps", self.spin_focus_steps)
        form.addRow("Auto-focus each", self.check_focus_autofocus)
        # "Stack", not "Focus stack": it sits directly under "Focus", which already
        # supplies that half of the name, and it is the counterpart to the fluorescence
        # tab's "Z-Stack" -- one word each, differing only in what is varied.
        self.focus_stack_panel = self._panel(
            "Stack", form, header=self.check_focus_stack, collapsed=True
        )
        return self.focus_stack_panel

    def _output_panel(self) -> TitledPanel:
        self.path_edit = self._field(QDirectoryLineEdit())
        self.filename_edit = self._field(QLineEdit())

        # What the run will actually be called. The box holds a base name; the run stamps
        # it with the time it started so two runs cannot land in one directory.
        self.label_stamped = QLabel("—")
        self.label_stamped.setStyleSheet(_MUTED)

        form = self._form()
        form.addRow("Folder", self.path_edit)
        form.addRow("Name", self.filename_edit)
        form.addRow("", self.label_stamped)
        self.output_panel = self._panel("Output", form, collapsed=True)
        return self.output_panel

    def _connect(self) -> None:
        self.combo_beam.currentIndexChanged.connect(self._on_changed)
        self.combo_resolution.currentIndexChanged.connect(self._on_changed)
        self.spin_dwell.valueChanged.connect(self._on_changed)
        self.spin_hfw.valueChanged.connect(self._on_changed)
        self.check_autocontrast.toggled.connect(self._on_changed)
        self.check_focus_stack.toggled.connect(self._on_focus_stack_toggled)
        self.spin_focus_steps.valueChanged.connect(self._on_changed)
        self.check_focus_autofocus.toggled.connect(self._on_changed)
        self.combo_autofocus.currentIndexChanged.connect(self._on_changed)
        self.filename_edit.textChanged.connect(self._on_changed)
        self.path_edit.lineEdit.textChanged.connect(self._on_changed)
        self.grid.changed.connect(self._on_changed)
        self._on_focus_stack_toggled()

    # ── reactions ────────────────────────────────────────────────────────

    def _on_focus_stack_toggled(self) -> None:
        """Grey the parameters out rather than leaving them live and implying they
        apply -- the fluorescence tab's Z-Stack does the same."""
        enabled = self.check_focus_stack.isChecked()
        self.spin_focus_steps.setEnabled(enabled)
        self.check_focus_autofocus.setEnabled(enabled)
        self._on_changed()

    def _on_changed(self) -> None:
        self._refresh_derived()
        self.settings_changed.emit()

    def _refresh_derived(self) -> None:
        settings = self.get_settings()
        # A tile is square-ish only when the resolution is; report both axes from the
        # shape rather than assuming, which is what `total_fov_y` does.
        width, height = settings.image_settings.resolution
        fov_x = settings.image_settings.hfw
        fov_y = fov_x * (height / width) if width else fov_x
        self.grid.set_tile_fov(fov_x, fov_y)

        from fibsem.ui.widgets.preflight import format_duration

        per_tile = settings.image_settings.scan_time
        self.label_scan_time.setText(
            f"{format_duration(settings.scan_time)}  ·  "
            f"{format_duration(per_tile)} per tile"
        )
        name = self.filename_edit.text().strip()
        # The shape of the name, not a name. The stamp is the time the run *starts*, so
        # there is no true value to show while it is still being set up -- and a real
        # time here would be read as one, then be wrong by however long the setup took.
        self.label_stamped.setText(f"saved as {name}-HH-MM-SS" if name else "—")
        self._refresh_focus_note(settings)

    def _refresh_focus_note(self, settings: OverviewAcquisitionSettings) -> None:
        """Say when the runner will not do what the mode says.

        `TiledAcquisitionRunner._compute_grid` promotes EACH_ROW to EACH_TILE for a
        spiral, because a spiral has no rows. That is the right call and it was silent:
        the setting read one thing and the run did another.
        """
        promoted = (
            settings.autofocus_settings.mode is AutoFocusMode.EACH_ROW
            and settings.tile_order is TileOrderStrategy.SPIRAL
        )
        self.label_focus_note.setVisible(promoted)
        if promoted:
            self.label_focus_note.setText(
                "A spiral has no rows, so this will focus at every tile."
            )

    # ── public API ───────────────────────────────────────────────────────

    @property
    def tile_mask(self) -> Optional[List[List[bool]]]:
        return self.grid.mask

    @tile_mask.setter
    def tile_mask(self, mask: Optional[List[List[bool]]]) -> None:
        self.grid.set_mask(mask)
        self._on_changed()

    def set_grid_size(self, rows: int, cols: int) -> None:
        self.grid.set_grid_size(rows, cols)

    def set_save_directory(self, path: Optional[str]) -> None:
        """Where a run writes. A method rather than a box the host reaches in and sets:
        the tab used to poke `image_settings_widget.path_edit` through two layers, which
        is a coupling to this widget's internal shape rather than to what it does."""
        self.path_edit.setText(str(path) if path else "")

    def get_settings(self) -> OverviewAcquisitionSettings:
        return OverviewAcquisitionSettings(
            image_settings=ImageSettings(
                resolution=tuple(self.combo_resolution.value()),
                dwell_time=self.spin_dwell.value() * constants.MICRO_TO_SI,
                hfw=self.spin_hfw.value() * constants.MICRO_TO_SI,
                autocontrast=self.check_autocontrast.isChecked(),
                beam_type=self.combo_beam.value(),
                save=True,
                path=self.path_edit.text() or None,
                filename=self.filename_edit.text(),
            ),
            nrows=self.grid.rows,
            ncols=self.grid.cols,
            overlap=self.grid.overlap,
            tile_mask=self.grid.mask,
            focus_stack_settings=FocusStackSettings(
                enabled=self.check_focus_stack.isChecked(),
                n_steps=int(self.spin_focus_steps.value()),
                auto_focus=self.check_focus_autofocus.isChecked(),
            ),
            autofocus_settings=AutoFocusSettings(mode=self.combo_autofocus.value()),
            tile_order=self.grid.tile_order,
        )

    def update_from_settings(self, settings: OverviewAcquisitionSettings) -> None:
        """Load every value without emitting on the way through."""
        image = settings.image_settings
        widgets = [
            self.combo_beam,
            self.combo_resolution,
            self.spin_dwell,
            self.spin_hfw,
            self.check_autocontrast,
            self.check_focus_stack,
            self.spin_focus_steps,
            self.check_focus_autofocus,
            self.combo_autofocus,
            self.filename_edit,
            self.path_edit.lineEdit,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        try:
            self.combo_beam.set_value(image.beam_type)
            self.combo_resolution.set_value(list(image.resolution))
            self.spin_dwell.setValue(image.dwell_time * constants.SI_TO_MICRO)
            self.spin_hfw.setValue(image.hfw * constants.SI_TO_MICRO)
            self.check_autocontrast.setChecked(image.autocontrast)
            self.check_focus_stack.setChecked(settings.focus_stack_settings.enabled)
            self.spin_focus_steps.setValue(settings.focus_stack_settings.n_steps)
            self.check_focus_autofocus.setChecked(
                settings.focus_stack_settings.auto_focus
            )
            self.combo_autofocus.set_value(settings.autofocus_settings.mode)
            self.filename_edit.setText(image.filename or "")
            self.path_edit.setText(str(image.path) if image.path else "")
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        self.grid.apply(
            rows=settings.nrows,
            cols=settings.ncols,
            overlap=settings.overlap,
            tile_order=settings.tile_order,
            mask=settings.tile_mask,
        )
        self._on_focus_stack_toggled()
        self._refresh_derived()

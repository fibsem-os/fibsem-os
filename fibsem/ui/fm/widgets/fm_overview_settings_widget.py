"""Every setting an FM overview acquisition takes, in one self-contained widget.

Two properties this widget is built around:

* **No parent coupling.** It is told what it needs rather than reaching into its parent
  for channel names, so it can be embedded anywhere.
* **Complete.** It covers `tile_order` and `tile_mask`, which the acquisition gained
  with sparse tilesets and tile ordering and which nothing could set from the UI.
"""

from typing import List, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.fm.structures import (
    AutoFocusMode,
    AutoFocusSettings,
    ChannelSettings,
    ObjectiveStartPosition,
    OverviewParameters,
    ZParameters,
)
from fibsem.structures import TileOrderStrategy
from fibsem.ui.fm.widgets.autofocus_widget import AutofocusWidget
from fibsem.ui.fm.widgets.z_parameters_widget import ZParametersWidget
from fibsem.ui.tokens import (
    TEXT_MUTED_COLOR,
)
from fibsem.ui.widgets.custom_widgets import (
    QDirectoryLineEdit,
    TitledPanel,
    ValueComboBox,
)
from fibsem.ui.widgets.overview_acquisition_settings_widget import (
    DEFAULT_OVERVIEW_FILENAME,
    stamped_overview_name,
)
from fibsem.ui.widgets.overview_grid_settings_widget import (
    OverviewGridSettingsWidget,
)

MUTED = f"color: {TEXT_MUTED_COLOR}; font-size: 11px;"

OBJECTIVE_START_LABELS = {
    ObjectiveStartPosition.CURRENT: "Current position",
    ObjectiveStartPosition.FOCUS: "Saved focus position",
}


def format_objective_start(start, position: Optional[float]) -> str:
    """Label one start option, with what it currently means in millimetres.

    An option that names a position should say which one -- "saved focus position" is
    only actionable if you can see it is 8.000 mm and the objective is at 6.000. Shared
    with the confirmation dialog so the two cannot describe the same run differently.
    """
    label = OBJECTIVE_START_LABELS.get(start, start.name)
    if position is None:
        return f"{label} (none saved)"
    return f"{label} ({position * constants.SI_TO_MILLI:.3f} mm)"


OBJECTIVE_START_TOOLTIPS = {
    ObjectiveStartPosition.CURRENT: "Start from wherever the objective is now.",
    ObjectiveStartPosition.FOCUS: (
        "Start from the focus saved with 'Set Focus Position'. Refuses to run if none "
        "has been saved."
    ),
}

AUTOFOCUS_LABELS = {
    AutoFocusMode.NONE: "Don't auto-focus",
    AutoFocusMode.ONCE: "Once, at the start",
    AutoFocusMode.EACH_ROW: "On each new row",
    AutoFocusMode.EACH_TILE: "On every tile",
}

SPINBOX_MIN_WIDTH = 92


class FMOverviewSettingsWidget(QWidget):
    """Grid, sampling and autofocus settings for a tileset acquisition."""

    changed = pyqtSignal()

    def __init__(
        self,
        parameters: Optional[OverviewParameters] = None,
        channel_settings: Optional[List[ChannelSettings]] = None,
        z_parameters: Optional[ZParameters] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._tile_fov: Optional[tuple] = None  # (fov_x, fov_y) metres, set externally
        # What the start options currently mean. None until a host says -- the standalone
        # settings widget has no microscope to ask.
        self._objective_current: Optional[float] = None
        self._objective_focus: Optional[float] = None
        # Owned here rather than by the parent, so every overview setting sits in one
        # widget and their order is this widget's to decide.
        self.z_widget = ZParametersWidget(z_parameters or ZParameters())
        # The sweep itself -- method, channel, and the coarse/fine passes -- is already
        # a solved widget backed by `AutoFocusSettings`. Only *when* to run it is an
        # overview concern, so that is all this widget adds.
        self.autofocus_widget = AutofocusWidget(list(channel_settings or []))
        self.autofocus_widget.set_pass_editing_enabled(True)
        self._init_ui()
        # Always seeded, from the dataclass when the caller supplies nothing. The grid
        # controls used to carry their own defaults -- `spin_overlap.setValue(0.1)` --
        # which is `OverviewParameters.overlap` written a second time, and the shared
        # panel cannot carry it because the beam side defaults to 0.0 for the same
        # setting. So the dataclass is the one place it is stated.
        self.parameters = parameters if parameters is not None else OverviewParameters()
        self._refresh_derived()

    # ── construction ─────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        # Rows, columns, overlap, tile order and the mask, from the module written to
        # end their being written twice. What stays here is everything that decides
        # what a *tile* is -- channels, exposure, z, focus -- which is the half that is
        # genuinely modality-specific.
        self.grid = OverviewGridSettingsWidget()

        # No label: this sits in the Z-Stack panel header, which already names it.
        self.check_zstack = QCheckBox()
        self.check_zstack.setToolTip("Acquire a z-stack at each tile")

        self.combo_autofocus_mode = ValueComboBox(
            items=list(AutoFocusMode),
            format_fn=lambda m: AUTOFOCUS_LABELS.get(m, m.name),
        )

        # Above the auto-focus mode, because it decides where a sweep starts from.
        self.combo_objective_start = ValueComboBox(
            items=list(ObjectiveStartPosition),
            format_fn=lambda s: OBJECTIVE_START_LABELS.get(s, s.name),
        )
        for index, start in enumerate(ObjectiveStartPosition):
            self.combo_objective_start.setItemData(
                index, OBJECTIVE_START_TOOLTIPS.get(start, ""), Qt.ToolTipRole
            )

        # The app's spinboxes carry -/+ buttons, which eat most of a narrow field and
        # leave the value clipped. Give them a floor. The grid panel applies the same
        # rule to its own fields, for the same reason.
        for widget in (self.combo_autofocus_mode, self.combo_objective_start):
            widget.setMinimumWidth(SPINBOX_MIN_WIDTH)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # "When to focus" is the overview's own setting; everything below it -- method,
        # channel, and the sweep passes -- belongs to the sweep and is delegated.
        focus_form = QFormLayout()
        focus_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        focus_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        focus_form.addRow("Start from", self.combo_objective_start)
        focus_form.addRow("Auto-focus", self.combo_autofocus_mode)

        focus_layout = QVBoxLayout()
        focus_layout.setContentsMargins(6, 6, 6, 6)
        focus_layout.setSpacing(6)
        focus_layout.addLayout(focus_form)
        focus_layout.addWidget(self.autofocus_widget)
        focus_content = QWidget()
        focus_content.setLayout(focus_layout)
        self.focus_panel = TitledPanel("Focus", content=focus_content)

        # The enable checkbox goes in the panel header, not the body: it is the one
        # piece of z-stack state worth seeing when the panel is folded away, and the
        # parameters below it grey out when it is off rather than staying live and
        # implying they apply.
        zstack_layout = QVBoxLayout()
        zstack_layout.setContentsMargins(6, 6, 6, 6)
        zstack_layout.setSpacing(4)
        zstack_layout.addWidget(self.z_widget)
        zstack_content = QWidget()
        zstack_content.setLayout(zstack_layout)
        self.zstack_panel = TitledPanel("Z-Stack", content=zstack_content)
        self.zstack_panel.add_header_widget(self.check_zstack)

        # Where a run lands. Same two fields and the same rule as the FIB/SEM tab:
        # the box holds a base name and the run stamps it with the time it started, so
        # two runs cannot land on each other. `OverviewDestination` steps a taken name
        # besides, which covers two runs inside the same second.
        self.path_edit = QDirectoryLineEdit()
        self.path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.filename_edit = QLineEdit(DEFAULT_OVERVIEW_FILENAME)
        self.filename_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.label_destination = QLabel("")
        self.label_destination.setStyleSheet(MUTED)

        output_form = QFormLayout()
        output_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        output_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        output_form.setContentsMargins(6, 6, 6, 6)
        output_form.setSpacing(6)
        output_form.addRow("Folder", self.path_edit)
        output_form.addRow("Name", self.filename_edit)
        output_form.addRow("", self.label_destination)
        output_content = QWidget()
        output_content.setLayout(output_form)
        self.output_panel = TitledPanel("Output", content=output_content)

        self.label_summary = QLabel()
        self.label_summary.setStyleSheet(MUTED)
        self.label_summary.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.focus_panel)
        layout.addWidget(self.zstack_panel)
        # One panel, not two: the mask is a grid property, and the shared widget holds
        # it inside the Grid panel with the count in the header. No stretch on any of
        # them -- a stretched panel keeps claiming vertical space when it is collapsed,
        # so folding it leaves a tall empty box with the title floating in the middle.
        # The trailing stretch below takes the slack instead, kept here rather than left
        # to the host so the panels stay put in any container.
        layout.addWidget(self.grid)
        # Last, and folded: the folder is set by whoever owns the experiment and the
        # name rarely changes, so this is the panel you open when you want something
        # other than the default rather than one you touch each run.
        layout.addWidget(self.output_panel)
        layout.addWidget(self.label_summary)
        layout.addStretch()

        # Grid and the tile mask are what you touch every run; focus and the z-stack are
        # set once and then left alone, so they start folded to keep the column short.
        self.focus_panel.collapse()
        self.zstack_panel.collapse()
        self.output_panel.collapse()

        # One signal for all five: the shared panel resizes the mask with the grid and
        # reports every change through `changed`.
        self.grid.changed.connect(self._on_grid_changed)
        self.filename_edit.textChanged.connect(self._refresh_destination)
        self.path_edit.lineEdit.textChanged.connect(self._refresh_destination)
        self._refresh_destination()
        self.check_zstack.stateChanged.connect(self._on_zstack_toggled)
        self.combo_autofocus_mode.currentIndexChanged.connect(
            self._on_autofocus_changed
        )
        self.combo_objective_start.currentIndexChanged.connect(self._on_any_change)
        self.autofocus_widget.settings_changed.connect(self._on_any_change)

        self._on_autofocus_changed()
        self._on_zstack_toggled()

    # ── reactions ────────────────────────────────────────────────────────

    def _on_zstack_toggled(self) -> None:
        self.z_widget.setEnabled(self.check_zstack.isChecked())
        self._on_any_change()

    def set_grid_size(self, rows: int, cols: int) -> None:
        """Set both grid dimensions as a single change.

        Delegated. Setting the two spin boxes one after the other emits twice and passes
        through a size nobody asked for -- the new row count against the old column
        count -- which is invisible when a box is nudged by hand and constant when an
        edge is dragged on the canvas, where it fires on every motion event. The shared
        panel is where that is handled now.
        """
        self.grid.set_grid_size(rows, cols)

    @property
    def tile_mask(self) -> Optional[List[List[bool]]]:
        """Which tiles the next run would acquire, or None for all of them."""
        return self.grid.mask

    def set_mask(self, mask: Optional[List[List[bool]]]) -> None:
        self.grid.set_mask(mask)

    def add_grid_header_widget(self, widget) -> None:
        """Put a control in the Grid panel's header. See `OverviewGridSettingsWidget`."""
        self.grid.add_header_widget(widget)

    def _on_grid_changed(self) -> None:
        """Anything in the grid panel moved -- including the tile order, which is half
        of the spiral/per-row conflict."""
        self._warn_if_spiral_conflicts()
        self._on_any_change()

    def set_objective_positions(
        self, current: Optional[float], focus: Optional[float]
    ) -> None:
        """Tell the start options what they currently mean, in metres.

        Passed in rather than read: this widget takes no microscope, which is what lets
        it be built and tested on its own, and reading the objective is not free -- on
        a shared connection it takes the imaging channel (FIB-517).
        """
        self._objective_current = current
        self._objective_focus = focus
        combo = self.combo_objective_start
        # No `blockSignals`: `setItemText` changes a label, not the selection, so this
        # cannot read as a settings edit to anything listening on `currentIndexChanged`.
        # Repopulating the items instead would, which is why it does not.
        for index in range(combo.count()):
            start = combo.itemData(index)
            position = current if start is ObjectiveStartPosition.CURRENT else focus
            combo.setItemText(index, format_objective_start(start, position))

    def _on_autofocus_changed(self) -> None:
        mode = self.combo_autofocus_mode.value()
        self.autofocus_widget.setEnabled(mode is not AutoFocusMode.NONE)
        self._warn_if_spiral_conflicts()
        self._on_any_change()

    def _warn_if_spiral_conflicts(self) -> None:
        """A spiral revisits rows out of order, so the runner promotes EACH_ROW.

        Surfaced here rather than left as a log line, so the setting the user chose and
        the setting that will run are not quietly different things.
        """
        spiral = self.grid.tile_order is TileOrderStrategy.SPIRAL
        each_row = self.combo_autofocus_mode.value() is AutoFocusMode.EACH_ROW
        self._spiral_conflict = spiral and each_row

    def _on_any_change(self) -> None:
        self._refresh_derived()
        self.changed.emit()

    # ── derived display ──────────────────────────────────────────────────

    def set_tile_fov(self, fov_x: float, fov_y: float) -> None:
        """Tell the widget one tile's field of view, so it can report total area."""
        self._tile_fov = (fov_x, fov_y)
        # Kept here as well as handed on, because `total_fov` answers None until a tile
        # size is known and the shared panel answers (0, 0) -- a distinction its own
        # label does not need and this widget's callers do.
        self.grid.set_tile_fov(fov_x, fov_y)
        self._refresh_derived()

    def total_fov(self) -> Optional[tuple]:
        """Width and height of the whole mosaic in metres, or None if the tile FOV
        has not been supplied.

        The grid always spans the full rectangle regardless of the mask: skipped tiles
        keep their place, so the area covered is a property of rows/cols/overlap alone.
        """
        if self._tile_fov is None:
            return None
        return self.grid.total_fov

    def _refresh_derived(self) -> None:
        # Warnings only -- the tile count is already on the mask panel, and saying it
        # twice just makes the second one look like a different number.
        parts = []
        if getattr(self, "_spiral_conflict", False):
            parts.append("per-row focus becomes per-tile for a spiral")
        if self._focus_without_passes():
            parts.append("auto-focus is on but every sweep pass is disabled")
        self.label_summary.setText(" · ".join(parts))

    def _focus_without_passes(self) -> bool:
        """Focusing scheduled, but nothing for it to sweep.

        `AutoFocusSettings.enabled` is False when every pass is unticked, so the run
        would schedule focusing and then do nothing -- two controls that each look set
        correctly, contradicting each other.
        """
        settings = self.autofocus_settings
        return settings is not None and not settings.enabled

    # ── value ────────────────────────────────────────────────────────────

    def set_channel_names(self, names: List[str]) -> None:
        current = self.combo_autofocus_channel.value()
        self.combo_autofocus_channel.blockSignals(True)
        self.combo_autofocus_channel.clear()
        self.combo_autofocus_channel.add_values(list(names))
        if current in names:
            self.combo_autofocus_channel.set_value(current)
        self.combo_autofocus_channel.blockSignals(False)

    def set_channel_settings(self, channels: List[ChannelSettings]) -> None:
        """Keep the focus-channel choices in step with the channels being acquired."""
        self.autofocus_widget.update_channels(list(channels))

    @property
    def autofocus_channel_name(self) -> Optional[str]:
        channel = self.autofocus_widget.get_selected_channel()
        return channel.name if channel is not None else None

    @property
    def autofocus_settings(self) -> Optional[AutoFocusSettings]:
        """The sweep to run, or None when autofocus is off.

        None rather than settings-behind-a-disabled-control, matching how the
        acquisition already reads `autofocus_settings=None`: one place decides whether
        focusing happens, and it is the mode.
        """
        if self.combo_autofocus_mode.value() is AutoFocusMode.NONE:
            return None
        return self.autofocus_widget.get_autofocus_settings()

    @property
    def z_parameters(self) -> Optional[ZParameters]:
        """The z-stack settings, or None when z-stacking is off.

        None rather than the values behind a disabled checkbox: the acquisition takes
        `zparams=None` to mean "no z-stack", so this hands back exactly what it wants
        and there is no second place deciding whether the setting applies.
        """
        if not self.check_zstack.isChecked():
            return None
        return self.z_widget.z_parameters

    # ── where a run lands ────────────────────────────────────────────────

    def set_save_directory(self, path: Optional[str]) -> None:
        """Where a run writes, as the host understands it.

        A method rather than a box the host reaches into, matching the FIB/SEM tab: the
        widget is handed a microscope and knows nothing about experiments, so the folder
        arrives the same way the positions and the channels do.
        """
        self.path_edit.setText(str(path) if path else "")

    @property
    def save_directory(self) -> Optional[str]:
        """The folder as it stands, which is the box rather than what the host last set:
        a run goes where the field says, and the field is editable on purpose."""
        return self.path_edit.text().strip() or None

    @property
    def basename(self) -> str:
        """The name a run claims, stamped with the time it starts.

        Empty falls back to the default rather than to an unnamed run: the stem is a
        directory name, and `OverviewDestination` would otherwise be handed an empty
        string to make a folder from.
        """
        return stamped_overview_name(
            self.filename_edit.text().strip() or DEFAULT_OVERVIEW_FILENAME
        )

    def _refresh_destination(self) -> None:
        """Say what the run will actually be called, since the box does not show it."""
        name = self.filename_edit.text().strip() or DEFAULT_OVERVIEW_FILENAME
        self.label_destination.setText(f"saved as {name}-HH-MM-SS")

    @property
    def parameters(self) -> OverviewParameters:
        return OverviewParameters(
            rows=self.grid.rows,
            cols=self.grid.cols,
            overlap=self.grid.overlap,
            use_zstack=self.check_zstack.isChecked(),
            autofocus_mode=self.combo_autofocus_mode.value(),
            tile_order=self.grid.tile_order,
            tile_mask=self.grid.mask,
            objective_start=self.combo_objective_start.value(),
        )

    @parameters.setter
    def parameters(self, value: OverviewParameters) -> None:
        widgets = (
            self.check_zstack,
            self.combo_autofocus_mode,
            self.combo_objective_start,
        )
        for widget in widgets:
            widget.blockSignals(True)
        try:
            # The grid panel does its own blocking, so it is loaded outside this one --
            # `blockSignals` on the panel would not reach the boxes inside it anyway.
            self.grid.apply(
                value.rows, value.cols, value.overlap, value.tile_order, value.tile_mask
            )
            self.check_zstack.setChecked(value.use_zstack)
            self.combo_autofocus_mode.set_value(value.autofocus_mode)
            self.combo_objective_start.set_value(value.objective_start)
        finally:
            for widget in widgets:
                widget.blockSignals(False)
        # Both, because the setter blocked the signals that normally trigger them --
        # otherwise loading a z-stacked configuration leaves the z-parameters greyed
        # out while the checkbox says they apply.
        self._on_autofocus_changed()
        self._on_zstack_toggled()

"""The Coincident Milling task as the operator thinks of it (FIB-985).

The generic task form renders this task as two configs' worth of knobs: the
task's own fields, then a full milling config with the strategy settings
repeated per stage. The operator thinks of the monitoring settings as one
setting for the mill, the strategy is always Coincidence, and the pattern
position is placed by the setup step.

This widget is a *projection* of the existing config that writes through. A
monitoring value typed once lands on every coincidence stage's strategy
config; the protocol yaml, the setup record and what ``MillCoincidentTask``
runs keep their shape. Nothing new is stored.

Sections:

- **Monitoring** (shared by every stage): the mill's monitoring channel, with
  a "Copy from…" picker seeded from the protocol's fluorescence task; drop %,
  warmup, confirm-over frames, rolling window, timeout, timelapse frequency.
- **Milling**: the stage list and per-stage milling/pattern panels, with the
  Strategy panel hidden. A stage added here gets the Coincidence strategy and
  the current monitoring values.
- **Setup** (read-only): the site's record from Setup Coincidence Milling,
  when the host is the per-lamella editor.
- **Advanced** (collapsed): field of view, alignment, the post-mill z-stack.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Callable, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from fibsem.applications.autolamella.workflows.tasks.mill_coincident import (
    MILL_COINCIDENT_KEY,
    MillCoincidentTaskConfig,
)
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets import ChannelSettingsWidget
from fibsem.ui.widgets.custom_widgets import TitledPanel
from fibsem.ui.widgets.milling_task_viewer_widget import MillingTaskViewerWidget

if TYPE_CHECKING:
    from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
        SetupCoincidenceMillingTaskConfig,
    )
    from fibsem.microscope import FibsemMicroscope
    from fibsem.milling.tasks import FibsemMillingTaskConfig

_HINT_STYLE = "color: #868e93; font-size: 11px;"


class _StageTable(QWidget):
    """One row per stage: name, direction, current, width, height, depth.

    Edits write straight onto the live stage objects the full editor holds, so
    the two views never disagree; the widget re-syncs the list after each edit.
    Direction lives on rectangle-like patterns only; a trench derives its own
    order from its two halves, so its cell is blank.
    """

    stage_edited = pyqtSignal()
    add_requested = pyqtSignal()
    remove_requested = pyqtSignal(object)  # FibsemMillingStage

    _HEADERS = ("Stage", "Direction", "Current", "Width", "Height", "Depth", "")

    def __init__(self, microscope, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._microscope = microscope
        self._stages: list = []
        self._loading = False
        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setHorizontalSpacing(6)
        self._layout.setVerticalSpacing(3)
        self._rows: list = []
        self._directions: List[str] = []
        self._currents: List[float] = []
        try:
            from fibsem.structures import BeamType

            self._directions = list(
                microscope.get_available_values_cached("scan_direction", BeamType.ION)
                or []
            )
            self._currents = list(
                microscope.get_available_values_cached("current", BeamType.ION) or []
            )
        except Exception:
            logging.debug("Stage table: no available values from the microscope")
        self._build_header()
        # the name gets the slack; the numeric cells stay compact
        self._layout.setColumnStretch(0, 3)
        for col in range(1, 6):
            self._layout.setColumnStretch(col, 0)
        self.btn_add = QToolButton()
        self.btn_add.setText("+ Add stage")
        self.btn_add.clicked.connect(self.add_requested.emit)

    def _build_header(self) -> None:
        for col, text in enumerate(self._HEADERS):
            lbl = QLabel(text)
            lbl.setStyleSheet(
                "color: #868e93; font-size: 10px; text-transform: uppercase;"
            )
            self._layout.addWidget(lbl, 0, col)

    def set_stages(self, stages) -> None:
        """Rebuild the rows from *stages* (live objects, not copies)."""
        self._loading = True
        try:
            for widgets in self._rows:
                for w in widgets:
                    # out of the layout AND out of sight now: deleteLater only
                    # runs once control returns to the event loop, and a stale
                    # row painted over the new one is exactly what a host that
                    # rebuilds twice in one call would show
                    self._layout.removeWidget(w)
                    w.hide()
                    w.setParent(None)
                    w.deleteLater()
            self._rows = []
            self._layout.removeWidget(self.btn_add)
            self._stages = list(stages)
            for i, stage in enumerate(self._stages):
                self._add_row(i + 1, stage)
            self._layout.addWidget(self.btn_add, len(self._stages) + 1, 0, 1, 2)
        finally:
            self._loading = False

    def _add_row(self, row: int, stage) -> None:
        from PyQt5.QtWidgets import QComboBox, QLineEdit

        pattern = stage.pattern
        widgets = []

        name = QLineEdit(stage.name)
        name.editingFinished.connect(
            lambda s=stage, e=name: self._set(s, "name", e.text())
        )
        widgets.append(name)

        direction = QComboBox()
        has_direction = hasattr(pattern, "scan_direction")
        if has_direction:
            items = self._directions or [pattern.scan_direction]
            if pattern.scan_direction not in items:
                items = [pattern.scan_direction] + list(items)
            direction.addItems([str(d) for d in items])
            direction.setCurrentText(str(pattern.scan_direction))
            direction.currentTextChanged.connect(
                lambda text, p=pattern: self._set(p, "scan_direction", text)
            )
        else:
            # a trench mills its two halves in a fixed order: say what it is
            # rather than leave a blank the operator will try to fill
            direction.addItem(type(pattern).__name__.replace("Pattern", ""))
            direction.setEnabled(False)
            direction.setToolTip(
                f"{type(pattern).__name__} has no scan direction of its own."
            )
        direction.setMinimumWidth(100)
        widgets.append(direction)

        current = QComboBox()
        currents = self._currents or [stage.milling.milling_current]
        if stage.milling.milling_current not in currents:
            currents = [stage.milling.milling_current] + list(currents)
        for c in currents:
            current.addItem(f"{c * 1e12:.1f} pA", c)
        current.setCurrentIndex(currents.index(stage.milling.milling_current))
        current.currentIndexChanged.connect(
            lambda idx, s=stage, cb=current: self._set(
                s.milling, "milling_current", cb.itemData(idx)
            )
        )
        widgets.append(current)

        for attr in ("width", "height", "depth"):
            spin = QDoubleSpinBox()
            spin.setDecimals(2)
            spin.setRange(0.01, 500.0)
            spin.setSuffix(" µm")
            spin.setFixedWidth(88)
            target_attrs = self._pattern_attrs(pattern, attr)
            if target_attrs:
                spin.setValue(float(getattr(pattern, target_attrs[0])) * 1e6)
                spin.valueChanged.connect(
                    lambda v, p=pattern, attrs=target_attrs: self._set_many(
                        p, attrs, v * 1e-6
                    )
                )
                if len(target_attrs) > 1:
                    spin.setToolTip(
                        "Both trench halves; see All stage settings to differ."
                    )
            else:
                spin.setEnabled(False)
                spin.setSpecialValueText("—")
                spin.setValue(spin.minimum())
                spin.setToolTip(
                    f"{type(pattern).__name__} has no {attr}; see All stage settings."
                )
            widgets.append(spin)

        remove = QToolButton()
        remove.setText("✕")
        remove.setToolTip("Remove this stage")
        remove.clicked.connect(lambda _=False, s=stage: self.remove_requested.emit(s))
        widgets.append(remove)

        for col, w in enumerate(widgets):
            self._layout.addWidget(w, row, col)
        self._rows.append(widgets)

    @staticmethod
    def _pattern_attrs(pattern, attr: str) -> List[str]:
        """Which pattern fields a table column edits; a trench's height is its two halves."""
        if hasattr(pattern, attr):
            return [attr]
        if attr == "height" and hasattr(pattern, "upper_trench_height"):
            return ["upper_trench_height", "lower_trench_height"]
        return []

    def _set(self, target, attr: str, value) -> None:
        if self._loading:
            return
        setattr(target, attr, value)
        self.stage_edited.emit()

    def _set_many(self, target, attrs: List[str], value) -> None:
        if self._loading:
            return
        for attr in attrs:
            setattr(target, attr, value)
        self.stage_edited.emit()


class AutoLamellaCoincidentMillingTaskConfigWidget(QWidget):
    """Edit a :class:`MillCoincidentTaskConfig` as one mill, not per stage."""

    settings_changed = pyqtSignal(object)  # MillCoincidentTaskConfig

    def __init__(
        self,
        microscope: "FibsemMicroscope",
        config: Optional[MillCoincidentTaskConfig] = None,
        parent: Optional[QWidget] = None,
        channel_sources: Optional[Callable[[], List[ChannelSettings]]] = None,
        detach_milling: bool = False,
    ):
        super().__init__(parent)
        self.microscope = microscope
        self.config = config if config is not None else MillCoincidentTaskConfig()
        # where "Copy from…" gets its channels: the host knows the protocol
        self._channel_sources = channel_sources
        # A host with a milling column of its own (the protocol-level editor)
        # places ``milling_panel`` there itself; everything else stays here.
        self._detach_milling = detach_milling
        self._loading = False
        self._setup_ui()
        self._connect_signals()
        self.set_task_config(self.config)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        # ── Channel Settings ──────────────────────────────────────────────
        channel_box = QWidget()
        channel_layout = QVBoxLayout(channel_box)
        channel_layout.setContentsMargins(4, 4, 4, 4)
        channel_layout.setSpacing(4)

        # ── Stop Condition ────────────────────────────────────────────────
        monitoring = QWidget()
        grid = QGridLayout(monitoring)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(4)

        fm = getattr(self.microscope, "fm", None)
        self.channel_widget = ChannelSettingsWidget(fm=fm, parent=self)
        self.btn_copy_channel = QToolButton()
        self.btn_copy_channel.setText("Copy from…")
        self.btn_copy_channel.setPopupMode(QToolButton.InstantPopup)
        self.btn_copy_channel.setToolTip(
            "Seed the monitoring channel from one of the protocol's fluorescence "
            "imaging channels. Exposure and power usually want to be lower here: "
            "monitoring runs for minutes, a z-stack runs once."
        )
        self._copy_menu = QMenu(self.btn_copy_channel)
        self._copy_menu.aboutToShow.connect(self._populate_copy_menu)
        self.btn_copy_channel.setMenu(self._copy_menu)
        self.btn_copy_channel.setVisible(self._channel_sources is not None)

        # the form's own titled header would repeat this panel's title (or the
        # channel's name) directly beneath it; the panel is the header here
        self.channel_widget._panel._header.setVisible(False)
        channel_layout.addWidget(self.channel_widget)
        channel_hint = QLabel(
            "The channel the mill watches. Short exposure, low power: it runs for minutes."
        )
        channel_hint.setStyleSheet(_HINT_STYLE)
        channel_hint.setWordWrap(True)
        channel_layout.addWidget(channel_hint)

        row = 0
        self.spin_drop = QSpinBox()
        self.spin_drop.setRange(5, 95)
        self.spin_drop.setSuffix(" % drop")
        self.spin_drop.setToolTip(
            "Stop (unsupervised) or alert (supervised) when the rolling mean falls "
            "this far below its peak."
        )
        row = self._add_row(grid, row, "Stop at", self.spin_drop, "below the peak")

        self.spin_warmup = QDoubleSpinBox()
        self.spin_warmup.setRange(0.0, 600.0)
        self.spin_warmup.setDecimals(0)
        self.spin_warmup.setSuffix(" s")
        self.spin_warmup.setToolTip("Frames in this window do not set the peak.")
        row = self._add_row(
            grid, row, "Warmup", self.spin_warmup, "before the peak is tracked"
        )

        self.spin_confirm = QSpinBox()
        self.spin_confirm.setRange(1, 1000)
        self.spin_confirm.setSuffix(" frames")
        self.spin_confirm.setToolTip(
            "How many frames in a row must sit below the threshold before the drop counts."
        )
        row = self._add_row(
            grid, row, "Confirm over", self.spin_confirm, "in a row below the threshold"
        )

        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 1000)
        self.spin_window.setSuffix(" frames")
        self.spin_window.setToolTip(
            "The intensity compared against the threshold is the mean of this many "
            "recent frames, not a single frame."
        )
        row = self._add_row(
            grid,
            row,
            "Rolling mean",
            self.spin_window,
            "frames averaged before the test",
        )

        self.spin_timeout = QDoubleSpinBox()
        self.spin_timeout.setRange(0.5, 150.0)
        self.spin_timeout.setDecimals(1)
        self.spin_timeout.setSuffix(" min")
        self.spin_timeout.setToolTip("The mill stops here whatever the intensity does.")
        row = self._add_row(grid, row, "Timeout", self.spin_timeout, "")

        self.spin_timelapse = QDoubleSpinBox()
        self.spin_timelapse.setRange(0.0, 600.0)
        self.spin_timelapse.setDecimals(1)
        self.spin_timelapse.setSuffix(" s")
        self.spin_timelapse.setToolTip(
            "Save one FM frame to the run folder this often. 0 saves every frame."
        )
        row = self._add_row(
            grid, row, "Timelapse every", self.spin_timelapse, "0 = every frame"
        )
        grid.setColumnStretch(2, 1)

        self.channel_panel = TitledPanel(
            "Channel Settings", content=channel_box, collapsible=True
        )
        # Copy from… lives in the panel header, out of the form's way
        self.channel_panel.add_header_widget(self.btn_copy_channel)
        outer.addWidget(self.channel_panel)
        self.stop_panel = TitledPanel(
            "Stop Condition", content=monitoring, collapsible=True
        )
        outer.addWidget(self.stop_panel)

        # ── Milling ───────────────────────────────────────────────────────
        self.milling_editor = MillingTaskViewerWidget(
            microscope=self.microscope,  # type: ignore[arg-type]
            milling_enabled=False,
            parent=self,
        )
        self.milling_editor.set_parameters_visible(False)
        self.milling_editor.set_alignment_visible(False)
        self.milling_editor.set_acquisition_visible(False)
        self._stages_widget().set_strategy_visible(False)
        milling_note = QLabel(
            "Every stage mills with the Coincidence strategy at one shared position."
        )
        milling_note.setStyleSheet(_HINT_STYLE)
        milling_note.setWordWrap(True)
        milling_box = QWidget()
        milling_layout = QVBoxLayout(milling_box)
        milling_layout.setContentsMargins(4, 4, 4, 4)
        milling_layout.setSpacing(4)
        milling_layout.addWidget(milling_note)
        # the mockup's table: one row per stage, the fields that differ between
        # a top-to-bottom and a bottom-to-top pass
        self.stage_table = _StageTable(self.microscope, parent=self)
        milling_layout.addWidget(self.stage_table)
        # everything else a stage has, behind a fold
        self.all_stage_settings_panel = TitledPanel(
            "All stage settings", content=self.milling_editor, collapsible=True
        )
        self.all_stage_settings_panel.collapse()
        milling_layout.addWidget(self.all_stage_settings_panel)
        self.milling_panel = TitledPanel(
            "Milling", content=milling_box, collapsible=True
        )
        outer.addWidget(self.milling_panel)

        # ── Setup (read-only) ─────────────────────────────────────────────
        self.label_setup = QLabel("")
        self.label_setup.setStyleSheet(
            "color: #d1d2d4; font-family: monospace; font-size: 11px; padding: 4px;"
        )
        self.label_setup.setWordWrap(True)
        self.setup_panel = TitledPanel(
            "Setup", content=self.label_setup, collapsible=True
        )
        outer.addWidget(self.setup_panel)
        self.set_setup_record(None)

        # ── Advanced (collapsed) ──────────────────────────────────────────
        advanced = QWidget()
        adv = QGridLayout(advanced)
        adv.setContentsMargins(4, 4, 4, 4)
        adv.setHorizontalSpacing(8)
        adv.setVerticalSpacing(4)
        self.spin_fov = QDoubleSpinBox()
        self.spin_fov.setRange(5.0, 1000.0)
        self.spin_fov.setDecimals(1)
        self.spin_fov.setSuffix(" µm")
        self.spin_fov.setToolTip("FIB field of view the mill images and mills at.")
        arow = self._add_row(adv, 0, "Field of view", self.spin_fov, "")
        self.chk_alignment = QCheckBox("Beam-shift align between stages")
        adv.addWidget(self.chk_alignment, arow, 0, 1, 3)
        arow += 1
        self.chk_zstack = QCheckBox("Fluorescence z-stack after milling")
        self.chk_zstack.setToolTip(
            "Acquire a z-stack with the lamella's fluorescence task settings once "
            "the mill ends, alongside the FIB reference set."
        )
        adv.addWidget(self.chk_zstack, arow, 0, 1, 3)
        adv.setColumnStretch(2, 1)
        self.advanced_panel = TitledPanel(
            "Advanced", content=advanced, collapsible=True
        )
        self.advanced_panel.collapse()
        outer.addWidget(self.advanced_panel)
        outer.addStretch()

        self.setStyleSheet(stylesheets.NAPARI_STYLE)

    def set_shown(self, visible: bool) -> None:
        """Show or hide the whole widget, a detached milling panel included.

        Hosts call this rather than ``setVisible``: a panel placed in another
        column is not this widget's child, so Qt's own visibility does not reach
        it.
        """
        self.setVisible(visible)
        if self._detach_milling:
            self.milling_panel.setVisible(visible)

    @staticmethod
    def _add_row(
        grid: QGridLayout, row: int, label: str, widget: QWidget, hint: str
    ) -> int:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)
        if hint:
            lbl = QLabel(hint)
            lbl.setStyleSheet(_HINT_STYLE)
            grid.addWidget(lbl, row, 2)
        return row + 1

    def _connect_signals(self) -> None:
        for spin in (
            self.spin_drop,
            self.spin_warmup,
            self.spin_confirm,
            self.spin_window,
            self.spin_timeout,
            self.spin_timelapse,
        ):
            spin.valueChanged.connect(self._on_monitoring_changed)
        self.channel_widget.channel_changed.connect(self._on_channel_changed)
        self.milling_editor.settings_changed.connect(self._on_milling_changed)
        self.stage_table.stage_edited.connect(self._on_table_edited)
        self.stage_table.add_requested.connect(self._on_table_add)
        self.stage_table.remove_requested.connect(self._on_table_remove)
        self.spin_fov.valueChanged.connect(self._on_advanced_changed)
        self.chk_alignment.toggled.connect(self._on_advanced_changed)
        self.chk_zstack.toggled.connect(self._on_advanced_changed)

    # ------------------------------------------------------------------
    # Config in / out
    # ------------------------------------------------------------------

    def set_task_config(self, config: MillCoincidentTaskConfig) -> None:
        """Show *config*; the widget works on its own copy until read back."""
        self._loading = True
        try:
            self.config = copy.deepcopy(config)
            milling = self._milling_config()
            self._ensure_coincidence(milling.stages)
            strategy_config = self._first_strategy_config(milling)

            self.channel_widget.set_channel(self.config.monitoring_channel)
            self.spin_drop.setValue(
                int(round(strategy_config.intensity_drop_fraction * 100))
            )
            self.spin_warmup.setValue(float(strategy_config.warmup_duration))
            self.spin_confirm.setValue(int(strategy_config.consecutive_triggers))
            self.spin_window.setValue(int(strategy_config.rolling_window))
            self.spin_timeout.setValue(float(strategy_config.timeout) / 60.0)
            self.spin_timelapse.setValue(float(strategy_config.save_rate_limit))

            self.milling_editor.set_config(milling)
            self.stage_table.set_stages(self._live_stages())
            self.spin_fov.setValue(float(milling.field_of_view) * 1e6)
            self.chk_alignment.setChecked(bool(milling.alignment.enabled))
            self.chk_zstack.setChecked(bool(self.config.acquire_fluorescence_images))
        finally:
            self._loading = False

    def get_task_config(self) -> MillCoincidentTaskConfig:
        """The config as the widget holds it, monitoring written onto every stage."""
        milling = self.milling_editor.get_config()
        self._ensure_coincidence(milling.stages)
        self._apply_monitoring(milling.stages)
        milling.field_of_view = self.spin_fov.value() * 1e-6
        milling.alignment.enabled = self.chk_alignment.isChecked()
        self.config.milling[MILL_COINCIDENT_KEY] = milling
        self.config.acquire_fluorescence_images = self.chk_zstack.isChecked()
        return self.config

    def set_setup_record(
        self,
        record: Optional["SetupCoincidenceMillingTaskConfig"],
        site_name: Optional[str] = None,
    ) -> None:
        """What Setup Coincidence Milling recorded for the site, read-only.

        None means the host has no site (the protocol-level editor).
        """
        if record is None:
            self.label_setup.setText(
                "Set up per site by Setup Coincidence Milling: objective height, "
                "FM region and milling box position."
            )
            return
        who = f" · {site_name}" if site_name else ""
        if not record.is_set_up:
            self.label_setup.setText(
                f"Not set up{who}. Run Setup Coincidence Milling for this site; "
                "the mill holds it back until then."
            )
            return
        roi = record.fm_roi
        roi_str = (
            f"x {roi.left:.2f} y {roi.top:.2f} · {roi.width:.2f} × {roi.height:.2f}"
            if roi is not None
            else "whole frame"
        )
        offset = record.pattern_offset
        self.label_setup.setText(
            f"Objective  : {record.objective_position * 1e3:.3f} mm{who}\n"
            f"FM region  : {roi_str}\n"
            f"Milling box: {offset.x * 1e6:+.1f} µm, {offset.y * 1e6:+.1f} µm\n"
            f"Stop at    : {record.intensity_drop_fraction * 100:.0f} % drop (site)"
        )

    def set_microscope(self, microscope: "FibsemMicroscope") -> None:
        """Point at a different microscope after a reconnect (see FIB-525)."""
        self.microscope = microscope
        self.channel_widget.set_fm(getattr(microscope, "fm", None))
        self.milling_editor.microscope = microscope

    # ------------------------------------------------------------------
    # Write-through
    # ------------------------------------------------------------------

    def _milling_config(self) -> "FibsemMillingTaskConfig":
        milling = self.config.milling.get(MILL_COINCIDENT_KEY)
        if milling is None:
            # the config's own default: one coincidence stage
            milling = MillCoincidentTaskConfig().milling[MILL_COINCIDENT_KEY]
            self.config.milling[MILL_COINCIDENT_KEY] = milling
        return milling

    @staticmethod
    def _first_strategy_config(milling) -> CoincidenceMillingStrategyConfig:
        for stage in milling.stages:
            if isinstance(stage.strategy, CoincidenceMillingStrategy):
                return stage.strategy.config
        return CoincidenceMillingStrategyConfig()

    def _ensure_coincidence(self, stages) -> None:
        """Every stage mills with the Coincidence strategy; a new one is converted."""
        template = None
        for stage in stages:
            if isinstance(stage.strategy, CoincidenceMillingStrategy):
                template = stage.strategy.config
                break
        for stage in stages:
            if isinstance(stage.strategy, CoincidenceMillingStrategy):
                continue
            stage.strategy = CoincidenceMillingStrategy(
                config=copy.deepcopy(template)
                if template is not None
                else CoincidenceMillingStrategyConfig()
            )

    def _apply_monitoring(self, stages) -> None:
        """The monitoring controls onto every coincidence stage's strategy config."""
        for stage in stages:
            strategy = stage.strategy
            if not isinstance(strategy, CoincidenceMillingStrategy):
                continue
            cfg = strategy.config
            cfg.intensity_drop_fraction = self.spin_drop.value() / 100.0
            cfg.warmup_duration = float(self.spin_warmup.value())
            cfg.consecutive_triggers = int(self.spin_confirm.value())
            cfg.rolling_window = int(self.spin_window.value())
            cfg.timeout = int(round(self.spin_timeout.value() * 60.0))
            cfg.save_rate_limit = float(self.spin_timelapse.value())
            cfg.save_fm_images = True

    def _live_stages(self):
        return self._stages_widget().get_stages()

    def _stages_widget(self):
        return self.milling_editor.config_widget.milling_stages_widget

    def _on_monitoring_changed(self, *_) -> None:
        if self._loading:
            return
        # write through onto the editor's live stages, so a later get_config
        # reads them back with the values the operator sees
        stages = self._live_stages()
        self._ensure_coincidence(stages)
        self._apply_monitoring(stages)
        self._emit()

    def _on_channel_changed(self, channel: ChannelSettings) -> None:
        if self._loading:
            return
        self.config.monitoring_channel = channel
        self._emit()

    def _on_milling_changed(self, milling) -> None:
        if self._loading:
            return
        # a stage added in the list arrives with the default strategy: convert it
        # and give it the monitoring values before anyone reads the config
        stages = self._live_stages()
        self._ensure_coincidence(stages)
        self._apply_monitoring(stages)
        self.stage_table.set_stages(stages)
        self._emit()

    def _on_table_edited(self) -> None:
        """A cell in the compact table wrote onto a live stage: sync the full editor."""
        if self._loading:
            return
        self._stages_widget()._list.refresh_all()
        self._emit()

    def _on_table_add(self) -> None:
        """Add a stage as the list's own "+" would (a copy of the last one)."""
        self._stages_widget()._list._on_add_stage()

    def _on_table_remove(self, stage) -> None:
        self._stages_widget()._list.remove_stage(stage)
        self._stages_widget()._on_stage_removed(stage)

    def _on_advanced_changed(self, *_) -> None:
        if self._loading:
            return
        self._emit()

    def _emit(self) -> None:
        try:
            self.settings_changed.emit(self.get_task_config())
        except Exception:
            logging.exception("Coincident milling task widget: could not emit config")

    # ------------------------------------------------------------------
    # Copy from… the protocol's fluorescence channels
    # ------------------------------------------------------------------

    def _populate_copy_menu(self) -> None:
        self._copy_menu.clear()
        sources: List[ChannelSettings] = []
        if self._channel_sources is not None:
            try:
                sources = list(self._channel_sources())
            except Exception:
                logging.exception(
                    "Could not read the fluorescence channels to copy from"
                )
        if not sources:
            action = self._copy_menu.addAction("No fluorescence task channels")
            action.setEnabled(False)
            return
        for channel in sources:
            action = self._copy_menu.addAction(channel.pretty_name)
            action.triggered.connect(
                lambda _checked=False, c=channel: self._copy_channel(c)
            )

    def _copy_channel(self, source: ChannelSettings) -> None:
        """Take the source's filters and name; keep our exposure and power.

        Monitoring runs for minutes: the imaging exposure and power are the
        wrong starting point, so they are left as they are.
        """
        target = self.config.monitoring_channel
        target.name = source.name
        target.excitation_wavelength = source.excitation_wavelength
        target.emission_wavelength = source.emission_wavelength
        target.color = source.color
        self.channel_widget.set_channel(target)
        self._emit()

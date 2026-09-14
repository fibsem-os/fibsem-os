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

        # ── Monitoring ────────────────────────────────────────────────────
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

        channel_row = QHBoxLayout()
        channel_row.setContentsMargins(0, 0, 0, 0)
        channel_row.addWidget(self.channel_widget, 1)
        copy_col = QVBoxLayout()
        copy_col.addWidget(self.btn_copy_channel)
        copy_col.addStretch()
        channel_row.addLayout(copy_col)
        grid.addLayout(channel_row, 0, 0, 1, 3)

        row = 1
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
        self.spin_window = QSpinBox()
        self.spin_window.setRange(1, 1000)
        self.spin_window.setPrefix("window ")
        self.spin_window.setToolTip("Frames averaged into the rolling mean.")
        confirm_row = QHBoxLayout()
        confirm_row.setContentsMargins(0, 0, 0, 0)
        confirm_row.addWidget(self.spin_confirm)
        confirm_row.addWidget(self.spin_window)
        confirm_row.addStretch()
        grid.addWidget(QLabel("Confirm over"), row, 0)
        grid.addLayout(confirm_row, row, 1, 1, 2)
        row += 1

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

        self.monitoring_panel = TitledPanel(
            "Monitoring", content=monitoring, collapsible=True
        )
        outer.addWidget(self.monitoring_panel)

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
        milling_layout.addWidget(self.milling_editor)
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
        self._emit()

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

"""Pre-flight summary for a coincidence mill.

Replaces the stock `QMessageBox` in `FluorescenceCoincidenceViewerWidget._run_milling`,
which put the only per-stage detail behind "Show Details" as a `pformat` of the raw
config dict — developer output in the one place an operator checks depth and current
before committing.

Same house style as :class:`FMOverviewConfirmationDialog`: a meta line, count chips, a
detail block, and a primary action in the footer — with the visual constants, chip and
duration format imported from it so the two cannot drift. If a third of these appears,
lift the shared parts into their own module.

Milling is irreversible, so the primary action is never the default button: starting
takes a deliberate click.
"""

from typing import List, Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui import stylesheets
from fibsem.ui.fm.widgets.fm_overview_confirmation_dialog import (
    BORDER,
    PANEL,
    TEXT,
    TEXT_MUTED,
    TEXT_STRONG,
    _chip,
    format_duration,
)

BACKGROUND = "#262930"

# The dimensions that determine the milled volume, in the order they read. Driven off
# the pattern's own to_dict() rather than isinstance checks, so a new pattern type
# degrades to whichever of these it happens to have instead of showing nothing.
#
# Deliberately not every field: trench spacing and the upper/lower heights are visible
# as the pattern's shape on the canvas, and listing them here pushed the stage line to
# two wrapped rows where the separator read as a bullet. Full detail stays in the task
# config editor; this is the pre-flight check.
_DIMENSION_KEYS: List[Tuple[str, str]] = [
    ("width", "wide"),
    ("height", "tall"),
    ("depth", "deep"),
]


def _format_current(amps: float) -> str:
    """`1.0 nA` / `60 pA` — milling currents span three orders of magnitude, and
    everything-in-pA turns the common nA case into a four-digit number."""
    if amps >= 1e-9:
        return f"{amps * constants.SI_TO_NANO:.1f} nA"
    return f"{amps * constants.SI_TO_PICO:.0f} pA"


def _pattern_summary(stage: FibsemMillingStage) -> str:
    """`Rectangle 12.0 wide, 1.5 tall, 0.5 deep µm`, minus whatever it lacks."""
    pattern = stage.pattern
    try:
        values = pattern.to_dict()
    except Exception:
        return getattr(pattern, "name", "pattern")

    bits = [
        f"{values[key] * constants.SI_TO_MICRO:.1f} {label}"
        for key, label in _DIMENSION_KEYS
        if isinstance(values.get(key), (int, float)) and values.get(key)
    ]
    name = getattr(pattern, "name", "Pattern")
    return f"{name} {', '.join(bits)} µm" if bits else name


class CoincidenceMillingConfirmationDialog(QDialog):
    """Confirm a coincidence mill before it runs, with what it will do."""

    def __init__(
        self,
        task_config: FibsemMillingTaskConfig,
        lamella_name: str,
        channel_name: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.task_config = task_config
        self.lamella_name = lamella_name
        self.channel_name = channel_name

        self.setWindowTitle("Start Coincidence Milling")
        # Wide enough that a three-dimension pattern plus current and duration fits on
        # one line; the stage rows are the longest thing here and wrapping them splits
        # a stage across two lines for no gain.
        self.setMinimumWidth(540)
        self.setStyleSheet(f"QDialog {{ background: {BACKGROUND}; }}")
        self._init_ui()

    # ── content ──────────────────────────────────────────────────────────

    def _strategy_config(self):
        """The coincidence strategy config off the first enabled stage.

        The viewer seeds its status-bar controls from these as a single source of truth
        (`_seed_controls_from_strategy`), so reading the first is consistent with how
        the rest of the UI already treats them.
        """
        for stage in self.task_config.enabled_stages:
            config = getattr(getattr(stage, "strategy", None), "config", None)
            if config is not None and hasattr(config, "intensity_drop_fraction"):
                return config
        return None

    def _meta_line(self) -> str:
        stages = self.task_config.enabled_stages
        bits = [self.lamella_name, f"{len(stages)} stage{'s' if len(stages) != 1 else ''}"]
        fov = getattr(self.task_config, "field_of_view", None)
        if fov:
            bits.append(f"{fov * constants.SI_TO_MICRO:.0f} µm field of view")
        return " · ".join(bits)

    def _stage_rows(self) -> List[tuple]:
        """One detail row per stage, keyed by its name.

        Not one multi-line "Stages" value: with the name inline each stage overran the
        value column and wrapped, and a wrapped separator reads as a bullet. Putting the
        name in the label column shortens the line and aligns the stages down the page.
        """
        stages = self.task_config.enabled_stages
        if not stages:
            return [("Stages", "— none enabled")]
        return [
            (
                stage.name,
                f"{_format_current(stage.milling.milling_current)}"
                f" · {_pattern_summary(stage)}"
                f" · {format_duration(stage.estimated_time)}",
            )
            for stage in stages
        ]

    def _rows(self) -> List[tuple]:
        config = self._strategy_config()
        detail: List[tuple] = [("FM channel", self.channel_name or "— none selected")]
        detail.extend(self._stage_rows())

        if config is not None:
            # The trigger is three settings that only mean anything together, so they
            # read as one sentence rather than three rows the user has to assemble.
            detail.append((
                "Drop trigger",
                f"{config.intensity_drop_fraction:.0%} drop"
                f" · {config.rolling_window}-frame window"
                f" · {config.consecutive_triggers} consecutive",
            ))
            detail.append((
                "Monitoring",
                f"starts after {format_duration(config.warmup_duration)} warmup"
                f" · times out at {format_duration(config.timeout)}",
            ))
            detail.append((
                "Mode",
                "supervised — pauses for confirmation"
                if config.supervised
                else "unattended — runs to completion",
            ))
            saved = (
                f"saved (every {config.save_rate_limit} frames)"
                if config.save_fm_images
                else "not saved"
            )
            if config.acquire_fib_image:
                saved += " · FIB image acquired"
            detail.append(("FM images", saved))

        detail.append(("Estimated time", format_duration(self.task_config.estimated_time)))
        return detail

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        enabled = len(self.task_config.enabled_stages)
        total = len(self.task_config.stages)

        meta = QLabel(self._meta_line())
        meta.setStyleSheet(f"color: {TEXT_STRONG}; font-size: 12px;")
        meta.setWordWrap(True)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(_chip(f"{enabled} to mill"))
        if enabled != total:
            chips.addWidget(_chip(f"{total - enabled} disabled"))
        chips.addStretch()

        detail = QFrame()
        detail.setStyleSheet(
            f"QFrame {{ background: {PANEL}; border: 1px solid {BORDER};"
            f" border-radius: 4px; }}"
        )
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(12, 10, 12, 10)
        detail_layout.setSpacing(6)
        for label_text, value_text in self._rows():
            row = QHBoxLayout()
            row.setSpacing(12)
            label = QLabel(label_text)
            label.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; border: none;")
            label.setFixedWidth(104)
            # Stage names go in this column and are user-supplied, so wrap rather than
            # clip — a truncated stage name in a pre-flight check is worse than two lines.
            label.setWordWrap(True)
            value = QLabel(value_text)
            value.setStyleSheet(f"color: {TEXT}; font-size: 11px; border: none;")
            value.setWordWrap(True)
            row.addWidget(label, alignment=Qt.AlignTop)
            row.addWidget(value, stretch=1)
            detail_layout.addLayout(row)

        self.button_start = QPushButton("Start Milling")
        self.button_start.setStyleSheet(stylesheets.PRIMARY_BUTTON_STYLESHEET)
        self.button_start.setMinimumHeight(30)
        self.button_start.clicked.connect(self.accept)

        button_cancel = QPushButton("Cancel")
        button_cancel.setStyleSheet(stylesheets.SECONDARY_BUTTON_STYLESHEET)
        button_cancel.setMinimumHeight(30)
        button_cancel.clicked.connect(self.reject)
        # Milling is irreversible and this dialog opens on the click that would start
        # it, so Enter must not fire it — Cancel takes the default.
        button_cancel.setDefault(True)
        self.button_start.setAutoDefault(False)

        footer = QHBoxLayout()
        footer.addStretch()
        footer.addWidget(button_cancel)
        footer.addWidget(self.button_start)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        layout.addWidget(meta)
        layout.addLayout(chips)
        layout.addWidget(detail)
        layout.addLayout(footer)

        if enabled == 0:
            self.button_start.setEnabled(False)
            self.button_start.setToolTip("No milling stages are enabled.")

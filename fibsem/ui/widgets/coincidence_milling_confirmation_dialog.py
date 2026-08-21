"""Pre-flight summary for a coincidence mill.

Replaces the stock `QMessageBox` in `FluorescenceCoincidenceViewerWidget._run_milling`,
which put the only per-stage detail behind "Show Details" as a `pformat` of the raw
config dict — developer output in the one place an operator checks depth and current
before committing.

Same house style as :class:`FMOverviewConfirmationDialog`: a meta line, count chips, a
detail block, and a primary action in the footer — built from
`fibsem.ui.widgets.preflight` so the three cannot drift.

Milling is irreversible, so the primary action is never the default button: starting
takes a deliberate click.
"""

from typing import List, Optional, Tuple

from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from fibsem import constants
from fibsem.milling.base import FibsemMillingStage
from fibsem.milling.tasks import FibsemMillingTaskConfig
from fibsem.ui import stylesheets
from fibsem.ui.widgets.preflight import (
    BACKGROUND,
    chip,
    detail_block,
    format_duration,
    meta_label,
)

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

    def _current_chip_text(self) -> Optional[str]:
        """`1.0 nA`, or `60 pA – 1.0 nA` when the stages differ.

        A range rather than a single figure: a task that steps down from rough to
        polish is the normal case, and showing only the first stage's current would
        misreport the one number an operator is most likely to be checking for.
        """
        currents = sorted(
            {stage.milling.milling_current for stage in self.task_config.enabled_stages}
        )
        if not currents:
            return None
        if len(currents) == 1:
            return _format_current(currents[0])
        return f"{_format_current(currents[0])} – {_format_current(currents[-1])}"

    def _meta_line(self) -> str:
        stages = self.task_config.enabled_stages
        bits = [
            self.lamella_name,
            f"{len(stages)} stage{'s' if len(stages) != 1 else ''}",
        ]
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
            trigger = (
                f"{config.intensity_drop_fraction:.0%} drop"
                f" · {config.rolling_window}-frame window"
                f" · {config.consecutive_triggers} consecutive"
            )
            if config.supervised:
                # The strategy only acts on _drop_detected when NOT supervised, so in
                # supervised mode these thresholds still drive the readout but stop
                # nothing. Saying so here beats letting the operator infer that the
                # mill will halt itself when it will not.
                trigger += "  → monitoring only; you stop the mill"
            detail.append(("Drop trigger", trigger))
            detail.append(
                (
                    "Monitoring",
                    f"starts after {format_duration(config.warmup_duration)} warmup"
                    f" · times out at {format_duration(config.timeout)}",
                )
            )
            detail.append(
                (
                    "Mode",
                    "supervised — you stop the mill manually"
                    if config.supervised
                    else "automated — stops itself on the drop trigger",
                )
            )
            saved = (
                f"saved (every {config.save_rate_limit} frames)"
                if config.save_fm_images
                else "not saved"
            )
            if config.acquire_fib_image:
                saved += " · FIB image acquired"
            detail.append(("FM images", saved))

        detail.append(
            ("Estimated time", format_duration(self.task_config.estimated_time))
        )
        return detail

    # ── layout ───────────────────────────────────────────────────────────

    def _init_ui(self) -> None:
        enabled = len(self.task_config.enabled_stages)
        total = len(self.task_config.stages)

        meta = meta_label(self._meta_line())

        # Counts first, then the two facts worth reading before anything else: what
        # current is going into the sample, and whether it will stop for you. The FM
        # dialog uses chips only for counts; these are states, but a chip carries no
        # colour, so a state here reads as a fact rather than as a status light.
        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(chip(f"{enabled} to mill"))
        if enabled != total:
            chips.addWidget(chip(f"{total - enabled} disabled"))
        current_text = self._current_chip_text()
        if current_text:
            chips.addWidget(chip(current_text))
        config = self._strategy_config()
        if config is not None:
            # "Automated", not "Unattended" or "Unsupervised": that is what the
            # supervision toggle a few pixels away already says (_update_supervised_button),
            # and what the border state is called. A third word for one mode is worse
            # than an imperfect one.
            chips.addWidget(chip("Supervised" if config.supervised else "Automated"))
        chips.addStretch()

        # Wider labels than the other two, and wrapping: stage names go in this column
        # and are user-supplied, so a truncated one in a pre-flight check is worse than
        # two lines.
        detail = detail_block(self._rows(), label_width=104, wrap_labels=True)

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

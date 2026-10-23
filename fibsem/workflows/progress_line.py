"""Measurable progress as one line, without Qt: milling, an overview's tiles, a spot
burn, a fluorescence acquisition. The main window's status bar draws it, and the App
Server's status answers it (`ProgressLines.follow`).

A line is a label, a step ("stage 2 of 3"), and a fraction with its numbers ("42% ·
1m 20s left") when there is a figure to draw. When an operation ends the line says
how -- done, cancelled, or failed -- as an *outcome*, shown for a moment.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

from fibsem.acting import AGENT
from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.imaging.spot import SpotBurnProgress, SpotBurnStatus
from fibsem.imaging.tiling.progress import (
    MODALITY_FLUORESCENCE,
    TiledProgress,
    TiledStatus,
)
from fibsem.milling.progress import (
    AgentMillOwner,
    MillingMessageTracker,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.util.durations import format_duration_rounded

# How long an outcome ("done", "cancelled", a failure) stays before the line goes.
OUTCOME_S = 2.0

# How long a fluorescence acquisition's line outlives its last report. Only
# `acquire_image` says FINISHED; the autofocus sweep and the device's own z-stack
# stop without a word, and their line would otherwise stay for the session (the
# FIB-374 failure). Every routine reports at least once per plane or step, well
# inside this.
FM_STALE_S = 3.0

FLUORESCENCE = "Fluorescence"

# What the line calls each state of a tiled run, under "FIB/SEM overview" or
# "Fluorescence overview": what kind of thing is happening, read at a glance,
# rather than the producer's own wording.
_TILE_STEPS = {
    TiledStatus.STARTING: "collecting tiles",
    TiledStatus.MOVING: "moving stage",
    TiledStatus.TILE_STARTED: "collecting tiles",
    TiledStatus.TILE_COLLECTED: "collecting tiles",
    TiledStatus.TILES_ACQUIRED: "collecting tiles",
    TiledStatus.STITCHING: "stitching tiles",
    TiledStatus.SAVING: "saving overview",
}


@dataclass(frozen=True)
class ProgressLine:
    """One line of measurable progress, or how it ended (*outcome*).

    *standalone* keeps *label* as the subject under a run: a failed task's own
    name, or the agent's mill, not the task the run is on."""

    label: str
    step: Optional[str] = None
    fraction: Optional[float] = None
    numbers: Optional[str] = None
    outcome: bool = False
    failed: bool = False
    standalone: bool = False


def _fraction_left(
    remaining: Optional[float], total: Optional[float]
) -> Tuple[Optional[float], Optional[str]]:
    """The fraction done and "42% · 1m 20s left" from a countdown; Nones without one."""
    if remaining is None or not total:
        return None, None
    fraction = min(max(1.0 - remaining / total, 0.0), 1.0)
    return (
        fraction,
        f"{int(fraction * 100)}% · {format_duration_rounded(remaining)} left",
    )


class ProgressReader:
    """Reads each source's reports into lines, keeping what a later report leaves
    out: a delegating strategy's label, a milling stage, a tiled run's count."""

    def __init__(self) -> None:
        # A delegating strategy names itself once and the backend's ticks carry no
        # words; this keeps the strategy's (FIB-797).
        self._milling_label = MillingMessageTracker()
        # The milling run the reports belong to, and its name when it is the
        # agent's own mill outside any task: the line is the agent's, not the run's
        # current task, to its end (FIB-1401).
        self._mill_owner = AgentMillOwner()
        # The mill's step ("stage 2 of 3") on its own, for a tick to keep: the
        # line's step may carry the agent's mill name before it (FIB-1413).
        self._mill_step: Optional[str] = None
        # A tiled run's last count, kept while a state without one (a stage move, the
        # stitch) changes the words: the bar must not snap back to nothing.
        self._tile_count: Optional[Tuple[float, str]] = None
        # Whether a tiled run is going, whose line wins over the per-tile reports of
        # the channels and planes inside it.
        self.tiling = False

    def milling(
        self, payload: object, agent_in_control: bool, showing: bool
    ) -> ProgressLine:
        """One milling report: the label the producer gave, which stage of how many,
        and the countdown once there is one. *showing* says a progress line is up,
        whose stage a tick keeps.

        Total over whatever arrives: the status bar calls it in a queued slot, where
        an exception is a process abort on PyQt5 (FIB-329), and a plugin strategy may
        still emit the old dict (`from_payload`)."""
        report = MillingProgress.from_payload(payload)
        agent_mill = self._mill_owner.agent_mill(report, agent_in_control)
        what, standalone = ("Agent", True) if agent_mill else ("Milling", False)
        ended = f"{agent_mill} " if agent_mill else ""
        if report.status is MillingProgressStatus.TASK_FAILED:
            reason = f": {report.error}" if report.error else ""
            return ProgressLine(
                what, f"{ended}failed{reason}", outcome=True, failed=True,
                standalone=standalone,
            )  # fmt: skip
        if report.status is MillingProgressStatus.TASK_CANCELLED:
            return ProgressLine(
                what, f"{ended}cancelled", outcome=True, standalone=standalone
            )
        if report.status.is_terminal:
            return ProgressLine(
                what, f"{ended}done", outcome=True, standalone=standalone
            )
        label = self._milling_label.label(report)
        step = None
        stage, total = report.display_stage, report.total_stages
        if stage and total and total > 1:
            step = f"stage {stage} of {total}"
        elif showing and report.status is not MillingProgressStatus.STAGE_STARTED:
            step = self._mill_step  # a tick does not repeat the stage count
        self._mill_step = step
        fraction = numbers = None
        if report.status is MillingProgressStatus.STAGE_UPDATE:
            fraction, numbers = _fraction_left(
                report.remaining_time, report.estimated_time
            )
        if agent_mill:
            step = " · ".join(p for p in (agent_mill, step) if p)
            return ProgressLine("Agent", step, fraction, numbers, standalone=True)
        return ProgressLine(label, step, fraction, numbers)

    def tiled(self, event: TiledProgress) -> ProgressLine:
        """One report from a tiled acquisition, either modality."""
        what = (
            "Fluorescence overview"
            if event.modality == MODALITY_FLUORESCENCE
            else "FIB/SEM overview"
        )
        self.tiling = not event.status.is_terminal
        if event.status.is_terminal:
            self._tile_count = None
            if event.status is TiledStatus.FAILED:
                reason = f": {event.error}" if event.error else ""
                return ProgressLine(what, f"failed{reason}", outcome=True, failed=True)
            if event.status is TiledStatus.CANCELLED:
                return ProgressLine(what, "cancelled", outcome=True)  # not a failure
            return ProgressLine(what, "done", outcome=True)
        words = _TILE_STEPS.get(event.status, "collecting tiles")
        if event.completed is not None and event.total:
            done = min(event.completed, event.total)
            fraction = done / event.total
            self._tile_count = (fraction, f"{done} of {event.total}")
            numbers = f"{int(fraction * 100)}%"
            if event.estimated_remaining_seconds > 0:
                left = format_duration_rounded(event.estimated_remaining_seconds)
                numbers += f" · {left} left"
        elif self._tile_count is not None:
            # The last count stands, but not its time left: it would sit frozen
            # through the stitch.
            numbers = f"{int(self._tile_count[0] * 100)}%"
        else:
            return ProgressLine(what, words)
        fraction, count = self._tile_count
        return ProgressLine(what, f"{words} · {count}", fraction, numbers)

    def spot_burn(self, report: SpotBurnProgress) -> ProgressLine:
        """One report from a spot burn, supervised or not."""
        what = "Spot burn"
        if report.status.is_terminal:
            if report.status is SpotBurnStatus.FAILED:
                reason = f": {report.error}" if report.error else ""
                return ProgressLine(what, f"failed{reason}", outcome=True, failed=True)
            if report.status is SpotBurnStatus.CANCELLED:
                return ProgressLine(what, "cancelled", outcome=True)
            return ProgressLine(what, "done", outcome=True)
        step = None
        if report.current_point and report.total_points:
            step = f"spot {report.current_point} of {report.total_points}"
        fraction, numbers = _fraction_left(
            report.total_remaining_time, report.total_estimated_time
        )
        return ProgressLine(what, step, fraction, numbers)

    def fluorescence(
        self, report: FluorescenceAcquisitionProgress
    ) -> Optional[ProgressLine]:
        """One report from a fluorescence acquisition: the channel, and the plane,
        the autofocus step or the channel count it is on. None while a tiled run is
        going: the tiles are its line. FINISHED is an outcome, which only a
        fluorescence line already up should show."""
        if self.tiling:
            return None
        status = report.status
        if status is FluorescenceAcquisitionStatus.FINISHED:
            return ProgressLine(FLUORESCENCE, "done", outcome=True)
        channel = report.channel or None
        count = fraction = None
        if status is FluorescenceAcquisitionStatus.ACQUIRING_AUTOFOCUS:
            parts = ["autofocus", channel]
            if report.zlevel and report.total_zlevels:
                count = f"step {report.zlevel} of {report.total_zlevels}"
                fraction = report.zlevel / report.total_zlevels
            if report.total_passes and report.total_passes > 1 and report.pass_index:
                parts.append(f"pass {report.pass_index} of {report.total_passes}")
        elif status is FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK:
            parts = [channel]
            if report.zlevel and report.total_zlevels:
                count = f"plane {report.zlevel} of {report.total_zlevels}"
                fraction = report.zlevel / report.total_zlevels
            if report.total_channels and report.total_channels > 1:
                parts.append(
                    f"channel {report.channel_index} of {report.total_channels}"
                )
        else:
            parts = [channel]
            if report.channel_index and report.total_channels:
                count = f"channel {report.channel_index} of {report.total_channels}"
                fraction = report.channel_index / report.total_channels
        step = " · ".join(p for p in (*parts, count) if p) or None
        numbers = None
        if fraction is not None:
            fraction = min(max(fraction, 0.0), 1.0)
            numbers = f"{int(fraction * 100)}%"
        return ProgressLine(FLUORESCENCE, step, fraction, numbers)


def progress_sources(microscope: Any) -> List[Tuple[Any, str]]:
    """*microscope*'s progress signals, each with the `ProgressReader` source it
    feeds. On the services: no imaging without a beam, no milling or spot burn
    without an ion beam; fluorescence on its optional device."""
    sources = []
    imaging = getattr(microscope, "imaging", None)
    if imaging is not None:
        sources.append((imaging.tiled_progress, "tiled"))
    milling = getattr(microscope, "milling", None)
    if milling is not None:
        sources.append((milling.progress.changed, "milling"))
    spot_burn = getattr(microscope, "spot_burn", None)
    if spot_burn is not None:
        sources.append((spot_burn.progress.changed, "spot_burn"))
    fm = getattr(microscope, "fm", None)
    if fm is not None:
        sources.append((fm.acquisition_progress_signal, "fluorescence"))
    return sources


class ProgressLines:
    """The last progress line a microscope said, for a reader that asks rather
    than listens: `current` is the line now, None when nothing measurable runs.
    An outcome lasts `OUTCOME_S`; a fluorescence line `FM_STALE_S` after its last
    report."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._reader = ProgressReader()
        self._line: Optional[ProgressLine] = None
        self._at = 0.0
        self._agent_in_control: Callable[[], bool] = lambda: False

    @property
    def current(self) -> Optional[ProgressLine]:
        line = self._line
        if line is None:
            return None
        age = self._clock() - self._at
        if (line.outcome and age > OUTCOME_S) or (
            line.label == FLUORESCENCE and age > FM_STALE_S
        ):
            return None
        return line

    def take(self, source: str, report: Any) -> None:
        """One report from *source* (`progress_sources`)."""
        reader = self._reader
        if source == "milling":
            line = reader.milling(
                report, self._agent_in_control(), self.current is not None
            )
        elif source == "tiled":
            line = reader.tiled(report)
        elif source == "spot_burn":
            line = reader.spot_burn(report)
        else:
            line = reader.fluorescence(report)
            if line is None:
                return
            if line.outcome:
                showing = self.current
                if showing is None or showing.label != FLUORESCENCE:
                    return
        self._line, self._at = line, self._clock()

    def follow(self, microscope: Any) -> Callable[[], None]:
        """Take *microscope*'s progress reports; returns the call that stops."""
        self._line = None
        control = getattr(microscope, "control", None)

        def agent_in_control() -> bool:
            current = getattr(control, "current", None)
            return current is not None and current.by == AGENT

        self._agent_in_control = agent_in_control
        slots = []
        for signal, source in progress_sources(microscope):

            def slot(report: Any, source: str = source) -> None:
                try:
                    self.take(source, report)
                except Exception:  # noqa: BLE001 - a status read must never cost the emitter
                    pass

            signal.connect(slot)
            slots.append((signal, slot))

        def stop() -> None:
            for signal, slot in slots:
                try:
                    signal.disconnect(slot)
                except Exception:
                    pass  # already gone with its microscope

        return stop

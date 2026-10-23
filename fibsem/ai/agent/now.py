"""What the agent is doing on the microscope now, in a line, without Qt: the
agent card's Now line and the web app's.

The action it holds control for ("focus sweep"), or while it mills, which mill,
how far and how long is left ("Milling 'trench' · stage 2 of 6 · about 15s
left"). A mill the agent starts lets go of control at once; it is still the
agent's to its end (`AgentMillOwner`).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from fibsem.acting import AGENT
from fibsem.milling.progress import (
    AgentMillOwner,
    MillingProgress,
    MillingProgressStatus,
)
from fibsem.util.durations import format_duration_rounded


class AgentNow:
    """Follows the agent's action and the mills it runs; `text` is the line."""

    def __init__(self) -> None:
        self.action: Optional[str] = None
        self._owner = AgentMillOwner()
        # name, stage, of, seconds left, and whether a stage is starting
        self._mill: Optional[
            Tuple[str, Optional[int], Optional[int], Optional[float], bool]
        ] = None

    def clear(self) -> None:
        self.action = None
        self._mill = None

    def take_control(self, in_control: Any) -> None:
        """Who is in control now (``InControl`` or None)."""
        if in_control is not None and in_control.by == AGENT:
            self.action = in_control.action
        else:
            self.action = None

    def take_progress(self, payload: object) -> None:
        """A milling report: the agent's own mill, or any mill while the agent
        holds the microscope (its ``run_task``), is on the line."""
        report = MillingProgress.from_payload(payload)
        own = self._owner.agent_mill(report, self.action is not None)
        if report.status.is_terminal or not (own or self.action is not None):
            self._mill = None
            return
        last = self._mill
        name = report.milling_task_name or own or (last[0] if last else "milling")
        stage, total = report.display_stage, report.total_stages
        if stage is None and last is not None:
            stage, total = last[1], last[2]  # a tick does not repeat the stage
        # A stage starting has no countdown yet: the last stage's is over.
        preparing = report.status in (
            MillingProgressStatus.TASK_STARTED,
            MillingProgressStatus.STAGE_STARTED,
        )
        remaining = report.remaining_time
        if remaining is None and last is not None and not preparing:
            remaining = last[3]
        self._mill = (name, stage, total, remaining, preparing)

    @property
    def text(self) -> Optional[str]:
        """The line, or None when the agent does nothing."""
        if self._mill is None:
            return self.action
        name, stage, total, remaining, preparing = self._mill
        parts = [f"Milling '{name}'"]
        if stage and total and total > 1:
            parts.append(f"stage {stage} of {total}")
        if preparing and remaining is None:
            parts.append("preparing")
        elif remaining is not None:
            parts.append(f"about {format_duration_rounded(remaining)} left")
        return " · ".join(parts)


def follow(
    now: AgentNow, microscope: Any, changed: Callable[[], None] = lambda: None
) -> Callable[[], None]:
    """Feed *now* from *microscope*'s control and milling progress, calling
    *changed* after each; returns the call that stops it."""
    now.clear()
    control = getattr(microscope, "control", None)
    progress = getattr(getattr(microscope, "milling", None), "progress", None)

    def on_control(in_control: Any) -> None:
        now.take_control(in_control)
        changed()

    def on_progress(payload: object) -> None:
        now.take_progress(payload)
        changed()

    signals = []
    if control is not None:
        control.changed.connect(on_control)
        signals.append((control.changed, on_control))
        now.take_control(control.current)
    if hasattr(progress, "changed"):
        progress.changed.connect(on_progress)
        signals.append((progress.changed, on_progress))

    def stop() -> None:
        for signal, slot in signals:
            try:
                signal.disconnect(slot)
            except Exception:
                pass  # already gone with its microscope

    return stop

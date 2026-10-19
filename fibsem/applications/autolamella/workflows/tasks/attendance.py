"""What a task needs from a person, said before the run.

A task declares what it asks while it runs (``asks``: the kinds it needs
answered before its next line, a mill's passes among them); the protocol says
whether a person decides (``attention``). From those, one line per task: whether the operator
has to be there while it runs. Nothing waits on a decision once a task ends.
Derived, never stored, so it cannot drift from what the task does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, Type

from fibsem.workflows.attention import Attention
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    DETECTION,
    MILLING,
    POINT_OF_INTEREST,
    SPOT_BURN,
    STATE,
)
from fibsem.workflows.proposals import kind_label as proposal_kind_label

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskConfig,
        AutoLamellaTaskProtocol,
    )
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
    from fibsem.workflows.grid.protocol import GridTaskProtocol

# How a kind is said to the operator. A kind asked mid-run is what the person
# is needed for; a kind left for afterwards is what waits on them.
KIND_LABELS = {
    DETECTION: "detection",
    STATE: "the position",
    ALIGNMENT_AREA: "the alignment area",
    POINT_OF_INTEREST: "the point of interest",
    MILLING: "milling",
    SPOT_BURN: "the spot burn",
}


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, proposal_kind_label(kind).lower())


def _named(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} and {names[-1]}"


@dataclass(frozen=True)
class Attendance:
    """What one task needs from a person, under one attention."""

    task_name: str
    attention: Attention
    # what needs a person at the microscope while the task runs: the kinds
    # it asks, in order
    present: Tuple[str, ...]

    @property
    def needs_a_person_present(self) -> bool:
        return self.attention is Attention.supervised and bool(self.present)

    @property
    def line(self) -> str:
        """One sentence or two, for under the chip and the pre-run summary."""
        if self.attention is Attention.automated:
            return "Nobody is asked."
        if self.present:
            return f"Needs you at the microscope while it runs: {_named(self.present)}."
        return "Runs on its own."


def attendance(
    task_cls: Type["AutoLamellaTask"],
    config: Optional["AutoLamellaTaskConfig"],
    attention: Attention,
    *,
    task_name: str = "",
) -> Attendance:
    """What ``task_cls`` needs from a person under ``config`` and ``attention``."""
    asks = task_cls.asks_for(config) if config is not None else task_cls.asks
    return Attendance(
        task_name=task_name or getattr(task_cls, "__name__", ""),
        attention=attention,
        present=tuple(kind_label(k) for k in asks),
    )


def attendance_for(
    protocol: "AutoLamellaTaskProtocol", task_name: str
) -> Optional[Attendance]:
    """The attendance of the task ``task_name`` names in ``protocol``, or None
    when the protocol has no config for it or its type is not registered."""
    from fibsem.applications.autolamella.workflows.tasks import get_tasks

    config = protocol.task_config.get(task_name)
    if config is None:
        return None
    task_cls = get_tasks().get(config.task_type)
    if task_cls is None:
        return None
    return attendance(
        task_cls,
        config,
        protocol.workflow_config.get_attention(task_name),
        task_name=task_name,
    )


@dataclass(frozen=True)
class RunAttendance:
    """Who a run needs, said before it: the tasks that wait for a person, a
    line per task, and the sentence for the run. Beside the run's estimate,
    not inside it: a duration does not depend on who is in the room."""

    # the tasks that pause for a person, in run order
    supervised: Tuple[str, ...] = ()
    # what each task needs from a person, said in a line; a task whose type
    # this build does not know has none
    lines: Mapping[str, str] = field(default_factory=dict)
    # the pre-run sentence: which tasks need a person present
    summary: str = ""

    @property
    def waits_for_a_human(self) -> bool:
        return bool(self.supervised)


def run_attendance(
    protocol: "AutoLamellaTaskProtocol", task_names: Sequence[str]
) -> RunAttendance:
    """What a run of ``task_names`` needs from a person."""
    workflow = protocol.workflow_config
    lines = {}
    present = []
    for name in task_names:
        att = attendance_for(protocol, name)
        if att is None:
            continue
        lines[name] = att.line
        if att.needs_a_person_present:
            present.append(name)
    if not present:
        summary = "This run needs nobody present."
    elif len(present) == len(task_names):
        summary = f"This run needs you present for every task: {_named(present)}."
    else:
        summary = (
            f"This run needs you present for {_named(present)}. "
            "Everything else runs on its own."
        )
    return RunAttendance(
        supervised=tuple(
            name
            for name in task_names
            if workflow.get_attention(name) is Attention.supervised
        ),
        lines=lines,
        summary=summary,
    )


def grid_run_attendance(
    protocol: "GridTaskProtocol", task_names: Sequence[str]
) -> RunAttendance:
    """What a grid run of ``task_names`` needs from a person: the tasks that
    wait for one. A grid task says nothing more about it."""
    supervised = []
    for name in task_names:
        config = protocol.task_config.get(name)
        if config is not None and config.attention is Attention.supervised:
            supervised.append(name)
    return RunAttendance(supervised=tuple(supervised))

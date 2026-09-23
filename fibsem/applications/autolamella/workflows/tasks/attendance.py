"""What a task needs from a person, said before the run.

A task declares what it asks while it runs (``questions``: the kinds it needs
answered before its next line), what it does at the microscope with the
tools (``sessions``), and what it leaves for afterwards (``proposer``). The
protocol says whether a person decides (``attention``). From those, one line
per task: whether the operator has to be there while it runs, and what waits
for their decision afterwards. Derived, never stored, so it cannot drift
from what the task does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Type

from fibsem.applications.autolamella.proposals import (
    ALIGNMENT_AREA,
    DETECTION,
    OVERVIEW_POSITIONS,
    POINT_OF_INTEREST,
    STATE,
    TASK_RESULT,
)
from fibsem.applications.autolamella.proposals import kind_label as proposal_kind_label
from fibsem.applications.autolamella.structures import Attention

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import (
        AutoLamellaTaskConfig,
        AutoLamellaTaskProtocol,
    )
    from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask

# How a kind is said to the operator. A kind asked mid-run is what the person
# is needed for; a kind left for afterwards is what waits on them.
KIND_LABELS = {
    DETECTION: "detection",
    STATE: "the position",
    ALIGNMENT_AREA: "the alignment area",
    POINT_OF_INTEREST: "the point of interest",
    TASK_RESULT: "its result",
    OVERVIEW_POSITIONS: "the lamella positions",
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
    # what needs a person at the microscope while the task runs, in order:
    # questions it asks, then sessions it runs
    present: Tuple[str, ...]
    # what it leaves for afterwards, as its label, or None
    later: Optional[str]
    # the tasks that require this one: what waits for the decision on ``later``
    waiters: Tuple[str, ...]
    # whether the Review workflow is on: with it off nothing waits afterwards
    review_on: bool

    @property
    def needs_a_person_present(self) -> bool:
        return self.attention is Attention.supervised and bool(self.present)

    @property
    def line(self) -> str:
        """One sentence or two, for under the chip and the pre-run summary."""
        if self.attention is Attention.automated:
            text = "Nobody is asked."
            if self.later and self.waiters and self.review_on:
                text += (
                    f" {_cap(self.later)} is open to correct until "
                    f"{_named(self.waiters)} starts."
                )
            return text
        text = (
            f"Needs you at the microscope while it runs: {_named(self.present)}."
            if self.present
            else "Runs on its own."
        )
        if self.later and self.review_on:
            if self.waiters:
                text += (
                    f" {_named(self.waiters)} waits for your decision on {self.later}."
                )
            else:
                text += f" Nothing waits on {self.later}; it is listed to check."
        return text


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def attendance(
    task_cls: Type["AutoLamellaTask"],
    config: Optional["AutoLamellaTaskConfig"],
    attention: Attention,
    *,
    task_name: str = "",
    waiters: Sequence[str] = (),
    review_on: bool = True,
) -> Attendance:
    """What ``task_cls`` needs from a person under ``config`` and ``attention``."""
    questions = (
        task_cls.questions_for(config) if config is not None else task_cls.questions
    )
    sessions = (
        task_cls.sessions_for(config) if config is not None else task_cls.sessions
    )
    present = tuple(kind_label(k) for k in questions) + tuple(sessions)
    proposer = task_cls.proposer
    later = kind_label(proposer.kind) if proposer is not None else None
    return Attendance(
        task_name=task_name or getattr(task_cls, "__name__", ""),
        attention=attention,
        present=present,
        later=later,
        waiters=tuple(waiters),
        review_on=review_on,
    )


def attendance_for(
    protocol: "AutoLamellaTaskProtocol", task_name: str, review_on: bool = True
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
    workflow = protocol.workflow_config
    waiters = [t.name for t in workflow.tasks if task_name in t.requires]
    return attendance(
        task_cls,
        config,
        workflow.get_attention(task_name),
        task_name=task_name,
        waiters=waiters,
        review_on=review_on,
    )


def run_attendance(
    protocol: "AutoLamellaTaskProtocol", task_names: List[str], review_on: bool = True
) -> str:
    """The pre-run sentence: which of ``task_names`` need a person present."""
    present = []
    for name in task_names:
        att = attendance_for(protocol, name, review_on)
        if att is not None and att.needs_a_person_present:
            present.append(name)
    if not present:
        return "This run needs nobody present."
    if len(present) == len(task_names):
        return f"This run needs you present for every task: {_named(present)}."
    return (
        f"This run needs you present for {_named(present)}. "
        "Everything else runs on its own."
    )

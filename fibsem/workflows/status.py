"""The status a task runs through, in any application's workflow."""

from enum import Enum, auto


class TaskStatus(Enum):
    NotStarted = auto()
    InProgress = auto()
    # The microscope work is done and the task's record waits on a decision in
    # the Review tab. Not finished: nothing that requires this task runs until
    # a decision moves it to Completed (confirmed) or Failed (rejected).
    AwaitingDecision = auto()
    Completed = auto()
    Failed = auto()
    Skipped = auto()
    Cancelled = auto()  # aborted by the user (Stop), distinct from a genuine Failure
    Removed = auto()  # pulled from the queue by the user before it ran

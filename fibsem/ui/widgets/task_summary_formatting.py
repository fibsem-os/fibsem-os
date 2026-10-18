"""
Shared status labels and colours for the task summary and the lamella and grid lists,
so they stay visually consistent.
"""

from fibsem.ui.tokens import (
    ACCENT_COLOR,
    ERROR_COLOR,
    OK_COLOR,
    ORANGE_COLOR,
    TEXT_MUTED_COLOR,
    WARN_COLOR,
)

# How a status name reads to a person; anything not listed reads as its name.
STATUS_LABELS = {
    "InProgress": "in progress",
    "AwaitingDecision": "awaiting decision",
}


def status_label(status: str) -> str:
    return STATUS_LABELS.get(status, status.lower())


# Semantic badge colours (matching the fibsem.ui.stylesheets palette) used by
# the workflow summary dialog: status -> (dot/strong colour, muted text colour).
STATUS_BADGE_COLORS = {
    "Completed": (OK_COLOR, "#7fd083"),
    "Failed": (ERROR_COLOR, "#e08585"),
    "Skipped": (TEXT_MUTED_COLOR, "#aeb4b9"),
    "InProgress": (ACCENT_COLOR, "#9cc7f5"),
    "Cancelled": (WARN_COLOR, "#e8c37f"),  # amber: user-aborted, not an error
    # orange: the border's colour for "waiting on you" -- the run is over, the
    # task is not finished until someone decides in the Review tab
    "AwaitingDecision": (ORANGE_COLOR, "#f2b56b"),
}

# Status order for count chips (these three are always shown, even at zero)
STATUS_CHIP_ORDER = ["Completed", "Failed", "Skipped"]

"""The watcher's lines for an instrument kind's own events.

The watcher (`fibsem.ai.agent.watch`) says what any instrument's agent hears:
proposals, requests, messages, the level, workflows and tasks. What only one
kind of instrument does, it leaves to that kind's lines here: the FIB-SEM's
milling (`fibsem`). A kind with none (a cryo-FM, or a playbook with no
instrument) hears none of them.

Each kind's ``LINES`` maps an event kind to a function from the event to its
wake-up, ``(kind, text)``, or None for no line.
"""

from typing import Optional

from fibsem.ai.agent import instruments
from fibsem.ai.agent.wake_lines import fibsem
from fibsem.ai.agent.wake_lines.base import Line, Lines, item_of

__all__ = ["Line", "Lines", "for_kind", "item_of"]

# Each kind's own lines, by its name; a kind not here has none.
KINDS = {"fibsem": fibsem.LINES}


def for_kind(kind: Optional[str]) -> Lines:
    """The lines instrument *kind* adds to the watcher's (the default kind's
    for None); none for a kind without its own."""
    return dict(KINDS.get(kind or instruments.DEFAULT, {}))

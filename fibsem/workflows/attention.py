"""Who decides a task's answers: a person (supervised) or nobody (automated).

A property of the producing task, set by the protocol author; any app's tasks
carry one.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any


class Attention(str, Enum):
    """Whether a person (or an agent) decides this task's answers. A property
    of the producing task, set by the protocol author.

    supervised -- a person decides. A question the task needs answered before
                  it can carry on is asked in the workflow, where its widget
                  is, and holds the run until it is answered. Nothing waits
                  on a decision after the task ends.
    automated  -- nobody is asked; the record is made and listed to check.

    Where the run waits is not a third setting: a "Review later" mode existed
    on development builds before 0.6.0 and is read as supervised.
    """

    supervised = "supervised"
    automated = "automated"


def attention_from(value: Any, where: str = "") -> Attention:
    """A stored value as an ``Attention``, for the loaders.

    Takes what a record can hold: an attention's own value, or the boolean
    supervise flag that stood in its place before it -- v0.5.2's task
    descriptions, and the per-stage supervision of the protocol before them.
    A value this build does not know reads as automated with a warning
    naming ``where``, because one unreadable field must not drop the whole
    task. For loaders: ``Attention(x)`` still raises on a bad literal in
    code, which is where it should.
    """
    if isinstance(value, bool):
        return Attention.supervised if value else Attention.automated
    if value in ("review", "review_later"):
        # A third mode, the operator deciding afterwards, existed on
        # development builds before 0.6.0 and never shipped. It is read as
        # Supervised: the task's own questions say where the run waits.
        return Attention.supervised
    try:
        return Attention(value)
    except ValueError:
        on = f" on {where}" if where else ""
        logging.warning(f"Unknown attention {value!r}{on}; read as automated.")
        return Attention.automated

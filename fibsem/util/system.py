"""Host-machine facts: how much room is left where the application is about to write.

Named for the host, not the instrument. ``fibsem.structures.SystemInfo`` is the
*microscope's* identity -- manufacturer, model, serial -- and a second meaning of
"system" in this codebase would be misread every time, so nothing here borrows that
name. This module answers only for the machine the application runs on.

Stdlib only, deliberately. ``shutil.disk_usage`` reaches a Windows mapped drive
(``D:``, ``Z:``) and a UNC path through ``GetDiskFreeSpaceExW``, and everything else
through ``statvfs``: a network share is not a special case here, and psutil would buy
nothing for disk.

Two things it does *not* do, both on purpose:

- **Quotas.** CPython's ``nt._getdiskusage`` returns the volume's total free bytes and
  discards ``GetDiskFreeSpaceExW``'s ``lpFreeBytesAvailable``, which is the figure a
  per-user quota caps. On a quota'd share this module therefore reports more room than
  the account can use. Nobody has reported a quota'd share yet; the fix if one turns
  up is a ``ctypes`` call here, not a change at the call sites.
- **Promise not to block.** A disconnected mapped drive does not fail fast -- an SMB
  reconnect can hang for tens of seconds. Call :func:`disk_space` off the GUI thread.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

PathLike = Union[str, os.PathLike]

# What counts as room to work in. Absolute figures rather than a fraction of the
# volume: 5% of a 4 TB share is 200 GB and perfectly fine, 5% of a 256 GB disk is 12 GB
# and is not, and the number anyone acts on is in GB either way.
LOW_FREE_SPACE_BYTES = 20_000_000_000  # 20 GB
CRITICAL_FREE_SPACE_BYTES = 10_000_000_000  # 10 GB


class FreeSpaceLevel(Enum):
    """How much room is left, as a judgement rather than a number.

    Advisory in all three cases. A level is something the UI colours; it is never
    something that stops a run. A user who knows they are writing 200 MB to a disk with
    4 GB left is right, and a dialog that refuses them is wrong.
    """

    AMPLE = "ample"
    LOW = "low"
    CRITICAL = "critical"


def classify_free_space(free: int) -> FreeSpaceLevel:
    """Which band ``free`` bytes falls in. See the thresholds above."""
    if free < CRITICAL_FREE_SPACE_BYTES:
        return FreeSpaceLevel.CRITICAL
    if free < LOW_FREE_SPACE_BYTES:
        return FreeSpaceLevel.LOW
    return FreeSpaceLevel.AMPLE


@dataclass(frozen=True)
class DiskSpace:
    """Bytes on the volume a path lands on.

    ``path`` is what was actually measured, which for a not-yet-created experiment is
    an existing ancestor of the path asked about -- kept so a caller can say which
    volume it is talking about, and so a surprising answer can be traced to the
    directory that produced it.
    """

    path: str
    total: int
    used: int
    free: int

    @property
    def fraction_used(self) -> float:
        """Used over total, 0.0 for a volume reporting no size at all."""
        return self.used / self.total if self.total else 0.0

    @property
    def level(self) -> FreeSpaceLevel:
        return classify_free_space(self.free)


def nearest_existing(path: PathLike) -> Optional[str]:
    """The closest ancestor of ``path`` that exists, or None if none does.

    A new experiment's directory has not been made yet, and ``disk_usage`` on a path
    that is not there raises rather than answering for the volume it would live on --
    which is the question. ``None`` covers a drive letter that is not mapped at all:
    every ancestor of ``Z:\\data\\runs`` is missing too, and the walk stops at the root
    rather than looping.
    """
    current = os.path.abspath(str(path))
    while not os.path.exists(current):
        parent = os.path.dirname(current)
        if parent == current:  # reached the root with nothing found
            return None
        current = parent
    return current


def disk_space(path: PathLike) -> Optional[DiskSpace]:
    """Total/used/free bytes for the volume ``path`` lands on, or None.

    None rather than an exception. A disconnected mapped drive is an ordinary state
    for this to be called in, every caller so far is a label, and a label has the
    option of saying nothing -- which is better than the alternatives, since there is
    nothing a caller could do with the OSError that this has not already done.

    Can block for seconds on an unreachable network share; see the module docstring.
    """
    target = nearest_existing(path)
    if target is None:
        logging.debug("no existing ancestor of %s to measure", path)
        return None
    try:
        usage = shutil.disk_usage(target)
    except OSError as exc:
        logging.debug("disk_usage(%s) failed: %s", target, exc)
        return None
    return DiskSpace(path=target, total=usage.total, used=usage.used, free=usage.free)


def directory_size(path: PathLike) -> int:
    """Bytes on disk under ``path``, unreadable entries skipped.

    ``os.scandir`` rather than ``os.walk`` plus ``getsize``, which stats every file a
    second time; an experiment is thousands of files and is routinely on a share,
    where each stat is a round trip. Explicitly iterative: a deep tree is not this
    machine's problem, but a recursion limit would be.

    Skipping is silent per entry and per directory. The number is for a label, and one
    unreadable file is not a reason to have nothing to show.
    """
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        # follow_symlinks=False throughout: a link into the tree would
                        # otherwise be counted twice, and one pointing out of it would
                        # charge the experiment for something that is not its own.
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        else:
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total

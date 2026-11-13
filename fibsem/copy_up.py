"""Copy an experiment's folder up to a store, one way.

The folder on the instrument's PC is the one that is worked on; the copy is
read by others and written only here. A copy sends what is new or changed
since the last one, never removes anything from the store, and can be run
again at any time: a copy that was cut short carries on where it stopped.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from fibsem.storage import Store, valid_name

__all__ = ["Copied", "copy_up"]

log = logging.getLogger(__name__)

# A source's modified time and its copy's may differ by this much and still
# be the same file: some shared drives keep times to two seconds.
SAME_TIME_S = 2.0


@dataclass
class Copied:
    """What one copy did."""

    sent: int = 0
    same: int = 0
    bytes: int = 0
    failed: List[str] = field(default_factory=list)
    stopped: bool = False

    @property
    def complete(self) -> bool:
        return not self.failed and not self.stopped


def copy_up(
    folder: Path,
    store: Store,
    under: str,
    stop: Optional[Callable[[], bool]] = None,
) -> Copied:
    """Send the files of *folder* that the store lacks, or holds an older or
    different-sized copy of, to *under* in the store. *stop* is asked before
    each file, and a true answer ends the copy there. A file that can't be
    read or sent is named in ``failed`` and the rest still goes."""
    folder = Path(folder)
    done = Copied()
    for here, folders, files in os.walk(folder):
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for file in sorted(files):
            if stop is not None and stop():
                done.stopped = True
                return done
            source = Path(here) / file
            name = f"{under}/{source.relative_to(folder).as_posix()}"
            if file.endswith(".part") or not valid_name(name):
                continue
            try:
                found = source.stat()
                kept = store.stat(name)
                if (
                    kept is not None
                    and kept.size == found.st_size
                    and abs(kept.mtime - found.st_mtime) <= SAME_TIME_S
                ):
                    done.same += 1
                    continue
                store.save(name, source)
            except OSError as error:
                log.warning("Not copied: %s (%s)", source, error)
                done.failed.append(name)
                continue
            done.sent += 1
            done.bytes += found.st_size
    return done

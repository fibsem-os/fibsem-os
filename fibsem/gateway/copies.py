"""The copy of this PC's experiments to a shared folder, kept by the gateway.

Off until the operator names the folder. Then, once a night, each experiment
on this PC is copied up one way (`fibsem.copy_up`): what is new or changed
goes, nothing is removed from the shared folder, and the folder on this PC
stays the one that is worked on. The nightly copy waits while a workflow
runs; "Copy now" is a person's and goes at once. A shared folder that is not
there costs a copy, never a run.
"""

from __future__ import annotations

import json
import logging
import os
import re
import socket
import threading
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fibsem.config import recent_experiment_paths
from fibsem.copy_up import copy_up
from fibsem.gateway.gateway import GatewayRefused, experiments_folder
from fibsem.index import EXPERIMENT_FILE, files_named
from fibsem.server.discovery import DISCOVERY_DIR
from fibsem.storage import FolderStore

__all__ = ["COPIES_FILE", "Copies", "experiment_folders"]

log = logging.getLogger(__name__)

COPIES_FILE = DISCOVERY_DIR / "gateway-copies.json"
# When the nightly copy goes, by this PC's clock.
NIGHTLY = time(2, 0)


def experiment_folders() -> List[Path]:
    """Each experiment's folder on this PC: those under the experiments
    folder, and the recent ones kept elsewhere."""
    found = {
        path.parent for path in files_named(Path(experiments_folder()), EXPERIMENT_FILE)
    }
    found |= {
        Path(path).parent for path in recent_experiment_paths() if os.path.isfile(path)
    }
    return sorted(found)


def _label() -> str:
    """This PC's folder in the shared folder: its host name."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", socket.gethostname()).strip("-.") or "pc"


class Copies:
    """The shared folder, the last copy and the next one."""

    def __init__(
        self,
        running: Callable[[], bool] = lambda: False,
        path: Optional[Path] = None,
        folders: Callable[[], List[Path]] = experiment_folders,
        now: Callable[[], datetime] = datetime.now,
        label: Callable[[], str] = _label,
    ) -> None:
        self._running = running
        self._path = Path(path) if path is not None else COPIES_FILE
        self._folders = folders
        self._now = now
        self._label = label
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        kept = self._read()
        self._folder: Optional[str] = kept.get("folder") or None
        self._last: Optional[Dict[str, Any]] = kept.get("last") or None
        # No copy for the night already passed when the gateway starts.
        self._next = self._after(self._now())

    # -- kept --------------------------------------------------------------------

    def _read(self) -> Dict[str, Any]:
        try:
            with open(self._path) as f:
                kept = json.load(f)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            log.warning("Copies in %s not read: %s", self._path, error)
            return {}
        return kept if isinstance(kept, dict) else {}

    def _write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        partial = self._path.with_name(self._path.name + ".partial")
        with open(partial, "w") as f:
            json.dump({"folder": self._folder, "last": self._last}, f, indent=2)
        os.replace(partial, self._path)

    @staticmethod
    def _after(moment: datetime) -> datetime:
        at = datetime.combine(moment.date(), NIGHTLY)
        return at if at > moment else at + timedelta(days=1)

    # -- reading -----------------------------------------------------------------

    def state(self) -> Dict[str, Any]:
        with self._lock:
            folder, last = self._folder, self._last
            copying = self._thread is not None and self._thread.is_alive()
            upcoming = self._next
        return {
            "folder": folder,
            "available": bool(folder) and FolderStore(Path(folder)).available(),
            "copying": copying,
            "last": last,
            "next": upcoming.isoformat(timespec="minutes") if folder else None,
        }

    # -- acting ------------------------------------------------------------------

    def set_folder(self, folder: Optional[str]) -> Dict[str, Any]:
        """Name the shared folder, or none to turn the copy off. It must be
        there now: a mistyped one would otherwise fail every night unseen."""
        folder = (folder or "").strip() or None
        if folder is not None:
            if not Path(folder).is_absolute():
                raise GatewayRefused(422, "bad_folder", "Give the folder's full path.")
            if not Path(folder).is_dir():
                raise GatewayRefused(
                    422, "no_folder", f"{folder} isn't a folder this PC can see."
                )
        with self._lock:
            self._folder = folder
            self._write()
        return self.state()

    def copy_now(self, who: str) -> Dict[str, Any]:
        """Start a copy now, whatever runs: a person asked. Raises
        `GatewayRefused` when there is nowhere to copy to."""
        self._start(who)
        return self.state()

    def poll(self) -> None:
        """Start the nightly copy when its time has come. One that would
        start while a workflow runs is tried again at the next poll."""
        now = self._now()
        with self._lock:
            due = self._folder is not None and now >= self._next
        if not due or self._running():
            return
        with self._lock:
            self._next = self._after(now)
        try:
            self._start("nightly")
        except GatewayRefused as refusal:
            self._finish({"at": now.isoformat(timespec="seconds"), "why": str(refusal)})

    def stop(self) -> None:
        """End a copy at its next file; the next one carries on from there."""
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=10.0)

    def _start(self, who: str) -> None:
        with self._lock:
            if self._folder is None:
                raise GatewayRefused(409, "no_folder", "No shared folder is set.")
            if self._thread is not None and self._thread.is_alive():
                raise GatewayRefused(409, "copying", "A copy is already going.")
            store = FolderStore(Path(self._folder))
            if not store.available():
                raise GatewayRefused(
                    503, "folder_gone", f"{self._folder} can't be reached now."
                )
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._copy, args=(store, who), name="gateway-copy", daemon=True
            )
            self._thread.start()

    def _copy(self, store: FolderStore, who: str) -> None:
        started = self._now()
        total = {"experiments": 0, "sent": 0, "same": 0, "bytes": 0, "failed": 0}
        complete, label, taken = True, self._label(), set()
        try:
            for folder in self._folders():
                # Two folders of one name on this PC: the later is told apart.
                name, n = folder.name, 1
                while name in taken:
                    n += 1
                    name = f"{folder.name}-{n}"
                taken.add(name)
                done = copy_up(folder, store, f"{label}/{name}", self._stop.is_set)
                total["experiments"] += 1
                total["sent"] += done.sent
                total["same"] += done.same
                total["bytes"] += done.bytes
                total["failed"] += len(done.failed)
                complete = complete and done.complete
                if done.stopped:
                    break
            why = None if complete else "Some files were not copied."
        except Exception as error:  # noqa: BLE001 - a copy never takes the gateway down
            log.exception("The copy to %s failed", store.root)
            complete, why = False, str(error)
        self._finish(
            {
                "at": started.isoformat(timespec="seconds"),
                "who": who,
                **total,
                "complete": complete,
                "why": why,
            }
        )

    def _finish(self, last: Dict[str, Any]) -> None:
        with self._lock:
            self._last = last
            try:
                self._write()
            except OSError as error:
                log.warning("The last copy wasn't written down: %s", error)
        log.info("Copy to the shared folder: %s", last)

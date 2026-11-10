"""Where an agent session's own files go: its plan and its transcript.

One session works on one experiment. Its files live in that experiment's
``agent/`` folder, beside its record. A session can start before any
experiment is open (opening one is the first step the agent proposes at
intake, never a rule), so until one is, its files wait in the session's
folder under the state root (``~/.fibsem/agent/sessions/<date>-<id>/agent/``).
The first experiment open while the session runs is the session's: its
waiting files move in, and its files stay there even if the operator then
opens another experiment.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Optional

from fibsem import paths
from fibsem.ai.agent import manager

FOLDER = "agent"
PREFIXES = ("plan-", "transcript-")


def _waiting(session_id: str) -> Path:
    """The session's own folder (fibsem-agent's, on this PC), under the
    state root."""
    root = paths.state_root() / "agent" / "sessions"
    return manager.session_folder(session_id, root) / FOLDER


class AgentFiles:
    """The folder each session's files go in, and the experiment it took."""

    def __init__(
        self,
        experiment_path: Callable[[], Optional[str]],
        waiting: Optional[Callable[[str], Path]] = None,
    ) -> None:
        self._experiment_path = experiment_path
        # Where a session's files wait while no experiment is open.
        self._waiting = waiting or _waiting
        # session id -> the experiment it took.
        self._taken: Dict[str, str] = {}

    def _open(self) -> Optional[str]:
        try:
            path = self._experiment_path()
        except Exception:  # noqa: BLE001 - no experiment is the answer
            logging.debug("could not read the open experiment", exc_info=True)
            return None
        return path if path and os.path.isdir(path) else None

    def experiment(self, session_id: Optional[str]) -> Optional[str]:
        """The experiment session *session_id* works on: the one it took,
        else the one open now, which it takes. None while none is open."""
        if not session_id:
            return self._open()
        taken = self._taken.get(session_id)
        if taken and os.path.isdir(taken):
            return taken
        found = self._open()
        if found is not None:
            self._taken[session_id] = found
            self._move_in(session_id, Path(found) / FOLDER)
        return found

    def folder(self, session_id: Optional[str]) -> Optional[Path]:
        """Where session *session_id*'s files go: its experiment's agent/
        folder, else its own folder's. None without a session or an
        experiment."""
        found = self.experiment(session_id)
        if found is not None:
            return Path(found) / FOLDER
        return self._waiting(session_id) if session_id else None

    def _move_in(self, session_id: str, target: Path) -> None:
        """The session's waiting files, into its experiment: a transcript
        adds to one there, a plan there wins."""
        source = self._waiting(session_id)
        if not source.is_dir() or source == target:
            return
        for prefix in PREFIXES:
            for path in source.glob(f"{prefix}{session_id}.*"):
                moved = target / path.name
                try:
                    target.mkdir(parents=True, exist_ok=True)
                    if not moved.exists():
                        shutil.move(str(path), moved)
                    elif prefix == "transcript-":
                        with open(moved, "a", encoding="utf-8") as out:
                            out.write(path.read_text(encoding="utf-8"))
                        path.unlink()
                except OSError:
                    logging.warning(
                        "could not move %s into %s", path, target, exc_info=True
                    )

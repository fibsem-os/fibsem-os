"""The Claude Code session, copied into the agent's session folder as the CLI
writes it.

The CLI keeps its transcript under ``~/.claude/projects``; the SDK hands every
line it writes there to a session store as well. This one appends them to
``transcript.jsonl`` in the session folder, and a specialist's to
``transcript/<its file>.jsonl``. Resuming still reads the CLI's own copy:
``load`` finds nothing here, so the SDK leaves ``--resume`` as it was.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

FILE = "transcript.jsonl"
FOLDER = "transcript"


class TranscriptMirror:
    """A session store (``ClaudeAgentOptions.session_store``) that writes
    each session's lines into *folder*; *session_id* is the main session's."""

    def __init__(self, folder: Path, session_id: Optional[str] = None) -> None:
        self.folder = folder
        self.session_id = session_id
        self._warned = False

    def path(self, key: Dict[str, Any]) -> Path:
        """Where a session's lines go: the main one's in ``transcript.jsonl``,
        any other's in ``transcript/``."""
        session, subpath = key.get("session_id"), key.get("subpath")
        if session == self.session_id and not subpath:
            return self.folder / FILE
        name = f"{session}/{subpath}" if subpath else session
        return self.folder / FOLDER / f"{name}.jsonl"

    async def append(self, key: Dict[str, Any], entries: List[dict]) -> None:
        await asyncio.to_thread(self._write, self.path(key), entries)

    async def load(self, key: Dict[str, Any]) -> None:
        # Resuming reads the CLI's own copy.
        return None

    def _write(self, path: Path, entries: List[dict]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as file:
                for entry in entries:
                    file.write(json.dumps(entry) + "\n")
        except OSError as error:
            if not self._warned:
                log.warning("Couldn't write %s: %s", path, error)
                self._warned = True

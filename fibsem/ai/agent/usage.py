"""What the agent's model calls used: tokens per call, per turn and per run,
and how full its context is.

Each call is one JSON line in the session folder's ``usage.jsonl``, so a run
review can split the tokens by phase, item or tool from the call times. Needs
nothing from the SDK: the harness hands in each message's ``usage``.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from fibsem.util import timestamps

log = logging.getLogger(__name__)

FILE = "usage.jsonl"
FIELDS = ("input", "output", "cache_read", "cache_write")
# The API's names for them.
_API = {
    "input": "input_tokens",
    "output": "output_tokens",
    "cache_read": "cache_read_input_tokens",
    "cache_write": "cache_creation_input_tokens",
}


def zero() -> Dict[str, int]:
    return dict.fromkeys(FIELDS, 0)


def tokens(usage: Optional[dict]) -> Dict[str, int]:
    """The API's usage as {input, output, cache_read, cache_write}."""
    usage = usage or {}
    return {key: int(usage.get(name) or 0) for key, name in _API.items()}


def add(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    return {key: a.get(key, 0) + b.get(key, 0) for key in FIELDS}


def prompt(counts: Dict[str, int]) -> int:
    """Everything the call read: new input, cached and newly cached."""
    return counts["input"] + counts["cache_read"] + counts["cache_write"]


def short(count: Optional[int]) -> str:
    """412 tokens as "412", 41_200 as "41k", 1_200_000 as "1.2M"."""
    if count is None:
        return "?"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}".rstrip("0").rstrip(".") + "M"
    if count >= 10_000:
        return f"{count / 1000:.0f}k"
    if count >= 1000:
        return f"{count / 1000:.1f}".rstrip("0").rstrip(".") + "k"
    return str(count)


def context_text(context: Optional[dict]) -> Optional[str]:
    """ "Context 412k of 1M, compacts at 970k"; None when nothing is known."""
    if not context or context.get("used") is None:
        return None
    text = f"Context {short(context['used'])}"
    if context.get("window"):
        text += f" of {short(context['window'])}"
    if context.get("compact_at"):
        text += f", compacts at {short(context['compact_at'])}"
    return text


class Calls:
    """The model calls of a run, each one line in *path* (none when None).

    The SDK sends a call's content blocks as separate messages that share the
    call's id and usage; they are one call here, written once the next call
    starts or the turn ends."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.turn = 0
        self.turn_tokens = zero()
        # How much the last call of the agent itself (not a specialist) read.
        self.context: Optional[int] = None
        self._open: Optional[dict] = None
        self._warned = False

    def seen(
        self,
        message_id: Optional[str],
        model: Optional[str],
        usage: Optional[dict],
        tools: List[str],
        agent: Optional[str] = None,
    ) -> None:
        """One message of a call; *agent* is the specialist's call id, if a
        specialist made it."""
        if self._open is not None and (
            message_id is None or self._open["message_id"] != message_id
        ):
            self._close()
        if self._open is None:
            self._open = {
                "message_id": message_id,
                "model": model,
                "agent": agent,
                **zero(),
                "tools": [],
            }
        call = self._open
        # Later blocks carry the call's usage as it grows: keep the largest.
        for key, value in tokens(usage).items():
            call[key] = max(call[key], value)
        call["tools"].extend(tools)

    def end_turn(self) -> Dict[str, int]:
        """The tokens of the turn just ended."""
        self._close()
        used, self.turn_tokens = self.turn_tokens, zero()
        self.turn += 1
        return used

    def _close(self) -> None:
        call, self._open = self._open, None
        if call is None:
            return
        counts = {key: call[key] for key in FIELDS}
        self.turn_tokens = add(self.turn_tokens, counts)
        if call["agent"] is None:
            self.context = prompt(counts)
        if self.path is None:
            return
        line = {
            "at": timestamps.now().isoformat(),
            "turn": self.turn + 1,
            **{key: value for key, value in call.items() if value is not None},
        }
        try:
            with open(self.path, "a", encoding="utf-8") as file:
                file.write(json.dumps(line) + "\n")
        except OSError as error:
            if not self._warned:
                log.warning("Couldn't write %s: %s", self.path, error)
                self._warned = True

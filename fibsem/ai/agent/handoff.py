"""A fresh context for the same session: the agent's ``context_reset`` tool,
the handoff note it writes, and the context sizes it is told of.

The agent calls ``context_reset`` with its note; when that turn ends the
harness keeps the note (``handoff.md`` in the session folder, and
``agent/handoff-<session id>.md`` beside the plan through the App Server),
starts a new CLI conversation under the same session, and gives it the
instrument kind's ``handoff.md`` ``start`` text with the note as its first
prompt. The session's id, chat, budget, spend and plan stay as they were.
Each context's model calls are re-read on every turn, so the harness also
says once, in a wake-up, when the context first passes each of the
playbook's ``context_marks``; the agent decides what to do about it. Needs
nothing from the SDK at import.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from fibsem.ai.agent import instruments, usage
from fibsem.util import timestamps

log = logging.getLogger("fibsem.agent.handoff")

TOOL = "context_reset"
# Every note of the session, newest last, in its folder.
FILE = "handoff.md"
# The note's parts: (argument, heading), the first three required.
PARTS: Tuple[Tuple[str, str], ...] = (
    ("done", "Done"),
    ("in_flight", "In flight"),
    ("next", "Next"),
    ("rules", "Standing rules not yet in the plan"),
    ("other", "Also"),
)
REQUIRED = ("done", "in_flight", "next")
MAX_PART = 8000
# The CLI's auto-compact window (`playbooks.SessionSettings.compact_window`).
COMPACT_WINDOW_ENV = "CLAUDE_CODE_AUTO_COMPACT_WINDOW"

DESCRIPTION = (
    "Start a fresh context when this turn ends: the same session, operator "
    "chat, budget, plan and experiment, without this conversation. Every turn "
    "re-reads the whole context, so a large one costs more each step; worth "
    "it at a natural boundary, not mid-task. Bring the plan up to date first, "
    "then write the note for an agent who knows the skills but nothing of "
    "today. done: what was done since the plan was last brought up to date. "
    "in_flight: what is running or waiting (a workflow and its task, proposal "
    "or request ids, a command's run id, a wake-up you set). next: the next "
    "steps, concretely. rules: the operator's standing instructions not yet in "
    "the plan's Rules. other: anything else the record doesn't hold (what you "
    "tried that failed, what to watch for). The fresh context reads its "
    "skills, the plan and this note, then where things stand, and carries on "
    "from next. End your turn after calling it."
)
SCHEMA = {
    "type": "object",
    "properties": {
        name: {"type": "string", "description": heading} for name, heading in PARTS
    },
    "required": list(REQUIRED),
}

# Said in the turn after a fresh context could not start.
FAILED = (
    "The fresh context you asked for could not start ({why}): you are still in "
    "the same context. Carry on here; you may call context_reset again later."
)
# The wake-up that starts the fresh context's first turn.
WAKE = "WAKE a fresh context: start from the handoff note above."


@dataclass
class Note:
    """One handoff note: its parts by argument name, when it was written and
    the context it hands to (2 for the session's first fresh context)."""

    parts: Dict[str, str]
    at: str = field(
        default_factory=lambda: timestamps.now().isoformat(timespec="seconds")
    )
    to: int = 2

    def markdown(self) -> str:
        lines = [f"## Handoff to context {self.to}, {self.at}"]
        for name, heading in PARTS:
            text = self.parts.get(name, "").strip()
            if text:
                lines += ["", f"### {heading}", "", text]
        return "\n".join(lines) + "\n"


class Marks:
    """The context sizes the agent is told of, each once while the context
    is above it: a mark above the context's size (after a compaction or a
    fresh context) is said again when it is passed again."""

    def __init__(self, marks: Optional[Sequence[int]] = None) -> None:
        self.marks = sorted(set(marks or ()))
        self._said: set = set()

    def crossed(self, used: Optional[int]) -> Optional[int]:
        """The highest mark *used* tokens first passes, or None."""
        if not used:
            return None
        self._said = {mark for mark in self._said if mark <= used}
        passed = [
            mark for mark in self.marks if mark <= used and mark not in self._said
        ]
        self._said.update(passed)
        return max(passed) if passed else None


class Handoff:
    """The session's fresh contexts: the note the agent asked one with
    (``pending``), the prompt and lines its next turn starts with, and how
    many contexts the session has had. *keep* writes a note to the session's
    folder; *contexts* is the count a resumed session had."""

    def __init__(
        self,
        words: instruments.Texts,
        marks: Optional[Sequence[int]] = None,
        keep: Optional[Callable[[str], None]] = None,
        contexts: int = 1,
    ) -> None:
        self.words = words
        self.marks = Marks(marks)
        self.contexts = contexts
        self.pending: Optional[Note] = None
        self._keep = keep
        # The fresh context's first prompt, and lines for the next turn.
        self._prompt: Optional[str] = None
        self._lines: List[str] = []

    def request(self, args: Dict[str, Any]) -> str:
        """The ``context_reset`` tool: what the agent is told."""
        if not self.words.handoff_start:
            return (
                "Refused: this session's instrument kind has no handoff.md, so "
                "it can't start a fresh context."
            )
        parts = {
            name: str(args.get(name) or "").strip()[:MAX_PART] for name, _ in PARTS
        }
        missing = [name for name in REQUIRED if not parts[name]]
        if missing:
            return f"Refused: the note needs {', '.join(missing)}."
        again = self.pending is not None
        self.pending = Note(parts, to=self.contexts + 1)
        log.info("The agent asked for a fresh context.")
        return (
            ("This note replaces the one you gave earlier this turn. " if again else "")
            + "A fresh context starts when this turn ends, with your skills, "
            "the plan and this note. End your turn now."
        )

    def take(self) -> Optional[Note]:
        note, self.pending = self.pending, None
        return note

    def keep(self, note: Note) -> None:
        """*note* in the session's folder, after the earlier ones."""
        if self._keep is None:
            return
        try:
            self._keep(note.markdown())
        except Exception:  # noqa: BLE001 - the note still reaches the agent
            log.warning("The handoff note could not be kept.", exc_info=True)

    def started(self, note: Note, skills: List[str], read: List[str], file: str):
        """A fresh context started: its first turn starts with the note."""
        self.contexts = note.to
        self._prompt = self.words.handoff_start.format(
            skills=", ".join(skills) or "none",
            read_before=", ".join(read) or "no others",
            note=note.markdown().strip(),
            file=file,
        )
        # A size said of the earlier context is not this one's.
        self._lines = []

    def failed(self, why: str) -> None:
        self._lines.append(FAILED.format(why=why))

    def measure(self, context: Optional[dict], turn_cost: Optional[float]) -> None:
        """After a turn: a line for the next one when the context first
        passes a mark."""
        context = context or {}
        if self.marks.crossed(context.get("used")) is None:
            return
        if not self.words.handoff_mark:
            return
        self._lines.append(
            self.words.handoff_mark.format(
                used=usage.short(context.get("used")),
                window=usage.short(context.get("window")),
                turn_cost=f"${turn_cost:.2f}" if turn_cost is not None else "unknown",
            )
        )

    def into(self, prompt: str) -> str:
        """*prompt* as the next turn says it: after a fresh context's first
        prompt, before the lines waiting for it."""
        if self._prompt:
            prompt = f"{self._prompt}\n\n{prompt}"
        for line in self._lines:
            prompt += f"\n\n{line}"
        self._prompt, self._lines = None, []
        return prompt


def skills_to_read(
    playbook, loaded: Dict[str, str], read: List[str], first: Optional[str] = None
) -> Tuple[List[str], List[str]]:
    """(The skills a fresh context reads: *first*, the *playbook*'s and its
    optional ones *loaded* now; the others it had *read* before.)"""
    skills = [first] if first else []
    if playbook is not None:
        skills += list(playbook.skills)
        skills += [
            o.skill for o in playbook.optional if loaded.get(o.skill) == "loaded"
        ]
    skills = list(dict.fromkeys(skills))
    return skills, [name for name in read if name not in skills]


def post(client, session_id: str, text: str) -> Optional[str]:
    """*text* added to the session's handoff file at the App Server; the
    file's path in the experiment as it answers, or None."""
    try:
        response = client.request(
            "POST",
            "/app/agent/handoff",
            json={"session_id": session_id, "text": text},
            timeout=10.0,
        )
        if response.status_code == 200:
            return response.json().get("path")
        log.warning(
            "The App Server didn't keep the handoff note (%s).", response.status_code
        )
    except Exception:  # noqa: BLE001 - kept in the session's folder anyway
        log.warning("Sending the handoff note failed.", exc_info=True)
    return None


def compact_hooks(hooks: Optional[dict], instructions: str) -> Optional[dict]:
    """*hooks* with a ``PreCompact`` hook that gives the CLI's compaction
    *instructions*: the CLI adds a callback hook's ``systemMessage`` to the
    compaction's custom instructions."""
    if not instructions:
        return hooks
    from claude_agent_sdk import HookMatcher

    async def hook(hook_input, tool_use_id, context):
        log.info(
            "Compacting (%s): asked to keep the handoff facts.",
            hook_input.get("trigger"),
        )
        return {"systemMessage": instructions}

    hooks = dict(hooks or {})
    hooks["PreCompact"] = list(hooks.get("PreCompact", [])) + [
        HookMatcher(hooks=[hook])
    ]
    return hooks


def compact_env(window: Optional[int]) -> Dict[str, str]:
    """The CLI's environment for an auto-compact *window* in tokens; none for
    the CLI's own."""
    return {COMPACT_WINDOW_ENV: str(int(window))} if window else {}

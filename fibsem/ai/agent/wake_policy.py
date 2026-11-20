"""What wakes the agent: its ``wake_on`` policy, and the digest of what waited.

Every wake re-reads the agent's whole context, so a wake costs what the
context costs, not what the line is about. Under ``all`` every watcher line
wakes the agent; under ``digest`` the routine ones (`wake_lines.NEEDS`) wait
and arrive together, at the top of a later wake: with the next line that
wakes it, once ``every_min`` has passed since the first was held, or once
``after_lines`` are held. Nothing is dropped: a held line always reaches the
agent, and the full record stays in app_get_events.

The playbook's ``session:`` block sets the policy for each level
(``wake_on_watch`` and so on). The agent changes it with the harness's
``wake_on`` tool, and from then on a level change keeps the agent's choice.
Separately, a long command's end whose result the agent already holds (it
read it with service_wait_for_run) never wakes it, under any policy.

The words the agent reads about all this are here, in one place.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from fibsem.ai.agent import usage, wake_lines
from fibsem.ai.agent import watch as watch_mod

log = logging.getLogger("fibsem.agent.harness")

ALL, DIGEST = wake_lines.WAKE_MODES
# The harness's own line when the digest is due: it starts a turn, and is
# never shown.
DIGEST_DUE = "digest_due"
# How long the CLI keeps the agent's prompt cache (CLAUDE_CODE_PROMPT_CACHE_TTL,
# set by fibsem-agent for the main conversation; its specialists keep the CLI's
# 5 min, as they run back to back). A wake after a quiet spell longer than this
# re-reads the whole context uncached.
PROMPT_CACHE_TTL = "1h"
CACHE_COLD_S = 60 * 60.0
# A context this large costs enough per wake to say so, and to let held lines
# ride past their count when the cache has gone cold.
LARGE_CONTEXT = 200_000
# The held lines a digest shows, newest first kept; the rest are counted.
DIGEST_SHOWN = 40
# The held lines kept at all; older ones are counted, by kind.
HELD_KEPT = 2000
# The long commands whose end the agent has read, newest kept.
RUNS_KEPT = 64
LEVELS = ("watch", "assist", "drive")


@dataclass
class Settings:
    """A session's policy defaults: the mode at each level, and when a digest
    is due. With none given every line wakes the agent."""

    watch: str = ALL
    assist: str = ALL
    drive: str = ALL
    every_min: float = 30.0
    after_lines: int = 20

    @classmethod
    def of(cls, session: Any) -> "Settings":
        """From a playbook's session settings; its blanks keep these defaults."""
        given = {
            "watch": getattr(session, "wake_on_watch", None),
            "assist": getattr(session, "wake_on_assist", None),
            "drive": getattr(session, "wake_on_drive", None),
            "every_min": getattr(session, "digest_min", None),
            "after_lines": getattr(session, "digest_lines", None),
        }
        return cls(**{key: value for key, value in given.items() if value is not None})

    def mode_at(self, level: Optional[str]) -> str:
        """The mode for *level*; ``all`` for a level not known."""
        return getattr(self, level, ALL) if level in LEVELS else ALL


def ended_runs(name: str, content: Any) -> List[str]:
    """The long commands a tool result shows ended: a service_wait_for_run
    answer's run, once it is no longer running. *content* is the result as
    the SDK gives it (text items holding JSON)."""
    if not str(name).endswith("service_wait_for_run"):
        return []
    found = []
    items = content if isinstance(content, list) else [content]
    for item in items:
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        run = data.get("run", data) if isinstance(data, dict) else None
        if (
            isinstance(run, dict)
            and isinstance(run.get("run_id"), str)
            and run.get("state") not in (None, "running")
        ):
            found.append(run["run_id"])
    return found


class WakeOn:
    """The session's wake policy, and the lines it holds.

    *say* puts a change on the transcript the operator reads; *keep* stores
    the agent's own choice in the session record, for a resume; *context*
    answers (seconds since the agent's last turn, its context's size), either
    None when not known."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        say: Optional[Callable[[str], None]] = None,
        keep: Optional[Callable[[Optional[dict]], None]] = None,
        context: Optional[Callable[[], Tuple[Optional[float], Optional[int]]]] = None,
    ) -> None:
        self.settings = settings or Settings()
        self.say = say or (lambda text: None)
        self.keep = keep or (lambda state: None)
        self.context = context
        self.level: Optional[str] = None
        self.mode = self.settings.mode_at(None)
        self.every_min = self.settings.every_min
        self.after_lines = self.settings.after_lines
        # Routine kinds that wake the agent under digest, and that wait under all.
        self.wake: frozenset = frozenset()
        self.hold: frozenset = frozenset()
        # "default" (the playbook's, for the level) or "agent".
        self.set_by = "default"
        self.held: List[str] = []
        # The held lines that count towards a digest (`keep_held`).
        self._counted = 0
        # Held lines past HELD_KEPT, by kind: counted, not kept.
        self._dropped: Dict[str, int] = {}
        self._first_at: Optional[str] = None
        self._due: Optional[str] = None
        self._runs: List[str] = []
        self._told_idle = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ring: Optional[Callable[[str], None]] = None
        self._timer: Optional[asyncio.TimerHandle] = None

    # --- the harness's side ------------------------------------------------

    def bind(self, loop: asyncio.AbstractEventLoop, ring: Callable[[str], None]):
        """Ring a digest that is due through *ring* on *loop* (`Harness.run`)."""
        self._loop, self._ring = loop, ring

    def holds(self, line: str) -> bool:
        """Whether *line* waits for a digest rather than waking the agent."""
        if self._known(line):
            # The agent read this end itself: news to no one.
            return True
        kind = getattr(line, "kind", None)
        facts = getattr(line, "facts", None) or {}
        need = wake_lines.needs(kind)
        routine = need == wake_lines.HOLD or (
            need == wake_lines.HOLD_IF_OK and facts.get("ok") is True
        )
        if not routine:
            return False
        if kind in self.hold:
            return True
        return self.mode == DIGEST and kind not in self.wake

    def keep_held(self, line: str) -> None:
        """Hold *line* for the next digest; the interval starts with the first.
        A command's end the agent already read rides along: it neither starts
        the interval nor counts towards the lines that make a digest due."""
        if not self.held:
            self._first_at = datetime.now().strftime("%H:%M")
        if not self._known(line):
            self._counted += 1
            if self._counted == 1:
                self._start_timer()
        self.held.append(line)
        while len(self.held) > HELD_KEPT:
            gone = self.held.pop(0)
            kind = getattr(gone, "kind", None) or "line"
            self._dropped[kind] = self._dropped.get(kind, 0) + 1
        log.info("Held for the digest (%d held): %s", len(self.held), line)

    def due(self) -> Optional[str]:
        """Why the held lines are due now, without a line that wakes the agent;
        None while they wait. A count reached on a large context gone cold
        waits for the interval or a line that wakes."""
        if not self._counted:
            return None
        if self._due:
            return self._due
        if self.after_lines and self._counted >= self.after_lines:
            if not self._cold():
                return f"at {self._counted} lines held"
        return None

    def take(self, why: str = "with this wake") -> Tuple[List[str], Optional[str]]:
        """The held lines and their digest, which empties the hold."""
        if not self.held:
            return [], None
        lines, dropped, first = self.held, self._dropped, self._first_at
        self.held, self._dropped, self._first_at, self._due = [], {}, None, None
        self._counted = 0
        self._cancel_timer()
        text = digest(lines, why, self.mode, first, dropped)
        self.say(f"Woke with {len(lines)} held line(s), {why}.")
        log.info("Digest: %d held line(s) delivered, %s.", len(lines), why)
        return lines, text

    def _known(self, line: str) -> bool:
        """Whether *line* is a long command's end, done, that the agent read
        itself (`saw_run_end`)."""
        facts = getattr(line, "facts", None) or {}
        return (
            getattr(line, "kind", None) == watch_mod.COMMAND_ENDED
            and facts.get("state") == "done"
            and facts.get("run_id") in self._runs
        )

    def saw_run_end(self, run_id: str) -> None:
        """The agent read run *run_id*'s end in a tool result."""
        if run_id in self._runs:
            return
        self._runs.append(run_id)
        del self._runs[:-RUNS_KEPT]

    def at_level(self, level: Optional[str], initial: bool = False) -> Optional[str]:
        """The operator set *level* (*initial*: as the session found it).
        The level's default applies unless the agent chose its own; what to
        tell the agent, when what wakes it changed or its choice was kept
        across Watch."""
        previous, self.level = self.level, level
        if level != "watch":
            self._told_idle = False
        if initial or previous == level:
            if self.set_by == "default":
                self._apply(self.settings.mode_at(level))
            return None
        if self.set_by == "agent":
            if "watch" in (previous, level):
                return LEVEL_KEPT.format(policy=self.summary())
            return None
        before = self.mode
        self._apply(self.settings.mode_at(level))
        if self.mode == before:
            return None
        self.say(f"Wake policy: {self.mode}, the default at {level}.")
        log.info("Wake policy: %s, the default at %s.", self.mode, level)
        return LEVEL_CHANGED.format(level=level, policy=self.summary())

    def idle_note(self) -> Optional[str]:
        """Once each time at Watch, when the context is large: what a wake
        costs now, and what the agent can do about it."""
        if self.level != "watch" or self._told_idle:
            return None
        _, used = self.context() if self.context else (None, None)
        if not used or used < LARGE_CONTEXT:
            return None
        self._told_idle = True
        return IDLE.format(used=usage.short(used), mode=self.mode)

    # --- the agent's side: the wake_on tool --------------------------------

    def set(
        self,
        mode: str = "",
        every_min: Optional[float] = None,
        after_lines: Optional[int] = None,
        wake: str = "",
        hold: str = "",
    ) -> str:
        """The wake_on tool: change what is given, keep the rest; answer the
        policy now. ``mode`` ``default`` goes back to the playbook's."""
        mode = (mode or "").strip().lower()
        if mode and mode not in (*wake_lines.WAKE_MODES, "default"):
            return f"Refused: mode is one of all, digest or default, not {mode!r}."
        kinds = {}
        for name, given in (("wake", wake), ("hold", hold)):
            parsed, why = _kinds(given)
            if why:
                return f"Refused: {name} {why}"
            kinds[name] = parsed
        if every_min is not None and every_min < 0:
            return "Refused: every_min is 0 (no interval) or more."
        if after_lines is not None and after_lines < 0:
            return "Refused: after_lines is 0 (no count) or more."
        if mode == "default":
            self.set_by = "default"
            self._apply(self.settings.mode_at(self.level))
            self.wake = self.hold = frozenset()
            self.keep(None)
        else:
            self.set_by = "agent"
            if mode:
                self.mode = mode
            if every_min is not None:
                self.every_min = float(every_min)
                self._restart_timer()
            if after_lines is not None:
                self.after_lines = int(after_lines)
            if kinds["wake"] is not None:
                self.wake = kinds["wake"]
            if kinds["hold"] is not None:
                self.hold = kinds["hold"]
            self.keep(self.state())
        self.say(f"Wake policy: {self.summary()}, set by the agent.")
        log.info("Wake policy: %s, set by the agent.", self.summary())
        return self.describe()

    def describe(self) -> str:
        """The policy as the tool answers it: what wakes the agent, what
        waits, and what is held now."""
        whose = (
            "your own choice"
            if self.set_by == "agent"
            else f"the playbook's default at {self.level or 'this level'}"
        )
        parts = [f"wake_on {self.summary()} ({whose})."]
        if self.mode == DIGEST:
            parts.append(DIGEST_ABOUT.format(due=self._due_text()))
        else:
            parts.append(ALL_ABOUT)
        held = self._held_text()
        parts.append(f"Held now: {held}." if held else "Nothing is held now.")
        parts.append(TOOL_HOW)
        return " ".join(parts)

    def summary(self) -> str:
        """``digest (every 30 min or 20 lines; also wake on mill_ended)``."""
        text = self.mode
        bits = []
        if self.mode == DIGEST or self.hold:
            bits.append(self._due_text())
        if self.wake:
            bits.append("also wake on " + ", ".join(sorted(self.wake)))
        if self.hold:
            bits.append("also hold " + ", ".join(sorted(self.hold)))
        return f"{text} ({'; '.join(bits)})" if bits else text

    def state(self) -> Optional[dict]:
        """The agent's own choice, for the session record; None for the
        playbook's default."""
        if self.set_by != "agent":
            return None
        return {
            "mode": self.mode,
            "every_min": self.every_min,
            "after_lines": self.after_lines,
            "wake": sorted(self.wake),
            "hold": sorted(self.hold),
        }

    def restore(self, state: Optional[dict]) -> None:
        """Take up the agent's choice kept by `state`, on a resume."""
        if not isinstance(state, dict) or state.get("mode") not in (
            wake_lines.WAKE_MODES
        ):
            return
        self.set_by = "agent"
        self.mode = state["mode"]
        self.every_min = float(state.get("every_min") or 0)
        self.after_lines = int(state.get("after_lines") or 0)
        self.wake = frozenset(k for k in state.get("wake") or [] if k in _ROUTINE)
        self.hold = frozenset(k for k in state.get("hold") or [] if k in _ROUTINE)

    # --- inside -------------------------------------------------------------

    def _apply(self, mode: str) -> None:
        self.mode = mode
        self.every_min = self.settings.every_min
        self.after_lines = self.settings.after_lines

    def _cold(self) -> bool:
        idle, used = self.context() if self.context else (None, None)
        return (
            idle is not None
            and used is not None
            and idle >= CACHE_COLD_S
            and used >= LARGE_CONTEXT
        )

    def _due_text(self) -> str:
        when = []
        if self.every_min:
            when.append(f"every {self.every_min:g} min")
        if self.after_lines:
            when.append(f"{self.after_lines} lines")
        return " or ".join(when) or "only with a line that wakes you"

    def _held_text(self) -> str:
        if not self.held and not self._dropped:
            return ""
        counts: Dict[str, int] = dict(self._dropped)
        for line in self.held:
            kind = getattr(line, "kind", None) or "line"
            counts[kind] = counts.get(kind, 0) + 1
        total = sum(counts.values())
        each = ", ".join(f"{n} {kind}" for kind, n in counts.items())
        return f"{total} line(s) since {self._first_at} ({each})"

    def _start_timer(self) -> None:
        if self._loop is None or not self.every_min or self._timer is not None:
            return
        self._timer = self._loop.call_later(self.every_min * 60, self._rang)

    def _restart_timer(self) -> None:
        self._cancel_timer()
        if self._counted:
            self._start_timer()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None

    def _rang(self) -> None:
        self._timer = None
        if not self._counted:
            return
        self._due = f"{self.every_min:g} min after the first was held"
        if self._ring is not None:
            self._ring(watch_mod.Wake(DIGEST_DUE, ""))


# The kinds a policy's per-kind lists may name: the routine ones.
_ROUTINE = frozenset(
    kind
    for kind, need in wake_lines.NEEDS.items()
    if need in (wake_lines.HOLD, wake_lines.HOLD_IF_OK)
)


def _kinds(given: str) -> Tuple[Optional[frozenset], Optional[str]]:
    """Line kinds named in *given* (``mill_ended, autofocus``): None to keep
    the list, an empty set for ``none``; or why they are refused."""
    given = (given or "").strip()
    if not given:
        return None, None
    if given.lower() == "none":
        return frozenset(), None
    names = [name.strip().lower() for name in given.split(",") if name.strip()]
    wrong = [name for name in names if name not in _ROUTINE]
    if wrong:
        return None, (
            f"names {', '.join(wrong)}: only routine kinds can be listed "
            f"({', '.join(sorted(_ROUTINE))}); the rest always wake you."
        )
    return frozenset(names), None


def digest(
    lines: List[str],
    why: str,
    mode: str = DIGEST,
    first: Optional[str] = None,
    dropped: Optional[Dict[str, int]] = None,
) -> str:
    """The held *lines* as one block: runs of the same kind on one line,
    each item kept; past `DIGEST_SHOWN` the oldest are counted, not shown."""
    dropped = dict(dropped or {})
    total = len(lines) + sum(dropped.values())
    shown = lines[-DIGEST_SHOWN:]
    for line in lines[: len(lines) - len(shown)]:
        kind = getattr(line, "kind", None) or "line"
        dropped[kind] = dropped.get(kind, 0) + 1
    since = f" since {first}" if first else ""
    head = f"Held for you{since} (wake_on {mode}), {total} line(s), delivered {why}:"
    body = list(_collapsed(shown))
    if dropped:
        counts = ", ".join(f"{n} {kind}" for kind, n in dropped.items())
        body.insert(0, CUT.format(n=sum(dropped.values()), counts=counts))
    return "\n".join([head, *body])


def _collapsed(lines: Iterable[str]) -> Iterable[str]:
    """Consecutive lines of one kind as one: ``EVENT task_completed (3):
    01-a Setup; 02-b Setup; 03-c Setup``, the words they share said once."""
    lines = list(lines)
    i = 0
    while i < len(lines):
        kind = getattr(lines[i], "kind", None)
        j = i + 1
        while j < len(lines) and getattr(lines[j], "kind", None) == kind:
            j += 1
        run = [str(line) for line in lines[i:j]]
        i = j
        shared = os.path.commonprefix(run) if len(run) > 1 else ""
        shared = shared[: shared.rfind(" ") + 1] if " " in shared else ""
        if not shared.strip():
            yield from run
            continue
        rest = "; ".join(line[len(shared) :].strip() for line in run)
        yield f"{shared.strip()} ({len(run)}): {rest}"


# --- what the agent reads ----------------------------------------------------

ALL_ABOUT = (
    "Every watcher line wakes you, except a long command's end you already "
    "read with service_wait_for_run, which comes with your next wake."
)
DIGEST_ABOUT = (
    "Routine lines wait and come together at the top of a later wake, as a "
    "digest: tasks started or finished normally, mill stages starting, a mill, "
    "autofocus or alignment that went well, someone else's long command ending "
    "done, the operator's notes on a grid. They come with the next line that "
    "wakes you, or {due}, whichever is first. Everything else wakes you at "
    "once: messages, the level, Stop, proposals and requests, failures, "
    "skipped or cancelled tasks, workflows starting and ending, a quiet "
    "workflow, deadlines and your own wake-ups."
)
TOOL_HOW = (
    "Change it with wake_on: mode all or digest (default: the playbook's for "
    "your level), every_min, after_lines, and wake or hold to list routine "
    "kinds that wake you or wait. app_get_events has the full record."
)
LEVEL_CHANGED = "What wakes you now: wake_on {policy}, the default at {level}."
LEVEL_KEPT = (
    "What wakes you is still your own choice, wake_on {policy}: wake_on "
    "mode=default takes this level's."
)
CUT = "({n} earlier held line(s) are not shown: {counts}. app_get_events has them.)"
IDLE = (
    "Your context is {used} tokens, and every wake re-reads all of it; after "
    "a quiet spell of more than an hour it is read uncached, at full price. You "
    "are at Watch with wake_on {mode}. If nothing needs you, you can hold more "
    "with wake_on, write your summary in the plan and end your turn, or start "
    "a fresh context with context_reset."
)

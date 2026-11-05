"""The agent's plan for its session: one plan.md the agent and the operator both
edit.

The agent writes it at intake under its playbook's headings (Brief, Playbook,
Rules, Steps, Progress, Summary by default) and keeps it up to date, one
section at a time (``plan_set_section``). The operator reads it on the Agent
page and can edit it; the agent is told which sections changed (a ``PLAN
EDITED`` line) and reads it again. It replaces the agent's checklist: the
session's ``checklist`` is read from it (the Rules as standing rules, the
Steps' boxes as items).

The Brief's ``Sample:`` line is the session's sample tag, which feedback and
knowledge read. It is the open experiment's sample too (its metadata): one
named when the experiment was made holds from the start, and one the agent
writes is saved on the experiment. The agent writes it once; after that only
the operator changes it. Setting it puts an ``agent_sample_set`` event on the
record.

A rule with a time ("Finish by 07:00", or a date and time) is a deadline: the
session reports it, and fibsem-agent wakes the agent before it and at it. A
clock time alone is written out with its date as the plan is saved ("by
2026-11-06 07:00", its next occurrence), so it doesn't move once passed.

Kept at ``agent/plan-<session id>.md`` in the open experiment, beside the
transcript, and in memory when no experiment is open.
"""

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from fibsem.ai.agent.levels import AgentLevel
from fibsem.server.auth import (
    AuthConfig,
    is_operator_token,
    refusal_error,
    require_level,
)

FOLDER = "agent"
SECTIONS = ("Brief", "Playbook", "Rules", "Steps", "Progress", "Summary")
PLAN_EDITED = "agent_plan_edited"
SAMPLE_SET = "agent_sample_set"
MAX_SECTION = 20_000
MAX_SECTIONS = 20
MAX_NAME = 60
_SECTION = re.compile(r"^## +(.+?)\s*$")
_GROUP = re.compile(r"^### +(.+?)\s*$")
_BOX = re.compile(r"^\s*[-*] +\[( |x|X)\] +(.+?)\s*$")
_BULLET = re.compile(r"^\s*[-*] +(.+?)\s*$")
_SAMPLE = re.compile(r"^\s*\**sample\**\s*:\s*\**\s*(.+?)\s*$", re.IGNORECASE)
_SESSION = re.compile(r"^[A-Za-z0-9_-]{1,100}$")
# A rule's deadline: "by 07:00", "by 2026-11-06 07:00" or "by 2026-11-06T07:00".
_BY = re.compile(
    r"\bby\s+\**((?:\d{4}-\d{2}-\d{2}[T ])?\d{1,2}:\d{2})(?![\d:])", re.IGNORECASE
)
# A clock time without its date, which the plan is saved with its date.
_BARE = re.compile(r"\b(by\s+\**)(\d{1,2}:\d{2})(?![\d:])", re.IGNORECASE)
TITLE = "# Plan"


def _invalid(message: str) -> HTTPException:
    return refusal_error(422, "invalid", message)


def _name(raw: Any) -> str:
    if (
        not isinstance(raw, str)
        or not raw.strip()
        or len(raw) > MAX_NAME
        or "\n" in raw
        or "#" in raw
    ):
        raise _invalid(
            f"section is a heading of at most {MAX_NAME} characters, one of "
            f"{list(SECTIONS)} or the playbook's own."
        )
    name = raw.strip()
    # The default headings in their own spelling, whatever the case given.
    return next((s for s in SECTIONS if s.lower() == name.lower()), name)


def _text(raw: Any) -> str:
    if not isinstance(raw, str) or len(raw) > MAX_SECTION:
        raise _invalid(
            f"text is the section's markdown, at most {MAX_SECTION} characters."
        )
    lines = raw.strip("\n").splitlines()
    if any(_SECTION.match(line) or line.startswith("# ") for line in lines):
        raise _invalid("A section's text has no # or ## headings: use ### for groups.")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def parse(markdown: str) -> Dict[str, str]:
    """A plan's sections by heading, in the file's order; text before the
    first ``##`` heading is dropped."""
    sections: Dict[str, str] = {}
    name: Optional[str] = None
    lines: List[str] = []
    for line in markdown.splitlines():
        found = _SECTION.match(line)
        if found:
            if name is not None:
                sections[name] = "\n".join(lines).strip("\n")
            name, lines = _name(found.group(1)), []
        elif name is not None:
            lines.append(line.rstrip())
    if name is not None:
        sections[name] = "\n".join(lines).strip("\n")
    return sections


def render(sections: Dict[str, str]) -> str:
    """The plan as markdown: the default headings in their order, a
    playbook's own (Results) in the order written, before Summary."""
    *main, last = SECTIONS
    order = (
        [s for s in main if s in sections]
        + [s for s in sections if s not in SECTIONS]
        + [last] * (last in sections)
    )
    parts = [TITLE]
    for name in order:
        parts.append(f"## {name}\n\n{sections[name]}".rstrip())
    return "\n\n".join(parts) + "\n"


def steps(text: str) -> List[Dict[str, Any]]:
    """The Steps' boxes: {group, text, done}, group the ``###`` above, "" for none."""
    found, group = [], ""
    for line in text.splitlines():
        heading = _GROUP.match(line)
        if heading:
            group = heading.group(1)
            continue
        box = _BOX.match(line)
        if box:
            found.append(
                {"group": group, "text": box.group(2), "done": box.group(1) != " "}
            )
    return found


def rules(text: str) -> List[str]:
    return [m.group(1) for m in map(_BULLET.match, text.splitlines()) if m]


def deadline(by: str, now: Optional[datetime] = None) -> datetime:
    """*by* as a time with its offset: "07:00" is its next occurrence here, an
    ISO date and time without an offset is this computer's local time."""
    now = now or datetime.now().astimezone()
    try:
        clock = datetime.strptime(by, "%H:%M")
    except ValueError:
        when = datetime.fromisoformat(by.replace(" ", "T"))
        return when if when.tzinfo else when.astimezone()
    when = now.replace(hour=clock.hour, minute=clock.minute, second=0, microsecond=0)
    return when if when > now else when + timedelta(days=1)


def rule_time(rule: str, now: Optional[datetime] = None) -> Optional[str]:
    """The time a rule gives ("by 07:00"), with its offset; None for none."""
    found = _BY.search(rule)
    if not found:
        return None
    try:
        return deadline(found.group(1), now).isoformat(timespec="minutes")
    except ValueError:
        return None


def dated(text: str, now: Optional[datetime] = None) -> str:
    """*text* with each "by 07:00" written "by 2026-11-06 07:00", its next
    occurrence; a time that is not one is left as it is."""

    def full(found: re.Match) -> str:
        try:
            when = deadline(found.group(2), now)
        except ValueError:
            return found.group(0)
        return found.group(1) + when.strftime("%Y-%m-%d %H:%M")

    return _BARE.sub(full, text)


def sample(sections: Dict[str, str]) -> Optional[str]:
    """The Brief's ``Sample:`` line, None when there is none."""
    for line in sections.get("Brief", "").splitlines():
        found = _SAMPLE.match(line)
        if found and found.group(1).strip("*_ "):
            return found.group(1).strip("*_ ")
    return None


class PlanStore:
    """The session's plan, in memory and beside the open experiment."""

    def __init__(
        self,
        experiment_path: Callable[[], Optional[str]],
        experiment_sample: Optional[Callable[[], Optional[str]]] = None,
        set_experiment_sample: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._experiment_path = experiment_path
        # The open experiment's sample (its metadata), read and written with
        # the Brief's: a sample named when the experiment was made is the
        # session's from the start.
        self._experiment_sample = experiment_sample or (lambda: None)
        self._set_experiment_sample = set_experiment_sample or (lambda sample: False)
        self.session_id: Optional[str] = None
        self.sections: Dict[str, str] = {}
        self.updated: Optional[str] = None
        self.updated_by: Optional[str] = None

    def _folder(self) -> Optional[Path]:
        path = self._experiment_path()
        if not path or not os.path.isdir(path):
            return None
        return Path(path) / FOLDER

    def _file(self, session_id: str) -> Optional[Path]:
        folder = self._folder()
        return folder / f"plan-{session_id}.md" if folder is not None else None

    def current(self, session_id: Optional[str]) -> Dict[str, str]:
        """The plan of *session_id*: what is held, else its file in the open
        experiment (a restarted app, a resumed session)."""
        if session_id != self.session_id:
            self.session_id, self.sections = session_id, {}
            self.updated = self.updated_by = None
            path = self._file(session_id) if session_id else None
            if path is not None and path.is_file():
                try:
                    self.sections = parse(path.read_text(encoding="utf-8"))
                    self.updated = (
                        datetime.fromtimestamp(path.stat().st_mtime)
                        .astimezone()
                        .isoformat(timespec="seconds")
                    )
                except (OSError, ValueError, HTTPException):
                    logging.warning("could not read the plan %s", path, exc_info=True)
        return self.sections

    def write(self, session_id: str, sections: Dict[str, str], by: str) -> None:
        self.session_id, self.sections = session_id, sections
        self.updated = datetime.now().astimezone().isoformat(timespec="seconds")
        self.updated_by = by
        path = self._file(session_id)
        if path is None:
            return
        try:
            path.parent.mkdir(exist_ok=True)
            partial = path.with_name(path.name + ".partial")
            partial.write_text(render(sections), encoding="utf-8")
            os.replace(partial, path)
        except OSError:
            logging.warning("could not write the plan %s", path, exc_info=True)

    def answer(self) -> Dict[str, Any]:
        sections = self.sections
        found = steps(sections.get("Steps", ""))
        return {
            "session_id": self.session_id,
            "markdown": render(sections) if sections else "",
            "sections": [{"name": n, "text": t} for n, t in sections.items()],
            "sample": sample(sections) or self.experiment_sample(),
            "steps": found,
            "done": sum(1 for s in found if s["done"]),
            "updated": self.updated,
            "updated_by": self.updated_by,
        }

    def experiment_sample(self) -> Optional[str]:
        try:
            return self._experiment_sample() or None
        except Exception:  # noqa: BLE001 - the plan still reads without it
            logging.debug("could not read the experiment's sample", exc_info=True)
            return None

    def set_experiment_sample(self, value: str) -> None:
        if value == self.experiment_sample():
            return
        try:
            self._set_experiment_sample(value)
        except Exception:  # noqa: BLE001 - the plan keeps it all the same
            logging.warning("could not set the experiment's sample", exc_info=True)

    def deadlines(self, session_id: Optional[str]) -> Optional[List[Dict[str, str]]]:
        """The Rules that give a time, each {text, by}. None with no plan."""
        sections = self.current(session_id)
        if not sections:
            return None
        found = []
        for rule in rules(sections.get("Rules", "")):
            by = rule_time(rule)
            if by:
                found.append({"text": rule, "by": by})
        return found

    def checklist(self, session_id: Optional[str]) -> Optional[List[Dict[str, str]]]:
        """The session's checklist read from its plan: the Rules standing, the
        Steps' boxes open or done. None with no plan."""
        sections = self.current(session_id)
        if not sections:
            return None
        items = [
            {"text": r, "state": "always"} for r in rules(sections.get("Rules", ""))
        ]
        items += [
            {"text": s["text"], "state": "done" if s["done"] else "open"}
            for s in steps(sections.get("Steps", ""))
        ]
        return items


def _session_id(auth: AuthConfig) -> str:
    session = auth.agent_session or {}
    found = session.get("session_id")
    if session.get("ended") or not isinstance(found, str) or not _SESSION.match(found):
        raise refusal_error(
            409,
            "no_session",
            "No agent session is running: the plan is the running session's.",
        )
    return found


def build_agent_plan_router(
    experiment_path: Callable[[], Optional[str]],
    emit: Optional[Callable[..., None]] = None,
) -> APIRouter:
    """``/app/agent/plan``: the agent writes it a section at a time, the
    operator edits it whole, both read it. *emit* puts an event on the record."""
    router = APIRouter(
        prefix="/app/agent", dependencies=[Depends(require_level(AgentLevel.WATCH))]
    )

    def _store(request: Request) -> PlanStore:
        store = getattr(request.app.state, "agent_plan", None)
        if store is None:
            store = request.app.state.agent_plan = PlanStore(experiment_path)
        return store

    def _record(kind: str, payload: Dict[str, Any], actor: str) -> None:
        if emit is None:
            return
        try:
            emit(kind, payload, actor=actor)
        except Exception:  # noqa: BLE001 - the record never costs the edit
            logging.debug("could not record %s", kind, exc_info=True)

    def _save(request: Request, sections: Dict[str, str]) -> Dict[str, Any]:
        auth: AuthConfig = request.app.state.auth
        by = (
            "operator"
            if is_operator_token(auth, request.headers.get("Authorization"))
            else "agent"
        )
        session_id = _session_id(auth)
        store = _store(request)
        before = store.current(session_id)
        if len(sections) > MAX_SECTIONS:
            raise _invalid(f"A plan has at most {MAX_SECTIONS} sections.")
        if "Rules" in sections:
            sections = {**sections, "Rules": dated(sections["Rules"])}
        was = sample(before) or store.experiment_sample()
        now = sample(sections)
        brief = sections.get("Brief") != before.get("Brief")
        if by == "agent" and brief and was and now != was:
            raise refusal_error(
                409,
                "sample_set",
                f"The sample is {was}, named once: only the operator changes it. "
                f"Keep the Brief's first line as 'Sample: {was}'.",
            )
        changed = [
            n for n in {**before, **sections} if before.get(n) != sections.get(n)
        ]
        store.write(session_id, sections, by)
        if now:
            store.set_experiment_sample(now)
        if now and now != was:
            _record(
                SAMPLE_SET, {"sample": now, "was": was, "session_id": session_id}, by
            )
        if by == "operator" and changed:
            _record(PLAN_EDITED, {"sections": changed, "session_id": session_id}, by)
        return store.answer()

    @router.get("/plan")
    def get_plan(request: Request):
        auth: AuthConfig = request.app.state.auth
        session = auth.agent_session or {}
        store = _store(request)
        store.current(session.get("session_id"))
        return store.answer()

    @router.post("/plan/section")
    def set_section(request: Request, body: Dict[str, Any]):
        name, text = _name(body.get("section")), _text(body.get("text"))
        store = _store(request)
        sections = dict(store.current(_session_id(request.app.state.auth)))
        if text:
            sections[name] = text
        else:
            sections.pop(name, None)
        return _save(request, sections)

    @router.put("/plan")
    def put_plan(request: Request, body: Dict[str, Any]):
        markdown = body.get("markdown")
        if not isinstance(markdown, str) or len(markdown) > MAX_SECTION * MAX_SECTIONS:
            raise _invalid(
                "markdown is the whole plan, its sections under ## headings."
            )
        sections = {n: _text(t) for n, t in parse(markdown).items()}
        return _save(request, {n: t for n, t in sections.items() if t})

    return router

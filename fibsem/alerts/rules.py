"""What is worth telling someone who is not at the instrument.

The same rule as the web app's notices: something needs an answer, or
something failed. An answer waits `wait_minutes` first, so someone at the PC
answers it before anyone is told, and it is dropped if answered meanwhile.
The agent's messages and what failed are told at once; a finished run only
when asked for.

Read from the session's record, the same records the agent card's rows come
from (``fibsem.ai.agent.recent``), so the card, the web notices and an alert
agree about what the agent said and what failed. What waits for an answer is
read from the session as it is now: the agent's request, its suggestion and
an agent asking to connect, and the run held on a decision.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.ai.agent.drawing import evidence_png
from fibsem.ai.agent.recent import ACTION, MESSAGE, RecentActions
from fibsem.workflows import evidence

# What an alert is about, and the setting that turns it on.
ANSWER, AGENT_MESSAGE, FAILED, FINISHED = "answer", "message", "failed", "finished"

# Where in the web app each is answered or shown (a page under its /ui).
AGENT_PAGE, EXPERIMENT_PAGE = "agent", "experiment"


@dataclass
class Alert:
    """One message for someone away: a headline, a line or two, where it
    happened, and the picture it is about."""

    key: str
    kind: str
    headline: str
    lines: List[str] = field(default_factory=list)
    context: str = ""
    page: str = AGENT_PAGE
    # Draws the picture when one is sent: images are off by default, and
    # drawing reads an image from disk.
    image: Optional[Callable[[], Optional[bytes]]] = None


def _words(name: Any) -> str:
    return str(name or "").replace("_", " ")


def _row_key(row: Any) -> str:
    # As the web app keys a row: what it is, when, and what it says.
    return f"{row.kind}|{row.at}|{row.text}"


class AlertRules:
    """Turns the session's records, and what waits in it, into alerts.

    ``feed`` takes each record of the session's stream; ``due`` says what to
    send now. Records fed with ``news=False`` are history: they set the rows
    up, and nothing in them is sent.
    """

    def __init__(self, session: Any, clock: Callable[[], float] = time.monotonic):
        self._session = session
        self._clock = clock
        self._rows = RecentActions()
        self._told: set = set()
        self._ready: List[Alert] = []
        # What waits for an answer: its key, to (first seen, its alert).
        self._waiting: Dict[str, Tuple[float, Alert]] = {}
        # The run held on a decision: (item, task) to the alert, while held.
        self._held: Dict[Tuple[str, str], Alert] = {}

    # --- the record ------------------------------------------------------------

    def feed(self, record: Dict[str, Any], news: bool = True) -> None:
        if self._rows.add(record):
            for row in self._rows.rows:
                key = _row_key(row) + (f"|{row.outcome}" if row.kind == ACTION else "")
                if key in self._told:
                    continue
                self._told.add(key)
                alert = self._row_alert(row, key)
                if news and alert is not None:
                    self._ready.append(alert)
        kind = record.get("kind")
        p = record.get("payload") if isinstance(record.get("payload"), dict) else {}
        if kind == "proposal_held":
            names = (str(p.get("item_name") or ""), str(p.get("task_name") or ""))
            self._held[names] = self._held_alert(*names)
        elif kind == "proposal_released":
            self._held.pop(
                (str(p.get("item_name") or ""), str(p.get("task_name") or "")), None
            )
        elif news and kind in ("workflow_failed", "workflow_stalled"):
            self._ready.append(self._run_alert(kind, p, record))
        elif news and kind == "workflow_completed":
            self._ready.append(self._run_alert(kind, p, record))

    def _row_alert(self, row: Any, key: str) -> Optional[Alert]:
        evidence = (row.evidence or {}).get("evidence_id")
        if row.kind == MESSAGE:
            return Alert(
                key,
                AGENT_MESSAGE,
                "Message from the agent",
                [row.text],
                self._context(row.level),
                image=self._evidence_image(evidence),
            )
        if row.kind == ACTION and row.outcome == "failed":
            return Alert(
                key,
                FAILED,
                f"The agent's action failed: {row.text}",
                [row.why] if row.why else [],
                self._context(row.level),
                image=self._evidence_image(evidence),
            )
        return None

    def _run_alert(self, kind: str, p: Dict[str, Any], record: Dict[str, Any]) -> Alert:
        name = p.get("experiment_name") or self._experiment_name()
        key = f"{kind}|{record.get('seq')}|{record.get('t')}"
        counts = p.get("task_counts") or {}
        done = ", ".join(f"{n} {_words(s)}" for s, n in counts.items() if n)
        if kind == "workflow_failed":
            lines = [str(p.get("error") or "")]
            remaining = p.get("tasks_remaining")
            if remaining:
                lines.append(
                    f"{remaining} tasks won't run until the run is started again."
                )
            return Alert(
                key,
                FAILED,
                "The run stopped with an error",
                lines,
                name,
                EXPERIMENT_PAGE,
            )
        if kind == "workflow_stalled":
            return Alert(
                key,
                FAILED,
                "The run stalled: decisions are waiting",
                ["Nothing failed, but the rest of the run waits on a decision."],
                name,
                EXPERIMENT_PAGE,
            )
        return Alert(
            key,
            FINISHED,
            f"Run finished: {name}" if name else "Run finished",
            [done] if done else [],
            name,
            EXPERIMENT_PAGE,
        )

    def _held_alert(self, item: str, task: str) -> Alert:
        what = " on ".join(x for x in (_words(task), item) if x)
        return Alert(
            f"held|{item}|{task}",
            ANSWER,
            f"The run is waiting for an answer: {what}"
            if what
            else "The run is waiting for an answer",
            ["Nobody has answered yet. The run waits until someone does."],
            self._context(),
            EXPERIMENT_PAGE,
        )

    # --- what waits now ----------------------------------------------------------

    def asks(self) -> List[Alert]:
        """What waits for an answer now, as alerts: the agent's request, its
        suggestion, an agent asking to connect, and the run held on a
        decision."""
        agent = self._session.agent
        found: List[Alert] = list(self._held.values())
        request = agent.current()
        if request is not None and getattr(request, "status", "") == "pending":
            view = request.to_dict()
            actions = [
                a.get("what") or a.get("tool") for a in view.get("actions") or []
            ]
            found.append(
                Alert(
                    f"request|{view.get('request_id')}",
                    ANSWER,
                    "The agent asks: " + ", ".join(str(a) for a in actions if a),
                    [str(view.get("reason") or "")],
                    self._context(),
                    image=self._picture("request"),
                )
            )
        suggestion = agent.suggestion()
        if suggestion and not (request is not None and request.status == "pending"):
            found.append(
                Alert(
                    f"suggestion|{suggestion.get('proposal_id')}",
                    ANSWER,
                    "The agent suggests: "
                    + " · ".join(
                        str(suggestion[k])
                        for k in ("decision", "proposal")
                        if suggestion.get(k)
                    ),
                    [str(suggestion.get("reason") or "")],
                    self._context(),
                    image=self._picture("suggestion"),
                )
            )
        asking = agent.asking()
        if asking:
            found.append(
                Alert(
                    f"connect|{asking.get('id')}",
                    ANSWER,
                    "An agent asks to connect",
                    [
                        f"{asking.get('name')} ({asking.get('origin')}). Nothing is "
                        "served to it until someone allows it."
                    ],
                    self._context(),
                )
            )
        for alert in found:
            alert.lines = [line for line in alert.lines if line]
        return found

    def due(self, wait_s: float, asks: Optional[List[Alert]] = None) -> List[Alert]:
        """What to send now: everything that happened since the last call,
        and each ask that has waited ``wait_s``, once. ``asks`` is what
        waits now (`asks`); None keeps what was last seen."""
        now = self._clock()
        if asks is not None:
            seen = {alert.key: alert for alert in asks}
            for key in list(self._waiting):
                if key not in seen:
                    del self._waiting[key]  # answered, or gone
            for key, alert in seen.items():
                self._waiting.setdefault(key, (now, alert))
        out, self._ready = self._ready, []
        for key, (since, alert) in self._waiting.items():
            if now - since >= wait_s and key not in self._told:
                self._told.add(key)
                minutes = max(1, round((now - since) / 60))
                alert.lines = alert.lines + [f"Waiting {minutes} min."]
                out.append(alert)
        return out

    # --- where, and the pictures -----------------------------------------------

    def _experiment_name(self) -> str:
        experiment = getattr(self._session, "experiment", None)
        return str(getattr(experiment, "name", "") or "")

    def _context(self, level: str = "") -> str:
        parts = [level.capitalize() if level else "", self._experiment_name()]
        return " · ".join(p for p in parts if p)

    def _picture(self, kind: str) -> Callable[[], Optional[bytes]]:
        agent = self._session.agent

        def draw() -> Optional[bytes]:
            picture = agent.picture(kind)
            return picture.get("png") if picture else None

        return draw

    def _evidence_image(
        self, evidence_id: Any
    ) -> Optional[Callable[[], Optional[bytes]]]:
        if not evidence_id:
            return None
        experiment = getattr(self._session, "experiment", None)
        path = getattr(experiment, "path", None)

        def draw() -> Optional[bytes]:
            record = evidence.find(os.fspath(path) if path else None, evidence_id)
            return evidence_png(record) if record else None

        return draw

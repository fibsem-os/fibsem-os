"""Every agent session the Hub knows of, in one list.

The sessions that ran on this hub PC are read from their folders here
(``~/.fibsem/hub/sessions/``, and ``~/.fibsem/hub/free-sessions/`` for those
with no instrument). Those that ran on a microscope PC are read
from that instrument's gateway (``GET /gateway/agent/sessions``), side by
side, and an instrument that doesn't answer is named rather than waited for.
Each row says which instrument, where it ran, its status, who started it and
what it spent; the totals add the spend up by instrument and by day.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fibsem import records
from fibsem.ai.agent.manager import MANAGED, read_json
from fibsem.gateway.gateway import GatewayRefused
from fibsem.hub.hub import Hub

__all__ = ["HUB_PC", "MICROSCOPE_PC", "hub_pc_sessions", "list_sessions"]

HUB_PC = "hub"
MICROSCOPE_PC = "pc"
# What the totals call the sessions with no instrument.
NO_INSTRUMENT = "No instrument"
SESSIONS_SHOWN = 50
SESSIONS_MAX = 200
# One instrument's list: short, so one that is slow never holds the page.
READ_S = 4.0
# The statuses a session can be carried on from.
_STOPPED = (records.ENDED, records.LEFT, records.CRASHED)
DAYS = 14


def hub_pc_sessions(root: Path, limit: int, free: bool = False) -> List[Dict[str, Any]]:
    """The newest *limit* sessions that ran on this hub PC under *root*:
    each for its instrument, or with *free* those with no instrument."""
    rows = []
    for found in records.list_sessions(limit, root=root):
        managed = read_json(Path(found.folder) / MANAGED)
        about = managed.get("about") or {}
        instrument = managed.get("microscope")
        if not instrument:
            continue
        if free:
            instrument = None
        rows.append(
            {
                "instrument": instrument,
                "runs_on": HUB_PC,
                "session_id": found.session_id,
                "started": found.started_at or about.get("started"),
                "ended": found.reason if found.status in _STOPPED else None,
                "status": found.status,
                "status_reason": found.reason,
                "last_active": found.last_active_at,
                "playbook": found.playbook,
                "goal": found.goal or about.get("goal"),
                "who": about.get("who"),
                "spent_usd": found.spent_usd,
                "last_level": about.get("level_now") or about.get("level"),
                "last_budget_usd": about.get("budget_usd"),
                # What it was given beyond its folder (a data folder,
                # connectors, code mode and its sites), if anything.
                "settings": about.get("settings") or None,
                "output": managed.get("output"),
                "gateway": True,
            }
        )
    return rows


def _instrument_sessions(
    hub: Hub, name: str, limit: int
) -> Tuple[str, Optional[List[Dict[str, Any]]], Optional[str]]:
    try:
        answer = hub.ask(
            name, "GET", f"/gateway/agent/sessions?limit={limit}", timeout=READ_S
        )
    except GatewayRefused as refusal:
        return name, None, str(refusal)
    rows = answer.get("sessions") if isinstance(answer, dict) else None
    return (
        name,
        [
            {**row, "instrument": name, "runs_on": MICROSCOPE_PC}
            for row in rows or []
            if isinstance(row, dict)
        ],
        None,
    )


def _totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_instrument: Dict[str, float] = {}
    by_day: Dict[str, float] = {}
    for row in rows:
        spent = row.get("spent_usd")
        if not isinstance(spent, (int, float)) or isinstance(spent, bool):
            continue
        name = row["instrument"] or NO_INSTRUMENT
        by_instrument[name] = by_instrument.get(name, 0.0) + spent
        day = str(row.get("started") or "")[:10]
        if day:
            by_day[day] = by_day.get(day, 0.0) + spent
    return {
        "spent_usd": round(sum(by_instrument.values()), 2),
        "by_instrument": {
            name: round(spent, 2) for name, spent in sorted(by_instrument.items())
        },
        "by_day": [
            {"day": day, "spent_usd": round(by_day[day], 2)}
            for day in sorted(by_day, reverse=True)[:DAYS]
        ],
    }


def list_sessions(
    hub: Hub,
    root: Path,
    limit: int = SESSIONS_SHOWN,
    free_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """``{sessions, totals, unread, more}``: the newest *limit* sessions on
    this hub PC (each instrument's, and under *free_root* those with no
    instrument) and on each paired instrument, newest first; the spend of the
    sessions read, by instrument and by day; and the instruments whose list
    could not be read, each with why."""
    limit = max(1, min(int(limit), SESSIONS_MAX))
    rows = hub_pc_sessions(root, limit + 1)
    more = len(rows) > limit
    rows = rows[:limit]
    if free_root is not None:
        free = hub_pc_sessions(free_root, limit + 1, free=True)
        more = more or len(free) > limit
        rows += free[:limit]
    unread: List[Dict[str, str]] = []
    paired = [i.name for i in hub.instruments() if hub.token(i.name) is not None]
    if paired:
        with ThreadPoolExecutor(max_workers=min(8, len(paired))) as pool:
            for name, found, why in pool.map(
                lambda name: _instrument_sessions(hub, name, limit + 1), paired
            ):
                if found is None:
                    unread.append({"instrument": name, "why": why or ""})
                    continue
                more = more or len(found) > limit
                rows += found[:limit]
    rows.sort(key=lambda row: str(row.get("started") or ""), reverse=True)
    return {
        "sessions": rows,
        "totals": _totals(rows),
        "unread": unread,
        "more": more,
    }

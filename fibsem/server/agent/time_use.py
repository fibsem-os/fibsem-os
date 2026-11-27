"""Where a session's time went: the model, the microscope, reads and analysis,
subagents, and the time between turns.

Read from the session's transcript (`fibsem.server.agent.transcript`): each
step's line says its tool and when it started, its done line how long it took,
and each turn's line how long the turn took. So the split needs nothing new on
the record, and the agent can ask it for any window of its run, a phase or a
site, by the times on its own notes.

- **Model**: a turn's time not spent in a tool. The model thinking and
  writing; the harness's own work is a few seconds a turn at most.
- **Microscope**: tools that image, move, mill, run the FM or wait on the
  microscope's runs.
- **Reads and analysis**: every other tool: the record, crops and measures,
  skills, the knowledge base.
- **Subagents**: each kind's runs, wall clock, split the same way inside
  them; and each run on its own (what it was asked, its microscope time and
  the microscope tools it called), so the runs that used the microscope show.
- **Between turns**: the workflow running alone, the agent asleep until a
  wake-up, or waiting on the operator.

Tools can run side by side, so a turn's tool time is the time any tool was
running, and the categories can add up to more than it.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

# A tool on the microscope: it images, moves, mills, runs the FM, or waits on
# a run there.
_MICROSCOPE_PREFIXES = (
    "beam_",
    "stage_",
    "fm_",
    "alignment_",
    "milling_",
    "coating_",
    "item_move_",
    "grid_move_to",
)
_MICROSCOPE = frozenset(
    (
        "device_set_parameter",
        "device_call_command",
        "service_set_parameter",
        "service_call_command",
        "service_wait_for_run",
        "microscope_set_state",
        "grid_load",
        "grid_unload",
        "grid_run_inventory",
    )
)
# Those named like a microscope tool that only read or work it out.
_NOT_MICROSCOPE = frozenset(
    (
        "beam_estimate_overview",
        "fm_estimate_overview",
        "milling_estimate_time",
        "milling_estimate_spot_burn",
        "milling_preview_patterns",
        "milling_check_stages",
        "milling_list_options",
        "milling_get_state",
    )
)
SUBAGENT_TOOLS = ("Agent", "Task")
CATEGORIES = ("microscope", "reads", "subagents", "compaction")
# The tools listed by time taken.
TOP_TOOLS = 8
# The subagent runs listed one by one.
TOP_RUNS = 20

Interval = Tuple[datetime, datetime]


def category(tool: str) -> str:
    """Which of `CATEGORIES` *tool* (its short name) is."""
    if tool in SUBAGENT_TOOLS:
        return "subagents"
    if tool == "compaction":
        return "compaction"
    if tool in _NOT_MICROSCOPE or tool.startswith(("stage_get_", "fm_get_")):
        return "reads"
    if tool in _MICROSCOPE or tool.startswith(_MICROSCOPE_PREFIXES):
        return "microscope"
    return "reads"


def _tool(step: Dict[str, Any]) -> str:
    """The tool a step line called: its ``tool``, or, on a line from before
    lines carried it, read from its words."""
    if step.get("tool"):
        return str(step["tool"])
    if step.get("subagent") is not None:
        return "Agent"
    verb = step.get("verb")
    if verb == "Compacted":
        return "compaction"
    if verb == "Called":
        return (step.get("text") or "").split(" ", 1)[0] or "other"
    return {"Read": "Skill", "Ran": "Bash", "Searched": "Grep"}.get(verb, "other")


def _union(intervals: Iterable[Interval]) -> float:
    """Seconds covered by any of *intervals*."""
    total, end = 0.0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += (b - a).total_seconds()
            end = b
        elif b > end:
            total += (b - end).total_seconds()
            end = b
    return total


def _clip(interval: Interval, window: Interval) -> Optional[Interval]:
    a, b = max(interval[0], window[0]), min(interval[1], window[1])
    return (a, b) if b > a else None


def _seconds(interval: Interval) -> float:
    return (interval[1] - interval[0]).total_seconds()


def _split(steps: List[Dict[str, Any]]) -> Dict[str, float]:
    """Seconds by category of *steps* (each with its ``span``)."""
    found = dict.fromkeys(CATEGORIES, 0.0)
    for step in steps:
        found[category(step["name"])] += _seconds(step["span"])
    return found


def _rounded(values: Dict[str, float]) -> Dict[str, float]:
    return {key: round(value) for key, value in values.items()}


def summarise(
    entries: List[Dict[str, Any]],
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Where one session's time went between *since* and *until* (the first
    line and *now*, else the last line, when not given), from its transcript
    *entries* (each with its ``_at``, as `transcript` reads them), in
    seconds."""
    entries = sorted(entries, key=lambda e: e["_at"])
    if not entries:
        return {"seconds": 0, "turns": 0, "notes": ["Nothing on the transcript yet."]}
    start = since or entries[0]["_at"]
    end = until or now or entries[-1]["_at"]
    if end <= start:
        return {"seconds": 0, "turns": 0, "notes": ["The window is empty."]}
    window = (start, end)

    answers = {
        e["step"]: e
        for e in entries
        if e["kind"] in ("done", "failed") and isinstance(e.get("step"), int)
    }
    steps = []
    for line in entries:
        if line["kind"] != "step" or not isinstance(line.get("seq"), int):
            continue
        answer = answers.get(line["seq"])
        took = answer.get("took_s") if answer else None
        if took is None:
            # Still running, or ended unanswered: up to the window's end.
            finish = end if answer is None else answer["_at"]
        else:
            finish = line["_at"] + timedelta(seconds=float(took))
        span = _clip((line["_at"], max(finish, line["_at"])), window)
        if span is None:
            continue
        steps.append(
            {
                **line,
                "name": _tool(line),
                "span": span,
                "subagent": line.get("subagent"),
            }
        )
    lead = [s for s in steps if s.get("run") is None]

    turns = []
    for line in entries:
        if line["kind"] == "turn" and line.get("took_s") is not None:
            span = _clip(
                (line["_at"] - timedelta(seconds=float(line["took_s"])), line["_at"]),
                window,
            )
            if span is not None:
                turns.append(span)
    seconds = _seconds(window)
    tools = _split(lead)
    busy = _union(s["span"] for s in lead)
    answer: Dict[str, Any] = {
        "since": start.isoformat(),
        "until": end.isoformat(),
        "seconds": round(seconds),
        "turns": len(turns),
        "lead": {
            "tools_s": round(busy),
            **_rounded({f"{k}_s": v for k, v in tools.items()}),
        },
        "notes": [],
    }
    if turns:
        in_turns = _union(turns)
        answer["in_turns_s"] = round(in_turns)
        answer["between_turns_s"] = round(max(seconds - in_turns, 0.0))
        answer["lead"]["model_s"] = round(max(in_turns - busy, 0.0))
    else:
        answer["notes"].append(
            "No turn lengths on this transcript (a session from before they "
            "were kept): the model's time and the time between turns are not known."
        )

    runs: Dict[str, Dict[str, float]] = {}
    each: List[Dict[str, Any]] = []
    for call in (s for s in lead if s["name"] in SUBAGENT_TOOLS):
        own = [s for s in steps if s.get("run") == call["seq"]]
        wall = _seconds(call["span"])
        inside = _union(s["span"] for s in own)
        kept = runs.setdefault(
            call.get("subagent") or "subagent",
            {
                "runs": 0,
                "wall_s": 0.0,
                "model_s": 0.0,
                **dict.fromkeys(("microscope_s", "reads_s"), 0.0),
            },
        )
        split = _split(own)
        kept["runs"] += 1
        kept["wall_s"] += wall
        kept["model_s"] += max(wall - inside, 0.0)
        kept["microscope_s"] += split["microscope"]
        kept["reads_s"] += split["reads"]
        used = sorted({s["name"] for s in own if category(s["name"]) == "microscope"})
        each.append(
            {
                "name": call.get("subagent") or "subagent",
                "about": (call.get("text") or "")[:120],
                "started": call["span"][0].isoformat(),
                "wall_s": round(wall),
                "microscope_s": round(split["microscope"]),
                "microscope_tools": used,
            }
        )
    answer["subagents"] = [
        {"name": name, **_rounded(values)}
        for name, values in sorted(runs.items(), key=lambda kv: -kv[1]["wall_s"])
    ]
    # Each run on its own, the ones that used the microscope most first: which
    # did the microscope work, and what it was asked.
    answer["subagent_runs"] = sorted(
        each, key=lambda r: (-r["microscope_s"], -r["wall_s"])
    )[:TOP_RUNS]

    by_tool: Dict[str, List[float]] = {}
    for step in steps:
        if step["name"] in SUBAGENT_TOOLS:
            continue
        by_tool.setdefault(step["name"], []).append(_seconds(step["span"]))
    slowest = sorted(by_tool.items(), key=lambda kv: -sum(kv[1]))[:TOP_TOOLS]
    answer["slowest_tools"] = [
        {
            "tool": name,
            "calls": len(times),
            "total_s": round(sum(times)),
            "mean_s": round(sum(times) / len(times), 1),
        }
        for name, times in slowest
    ]
    if sum(tools.values()) > busy + 1:
        answer["notes"].append(
            "Some tools ran side by side: the categories add up to more than tools_s."
        )
    return answer

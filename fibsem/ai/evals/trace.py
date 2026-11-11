"""What the agent did in each repeat of a run folder, from its log: the
skills it read, what it looked at, the tools it called, what it said and its
answer. ``python -m fibsem.ai.evals trace <out>`` prints it, to see why a
repeat missed or which skill it never read."""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fibsem.ai.evals import visual

AGENT_LOG = "agent.log"
TRANSCRIPT = "session/transcript.jsonl"
_STAMP = re.compile(r"^\d\d:\d\d:\d\d (.*)$")
_SKILL = re.compile(r"^Read the (\S+) skill$")
_READ = re.compile(r"^Read (\S+)$")
_CALL = re.compile(r"^Called (\S+)(?: (.*))?$")
_PATH = re.compile(r'"(?:path|name|image)": "([^"]+)"')
# A long answer or message is cut to this many characters.
CUT = 600


@dataclass
class Trace:
    skills: List[str] = field(default_factory=list)
    looked_at: List[str] = field(default_factory=list)
    tools: Counter = field(default_factory=Counter)
    said: List[str] = field(default_factory=list)
    failed: List[str] = field(default_factory=list)


def log_lines(text: str) -> List[str]:
    """The log's entries without their times; a line with no time carries on
    the entry before it."""
    entries: List[str] = []
    for line in text.splitlines():
        stamped = _STAMP.match(line)
        if stamped:
            entries.append(stamped.group(1))
        elif entries and line.strip():
            entries[-1] += "\n" + line
    return entries


def read_log(text: str) -> Trace:
    trace = Trace()
    for entry in log_lines(text):
        skill, read, call = _SKILL.match(entry), _READ.match(entry), _CALL.match(entry)
        if skill:
            trace.skills.append(skill.group(1))
        elif read:
            trace.looked_at.append(read.group(1))
        elif call:
            name = call.group(1).removeprefix("mcp__fibsem__")
            trace.tools[name] += 1
            if name.startswith(("experiment_read", "item_get_output_image")):
                found = _PATH.search(call.group(2) or "")
                if found:
                    trace.looked_at.append(found.group(1))
        elif entry.startswith("agent: "):
            trace.said.append(entry[len("agent: ") :])
        elif entry.startswith("failed: "):
            trace.failed.append(entry[len("failed: ") :])
    return trace


def transcript_text(path: Path) -> List[str]:
    """The agent's own words in the session's transcript, in full, when the
    run kept it."""
    said = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        message = entry.get("message") if isinstance(entry, dict) else None
        if entry.get("type") != "assistant" or not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                if block.get("text", "").strip():
                    said.append(block["text"].strip())
    return said


def read_run(folder: Path) -> Trace:
    log = folder / AGENT_LOG
    trace = (
        read_log(log.read_text(encoding="utf-8", errors="replace"))
        if log.is_file()
        else Trace()
    )
    if (folder / TRANSCRIPT).is_file():
        trace.said = transcript_text(folder / TRANSCRIPT) or trace.said
    return trace


def _cut(text: str) -> str:
    return text if len(text) <= CUT else text[:CUT] + " ..."


def _scores(result: Dict[str, Any]) -> str:
    parts = []
    for score in result.get("scores") or []:
        name = f"{score['part']}: " if score.get("part") else ""
        parts.append(name + score["result"])
    return ", ".join(parts) or "-"


def trace_markdown(out: Path, only: Optional[str] = None) -> str:
    """Each case's repeats side by side (result, skills read, tool calls,
    cost), then each repeat in full."""
    lines = [f"# What the agent did: {Path(out).name}", ""]
    for case_id, items in sorted(visual.load_runs(out).items()):
        if only and only not in case_id:
            continue
        traces = [(folder, run, read_run(folder)) for folder, run in items]
        lines += [
            f"## {case_id}",
            "",
            "| Repeat | Result | Skills read | Tool calls | Looked at | Cost |",
            "|---|---|---|---|---|---|",
        ]
        for _, (result, _), trace in traces:
            lines.append(
                f"| r{result.get('repeat')} | {_scores(result)} | "
                f"{', '.join(trace.skills) or 'none'} | {sum(trace.tools.values())} | "
                f"{len(trace.looked_at)} | ${float(result.get('spent_usd') or 0):.2f} |"
            )
        lines.append("")
        for _, (result, answer), trace in traces:
            lines += [f"### r{result.get('repeat')}: {_scores(result)}", ""]
            for score in result.get("scores") or []:
                if score.get("detail"):
                    lines.append(
                        f"- Score: {score.get('part') or ''} {score['detail']}"
                    )
            if trace.looked_at:
                lines.append(f"- Looked at: {', '.join(trace.looked_at)}")
            if trace.tools:
                calls = ", ".join(f"{n} x{c}" for n, c in trace.tools.most_common())
                lines.append(f"- Tools: {calls}")
            for failure in trace.failed:
                lines.append(f"- Refused: {_cut(failure)}")
            lines.append("")
            for words in trace.said:
                lines += ["> " + line for line in _cut(words).splitlines()] + [""]
            text = json.dumps(answer) if answer is not None else "none"
            lines += [f"Answer: `{_cut(text)}`", ""]
    return "\n".join(lines)

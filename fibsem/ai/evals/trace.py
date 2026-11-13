"""What the agent did in each repeat of a run folder: the skills it read, what
it looked at, the tools it called, what it said and its answer. Read from the
session's transcript (``session/transcript.jsonl``, Claude Code's own record of
each message and tool call), or from the agent's log for a run that kept none.
``python -m fibsem.ai.evals trace <out>`` prints it, to see why a repeat missed
or which skill it never read."""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from fibsem.ai.evals import visual

AGENT_LOG = "agent.log"
TRANSCRIPT = "session/transcript.jsonl"
# The agent's folder in a run: what it reads there is named from it.
WORK = "work"
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


def _take(trace: Trace, entry: str) -> None:
    """One step as the harness says it ("Read the x skill", "Called y {...}")."""
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


def read_log(text: str) -> Trace:
    """From the agent's log: each step's line, matched by its words."""
    trace = Trace()
    for entry in log_lines(text):
        if entry.startswith("agent: "):
            trace.said.append(entry[len("agent: ") :])
        elif entry.startswith("failed: "):
            trace.failed.append(entry[len("failed: ") :])
        else:
            _take(trace, entry)
    return trace


def _messages(path: Path):
    """The transcript's main-session messages, (type, content blocks, the
    folder the agent ran in)."""
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("isSidechain"):
            continue
        message = entry.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            cwd = entry.get("cwd")
            blocks = [b for b in content if isinstance(b, dict)]
            yield entry.get("type"), blocks, Path(cwd) if isinstance(cwd, str) else None


def read_transcript(path: Path, cwd: Optional[Path] = None) -> Trace:
    """From the session's transcript: each tool call by its name and arguments,
    put in the harness's words (`harness.step`), so a run reads the same from
    either source; a tool result marked as an error is a failure. A file read
    in the agent's folder is named from it: the folder each line says it ran in,
    else *cwd*."""
    from fibsem.ai.agent.harness import _clip, _result_text, step

    trace = Trace()
    for kind, blocks, ran_in in _messages(path):
        for block in blocks:
            if kind == "assistant" and block.get("type") == "text":
                if str(block.get("text", "")).strip():
                    trace.said.append(block["text"].strip())
            elif kind == "assistant" and block.get("type") == "tool_use":
                name, arguments = str(block.get("name", "")), block.get("input")
                verb, text = step(name, arguments, ran_in or cwd)
                _take(trace, f"{verb} {text}")
            elif kind == "user" and block.get("type") == "tool_result":
                if block.get("is_error"):
                    trace.failed.append(_clip(_result_text(block.get("content"))))
    return trace


def read_run(folder: Path) -> Trace:
    """A repeat's trace: from its transcript when it kept one, else its log."""
    if (folder / TRANSCRIPT).is_file():
        work = folder / WORK
        return read_transcript(folder / TRANSCRIPT, work if work.is_dir() else None)
    log = folder / AGENT_LOG
    if log.is_file():
        return read_log(log.read_text(encoding="utf-8", errors="replace"))
    return Trace()


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

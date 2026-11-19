"""The segmenter: an agent, run offline on an episode, that proposes its
segments (``segments.yaml``) for a person to review.

It reads the episode's record (events, the agent's transcripts, images) and
the case catalogue, and writes ``segments.yaml``; :func:`segments.check`
then checks it against the record, and the agent gets one turn to fix what
the check found. It only reads the episode and writes that one file: it
never reaches a microscope, and nothing it writes goes back into a run.
"""

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from fibsem.ai.evals.core import catalogue, segments
from fibsem.ai.evals.core.episode import Episode
from fibsem.util import timestamps

log = logging.getLogger(__name__)

FOLDER = "segmenter"
EVENTS_DIGEST = "events.txt"
CATALOGUE = "catalogue.yaml"
PROMPT = "prompt.md"
TOOLS = ("Read", "Glob", "Grep", "Write")
PAYLOAD_CHARS = 240

INSTRUCTIONS = """\
You are segmenting the record of one run of a microscope agent (an
"episode") into the decisions an eval case could score. You work offline
on files: nothing you do reaches a microscope.

The episode folder is your working folder:
- `episode.yaml`: where the run came from, and its agent sessions.
- `experiment/<name>/events.jsonl`: every event of the run, in time order.
  `segmenter/events.txt` is the same, one line each: id, time, actor, kind,
  item, and the start of the payload. Read that first; open events.jsonl
  for a payload in full.
- `sessions/<id>/transcript.jsonl`: what each agent session saw and said.
  It is long; search it (Grep) for the times and items you need.
- The experiment's images and files, under `experiment/<name>/`.
- `segmenter/catalogue.yaml`: the case types (tasks), what moment in a run
  each scores (`decision`), the inputs a case needs, and its answer shape.

Write `segments.yaml` in the episode folder, in this form:

```yaml
format: 0
episode: {episode}
segments:
- id: seg-01
  task: setup_placement        # a task from the catalogue, or null
  new_task:                    # only when task is null
    name: milling_current
    question: Which current for the next polishing pass, and why?
    answer_shape: choice       # one of the catalogue's no_task_yet shapes
  item: Lam-01                 # the lamella or site it is about, or null
  start_event: <event id>      # the cut: the case shows only what came before
  end_event: <event id>        # where the decision was acted on
  inputs:                      # each input the task names, as a path
    fib_image: Lam-01/ref_setup_ib.tif   # relative to experiment/<name>/
  decision: what the run decided, in a sentence
  facts:                       # only what the record states, when the task
    milling_angle_deg: 15      # needs it: target_nm, milling_angle_deg,
    target_nm: 200             # lamella_um (width, spacing)
  answer: {{...}}              # the run's decision in the task's answer shape,
                               # when you can say it; else leave it out
  override:                    # only when the operator changed or refused it
    what: what the operator did
    said: their words, quoted from the record
  reasons: why this is a decision worth a case, in a sentence
  status: proposed
```

How to segment:
1. One segment per decision a person could judge right or wrong from what
   was seen at the time: a placement, a stop, a site picked, a target found.
   Not every action is a decision; skip routine moves and images.
2. A task's decision text says which moment it scores. Use the task whose
   decision matches; one segment per lamella for a task decided pass by
   pass (its inputs list the passes).
3. start_event is the last event before the agent began working the
   decision out; every input must have been saved before it. Pick files
   the record names (an event's payload path), so the check can confirm it.
4. A decision worth scoring that no task covers gets task: null and a
   new_task. Name it plainly; reuse a name you already gave.
5. Overrides matter most: an operator edit, a declined suggestion or a
   message that corrects the agent. Quote what they said.
6. Write every segment with status: proposed. A person keeps or drops them.

When you have written segments.yaml, say how many segments you proposed,
and how many have no task yet. Do not write any other file.
"""


def _short(value: Any) -> str:
    text = json.dumps(value, default=str, separators=(",", ":"))
    return text if len(text) <= PAYLOAD_CHARS else text[: PAYLOAD_CHARS - 3] + "..."


def events_digest(events: List[Dict[str, Any]]) -> str:
    lines = []
    for event in events:
        lines.append(
            "\t".join(
                str(x)
                for x in (
                    event.get("id"),
                    event.get("t") or event.get("timestamp"),
                    event.get("actor"),
                    event.get("kind"),
                    event.get("item") or "-",
                    _short(event.get("payload") or {}),
                )
            )
        )
    return "\n".join(lines) + "\n"


def prepare(episode: Episode) -> Path:
    """The segmenter's folder in the episode: the events digest, the
    catalogue and the prompt, written fresh."""
    folder = episode.path / FOLDER
    folder.mkdir(exist_ok=True)
    (folder / EVENTS_DIGEST).write_text(events_digest(episode.events()))
    (folder / CATALOGUE).write_text(
        yaml.safe_dump(catalogue.catalogue(), sort_keys=False, width=100)
    )
    (folder / PROMPT).write_text(prompt(episode))
    return folder


def prompt(episode: Episode) -> str:
    return INSTRUCTIONS.format(episode=episode.name)


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _fix_text(problems: List[str]) -> str:
    listed = "\n".join(f"- {p}" for p in problems)
    return (
        "The check of segments.yaml against the record found these problems. "
        "Fix them in segments.yaml (drop a segment you cannot fix), then say "
        f"what you changed.\n\n{listed}"
    )


def _may_use(episode: Episode):
    """The agent's permission check for writing: segments.yaml only. Its
    read tools are allowed outright and work from the episode folder."""
    from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny

    allowed = (episode.path / segments.FILE).resolve()

    async def check(tool: str, data: Dict[str, Any], context: Any):
        target = data.get("file_path") or data.get("path")
        if tool == "Write":
            if target and Path(target).resolve() == allowed:
                return PermissionResultAllow()
            return PermissionResultDeny(message=f"Only {allowed} may be written.")
        return PermissionResultDeny(message=f"{tool} is not allowed here.")

    return check


async def _run(
    episode: Episode, model: Optional[str], budget_usd: float, effort: Optional[str]
) -> Dict[str, Any]:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        ResultMessage,
    )

    text = prompt(episode)
    options = ClaudeAgentOptions(
        cwd=str(episode.path),
        tools=list(TOOLS),
        allowed_tools=["Read", "Glob", "Grep"],
        setting_sources=[],
        strict_mcp_config=True,
        system_prompt={"type": "preset", "preset": "claude_code", "append": text},
        max_budget_usd=budget_usd,
        model=model,
        effort=effort,
        can_use_tool=_may_use(episode),
    )
    spent = 0.0
    said: List[str] = []

    async def turn(client, message: str) -> None:
        nonlocal spent
        await client.query(message)
        async for reply in client.receive_response():
            if isinstance(reply, ResultMessage):
                spent = float(reply.total_cost_usd or spent)
                if reply.result:
                    said.append(str(reply.result))

    async with ClaudeSDKClient(options) as client:
        await turn(client, "Segment this episode now, following your instructions.")
        result = segments.check(episode)
        if not result.ok:
            await turn(client, _fix_text(result.problems))
    return {
        "model": model or "default",
        "effort": effort,
        "prompt_hash": prompt_hash(text),
        "at": timestamps.now().isoformat(),
        "spent_usd": round(spent, 4),
        "said": said[-1] if said else None,
    }


def run(
    episode: Episode,
    model: Optional[str] = None,
    budget_usd: float = 5.0,
    effort: Optional[str] = None,
) -> segments.Check:
    """Propose the episode's segments with an agent; the check of what it
    wrote. A segments.yaml already there is kept as segments.previous.yaml."""
    prepare(episode)
    existing = segments.path(episode)
    if existing.is_file():
        existing.replace(episode.path / "segments.previous.yaml")
    info = asyncio.run(_run(episode, model, budget_usd, effort))
    data = segments.load(episode)
    if data:
        data["segmenter"] = info
        segments.save(episode, data)
    return segments.check(episode)

"""Subagents: what the agent calls for a second opinion or a review, each in a
fresh context with its own tools.

A subagent is ``<name>.md``: front matter (``name``, ``description``, ``tools``,
``model``, ``effort``, ``skills``) then its instructions, the format of a
checkout's ``.claude/agents/``. fibsemOS ships some there; the lab's own sit in
the knowledge base (``~/.fibsem/knowledge/agents/``) and win by name. The
agent can call every one of them; a playbook names the ones its work uses.

A subagent gets the tools its file lists, and only those the session also has:
the harness still denies the file and shell tools outside code mode, and the
server's level gate and blocklist still decide what a fibsem tool may do. A
lab's extra MCP servers (`extra_mcp`) add the tools their ``subagents`` name.
"""

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from fibsem.ai.agent import code_mode, extra_mcp, sources
from fibsem.ai.agent.catalog import CATALOG

SERVER = "fibsem"
# Front matter a subagent may set; the CLI reads the same names.
FIELDS = ("name", "description", "tools", "model", "effort", "skills")
# The CLI's tools a subagent may list besides the fibsem ones; the session's
# own allow and deny lists still decide whether it gets them.
BUILT_IN = tuple(code_mode.CODE_TOOLS) + ("Skill", "ToolSearch")


# The general worker (`.claude/agents/fibsem-worker.md`): the lead hands it
# one job on one site or lamella. It has every fibsem tool but `LEAD_ONLY`.
WORKER = "fibsem-worker"
# The tools that are the lead's job, not a worker's: the record (the plan,
# notes, the review list, the session's report, the knowledge base), the
# operator (messages and requests), the run as a whole (the workflow, the
# protocol, the experiment, grids, coating, every item at once) and
# suggesting at lower levels. A worker may decide a proposal on its own
# item when its brief says so. Every catalog tool is either the worker's or
# here (`tests/agent/test_agent_subagents.py`), so a new one is a choice.
LEAD_ONLY = frozenset(
    {
        "server_report_session",
        "plan_set_section",
        "plan_append",
        "plan_check_step",
        "review_add",
        "review_close",
        "experiment_add_note",
        "experiment_message_operator",
        "request_create",
        "request_withdraw",
        "knowledge_write",
        "proposal_suggest",
        "workflow_start",
        "workflow_stop",
        "workflow_stop_task",
        "workflow_plan",
        "workflow_plan_grids",
        "workflow_start_grids",
        "workflow_queue_grids",
        "workflow_requeue",
        "workflow_edit_queue",
        "workflow_set_options",
        "protocol_load",
        "protocol_save",
        "protocol_add_task",
        "protocol_remove_task",
        "protocol_reorder_tasks",
        "protocol_reorder_milling_stages",
        "protocol_set_schedule",
        "protocol_set_supervision",
        "protocol_set_task_requirements",
        "protocol_set_lamella_defaults",
        "protocol_update_task_config",
        "protocol_update_grid_task_config",
        "protocol_reorder_grid_tasks",
        "protocol_set_grid_task_requirements",
        "experiment_create",
        "experiment_load",
        "item_add",
        "item_remove",
        "item_set_objective_position_on_all",
        "grid_add",
        "grid_rename",
        "grid_remove",
        "grid_set_quality",
        "grid_set_note",
        "grid_assign_items",
        "grid_set_current",
        "grid_run_inventory",
        "grid_load",
        "grid_unload",
        "grid_move_to_slot",
        "coating_deposit",
        "coating_sputter",
    }
)
# A tools line's entry that lets a subagent start only the named ones.
_AGENT_LIMIT = re.compile(r"^Agent\(([\w-]+(?:\s*,\s*[\w-]+)*)\)$")


class SubagentError(ValueError):
    """A subagent's file that can't be used, with the reason in words."""


def known_tools(extras: Optional[Dict[str, extra_mcp.Server]] = None) -> List[str]:
    """Every tool a subagent may list: the fibsem tools, the CLI's, then those
    the lab's extra servers give specialists (*extras*, else read)."""
    if extras is None:
        try:
            extras = extra_mcp.load()
        except extra_mcp.ExtraMcpError:
            # fibsem-agent refuses to start on it, saying why.
            extras = {}
    fibsem = [f"mcp__{SERVER}__{spec.name}" for spec in CATALOG]
    return fibsem + list(BUILT_IN) + extra_mcp.subagent_allowed(extras)


def front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """A file's front matter and the text after it; None without front
    matter. The CLI reads it a line at a time, so a value with a colon in it,
    not quoted, still counts. Windows line endings read as Unix ones: a
    checkout on Windows writes the shipped files with them."""
    unix = text.replace("\r\n", "\n")
    front, sep, body = (
        unix[4:].partition("\n---") if unix.startswith("---\n") else ("", "", unix)
    )
    if not sep:
        return None, text
    body = body.split("\n", 1)[1] if "\n" in body else ""
    try:
        data = yaml.safe_load(front)
    except yaml.YAMLError:
        data = None
    if not isinstance(data, dict):
        found = re.findall(r"^([A-Za-z][\w-]*):[ \t]*(.*)$", front, re.MULTILINE)
        data = dict(found) if found else None
    return data, body


def names(value: Any) -> List[str]:
    """A list the CLI accepts as a list or as ``a, b, c``."""
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    raise SubagentError("tools and skills are a list of names")


def check(
    text: str,
    efforts: Iterable[str] = (),
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
) -> Dict[str, Any]:
    """The subagent's front matter, or a `SubagentError` saying what is wrong."""
    data, body = front_matter(text)
    if not data or not data.get("description"):
        raise SubagentError(
            "A subagent starts with front matter between '---' lines, "
            "with its name and description."
        )
    unknown = sorted(set(data) - set(FIELDS))
    if unknown:
        raise SubagentError(f"unknown setting(s) {', '.join(unknown)}")
    tools = names(data.get("tools"))
    known = set(known_tools(extras))
    # Agent(a, b): it may start those subagents, and no others.
    missing = [
        tool for tool in tools if tool not in known and not _AGENT_LIMIT.match(tool)
    ]
    if missing:
        raise SubagentError(f"no tool {', '.join(missing)}")
    efforts = list(efforts)
    effort = data.get("effort")
    if efforts and effort is not None and effort not in efforts:
        raise SubagentError(f"effort is one of {', '.join(efforts)}")
    if not body.strip():
        raise SubagentError("a subagent's instructions follow its front matter")
    return data


def check_file(
    name: str,
    root,
    efforts: Iterable[str] = (),
    extras: Optional[Dict[str, extra_mcp.Server]] = None,
) -> None:
    """Subagent *name*'s file, as the session would assemble it, checked."""
    path = sources.agent_file(name, root)
    if path is None:
        raise SubagentError(f"No subagent {name} in {sources.lab_folder('agents')}")
    try:
        check(path.read_text(encoding="utf-8"), efforts, extras)
    except SubagentError as error:
        raise SubagentError(f"{path}: {error}") from None


def render(data: Dict[str, Any], prompt: str) -> str:
    """A subagent's file from its settings and its instructions."""
    front = {k: data[k] for k in FIELDS if data.get(k) not in (None, "", [])}
    for key in ("tools", "skills"):
        if key in front:
            front[key] = ", ".join(names(front[key]))
    text = yaml.safe_dump(front, sort_keys=False, allow_unicode=True, width=10_000)
    return f"---\n{text}---\n\n{prompt.strip()}\n"


def add_tools(folder: Path, agents: Iterable[str], tools: Iterable[str]) -> None:
    """Put *tools* on the ``tools`` line of each of *agents* copied into
    ``folder/.claude/agents/``, after `sources.assemble` built it: only ever
    a session folder's copies, never the shipped files. A file with no line
    is left as it is, since no line means every tool."""
    tools = list(tools)
    if not tools:
        return
    for name in agents:
        path = folder / sources.CLAUDE / "agents" / f"{name}.md"
        data, body = front_matter(path.read_text(encoding="utf-8"))
        listed = names(data.get("tools")) if data else []
        more = [tool for tool in tools if tool not in listed]
        if listed and more:
            data["tools"] = listed + more
            path.write_text(render(data, body), encoding="utf-8")

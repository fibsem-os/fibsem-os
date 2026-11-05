"""Subagents: what the agent calls for a second opinion or a review, each in a
fresh context with its own tools.

A subagent is ``<name>.md``: front matter (``name``, ``description``, ``tools``,
``model``, ``effort``, ``skills``) then its instructions, the format of a
checkout's ``.claude/agents/``. fibsemOS ships some there; the lab's own sit in
the knowledge base (``~/.fibsem/knowledge/agents/``) and win by name. The
agent can call every one of them; a playbook names the ones its work uses.

A subagent gets the tools its file lists, and only those the session also has:
the harness still denies the file and shell tools outside code mode, and the
server's level gate and blocklist still decide what a fibsem tool may do.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from fibsem.ai.agent import code_mode, sources
from fibsem.ai.agent.catalog import CATALOG

SERVER = "fibsem"
# Front matter a subagent may set; the CLI reads the same names.
FIELDS = ("name", "description", "tools", "model", "effort", "skills")
# The CLI's tools a subagent may list besides the fibsem ones; the session's
# own allow and deny lists still decide whether it gets them.
BUILT_IN = tuple(code_mode.CODE_TOOLS) + ("Skill", "ToolSearch")


class SubagentError(ValueError):
    """A subagent's file that can't be used, with the reason in words."""


def known_tools() -> List[str]:
    """Every tool a subagent may list: the fibsem tools, then the CLI's."""
    return [f"mcp__{SERVER}__{spec.name}" for spec in CATALOG] + list(BUILT_IN)


def front_matter(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """A file's front matter and the text after it; None without front
    matter. The CLI reads it a line at a time, so a value with a colon in it,
    not quoted, still counts."""
    front, sep, body = (
        text[4:].partition("\n---") if text.startswith("---\n") else ("", "", text)
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


def check(text: str, efforts: Iterable[str] = ()) -> Dict[str, Any]:
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
    known = set(known_tools())
    missing = [tool for tool in tools if tool not in known]
    if missing:
        raise SubagentError(f"no tool {', '.join(missing)}")
    efforts = list(efforts)
    effort = data.get("effort")
    if efforts and effort is not None and effort not in efforts:
        raise SubagentError(f"effort is one of {', '.join(efforts)}")
    if not body.strip():
        raise SubagentError("a subagent's instructions follow its front matter")
    return data


def check_file(name: str, root, efforts: Iterable[str] = ()) -> None:
    """Subagent *name*'s file, as the session would assemble it, checked."""
    path = sources.agent_file(name, root)
    if path is None:
        raise SubagentError(f"No subagent {name} in {sources.lab_folder('agents')}")
    try:
        check(path.read_text(encoding="utf-8"), efforts)
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

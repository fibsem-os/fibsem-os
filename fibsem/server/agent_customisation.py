"""What the agent runs with, for the operator's Customisation dialog: the tools
this server offers, the skills and specialists ``fibsem-agent`` loads, and
what the agent may never touch. Read-only."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fibsem.ai.agent import blocklist, harness, sources
from fibsem.ai.agent.catalog import CATALOG, GROUP_TITLES, ToolSpec
from fibsem.server.responses import (
    AgentCustomisation,
    BlockRow,
    SkillRow,
    ToolRow,
)

# The dialog's filter: the catalog's groups (a tool name's first word) in fewer
# categories.
CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Microscope": ("server", "plan", "microscope", "device", "service", "stage"),
    "Imaging": ("beam", "fm", "correlation"),
    "Milling": ("milling", "alignment", "coating"),
    "Analysis": ("analysis",),
    "Experiment & protocol": ("app", "experiment", "protocol"),
    "Lamellae & grids": ("item", "grid"),
    "Workflow & approvals": ("workflow", "proposal", "request"),
    "Knowledge": ("knowledge",),
}
_CATEGORY_OF = {
    group: title for title, groups in CATEGORIES.items() for group in groups
}


def _tool(spec: ToolSpec) -> ToolRow:
    group = spec.name.split("_")[0]
    return ToolRow(
        name=spec.name,
        description=spec.description,
        level=spec.level,
        category=_CATEGORY_OF[group],
        group=GROUP_TITLES[group],
        hardware=spec.hardware,
    )


def description(path: Path) -> str:
    """The ``description`` line in a skill's or agent's front matter, or "".
    Read as a line, not YAML: a description often has a colon in it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    for line in text.split("---", 2)[1].splitlines():
        if line.startswith("description:"):
            return line[len("description:") :].strip().strip("\"'")
    return ""


def _skills(names: Iterable[str], find) -> List[SkillRow]:
    rows = []
    for name in names:
        found = find(name)
        rows.append(SkillRow(name, description(found) if found else ""))
    return rows


def customisation(tools: Iterable[ToolSpec] = CATALOG) -> AgentCustomisation:
    """What the agent runs with: *tools* are the ones this server offers."""
    checkout = harness.find_checkout()

    def skill(name: str) -> Optional[Path]:
        found = sources.skill_dir(name, checkout)
        return found / sources.SKILL_FILE if found else None

    return AgentCustomisation(
        tools=[_tool(spec) for spec in tools],
        categories=list(CATEGORIES),
        skills=_skills(harness.skills_for(), skill),
        specialists=_skills(
            harness.subagents_for(), lambda name: sources.agent_file(name, checkout)
        ),
        never=[BlockRow(**row) for row in blocklist.rules()],
    )

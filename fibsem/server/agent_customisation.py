"""What the agent runs with, for the operator's Customisation dialog: the tools
this server offers, the skills and specialists ``fibsem-agent`` loads, the
profiles on this PC and what the agent may never touch. Read-only."""

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from fibsem.ai.agent import blocklist, harness, profiles
from fibsem.ai.agent.catalog import CATALOG, GROUP_TITLES, ToolSpec
from fibsem.server.responses import (
    AgentCustomisation,
    BlockRow,
    ProfileRow,
    ReminderRow,
    SkillRow,
    ToolRow,
)

# The dialog's filter: the catalog's groups (a tool name's first word) in fewer
# categories.
CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "Microscope": ("server", "microscope", "device", "service", "stage"),
    "Imaging": ("beam", "fm", "correlation"),
    "Milling": ("milling", "alignment", "coating"),
    "Analysis": ("analysis",),
    "Experiment & protocol": ("app", "experiment", "protocol"),
    "Lamellae & grids": ("item", "grid"),
    "Workflow & approvals": ("workflow", "proposal", "request"),
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


def _skills(names: Iterable[str], folder: Path, file: str) -> List[SkillRow]:
    return [
        SkillRow(name, description(folder / file.format(name=name))) for name in names
    ]


def _profile(path: Path, checkout: Optional[Path]) -> ProfileRow:
    try:
        found = profiles.load(path, checkout=checkout)
    except profiles.ProfileError as error:
        return ProfileRow(name=path.stem, path=str(path), hash="", error=str(error))
    options = profiles.HOOK_OPTIONS
    return ProfileRow(
        name=found.name,
        path=str(found.path),
        hash=found.hash,
        model=found.model,
        effort=found.effort,
        max_budget_usd=found.max_budget_usd,
        max_turns=found.max_turns,
        mode=found.mode,
        skills=list(found.skills),
        prompt=found.prompt,
        reminders=[
            ReminderRow(
                options[hook.when].label if hook.when in options else hook.when,
                hook.say,
            )
            for hook in found.hooks
        ],
    )


def customisation(tools: Iterable[ToolSpec] = CATALOG) -> AgentCustomisation:
    """What the agent runs with: *tools* are the ones this server offers."""
    checkout = harness.find_checkout()
    claude = (checkout or Path(".")) / ".claude"
    return AgentCustomisation(
        tools=[_tool(spec) for spec in tools],
        categories=list(CATEGORIES),
        skills=_skills(harness.SKILLS, claude / "skills", "{name}/SKILL.md"),
        specialists=_skills(harness.SPECIALISTS, claude / "agents", "{name}.md"),
        profiles=[_profile(path, checkout) for path in profiles.available(checkout)],
        reminder_options=[option.label for option in profiles.HOOK_OPTIONS.values()],
        never=[BlockRow(**row) for row in blocklist.rules()],
    )

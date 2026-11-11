"""Where the agent's skills, subagents and playbooks come from.

Two places, the first found winning by name:

1. The lab's own, in the knowledge base on this PC
   (``~/.fibsem/knowledge/skills/<name>/SKILL.md``,
   ``~/.fibsem/knowledge/agents/<name>.md`` and
   ``~/.fibsem/knowledge/playbooks/<name>.yaml``).
2. What fibsemOS ships: a checkout's ``.claude/``, or, from a pip install, the
   copy of it inside the package (``SHIPPED``, made when the wheel is built).

The CLI reads skills only from its working folder, so `assemble` copies the
ones a session uses into that folder's ``.claude``.
"""

import shutil
from pathlib import Path
from typing import Iterable, List, Optional

# The shipped copy of the checkout's .claude, laid out as the checkout has it.
SHIPPED = Path(__file__).resolve().parent / "shipped"
CLAUDE = ".claude"
SKILL_FILE = "SKILL.md"


def lab_folder(kind: str) -> Path:
    """The lab's own *kind* (skills, agents, playbooks), not made if missing:
    `fibsem.knowledge.lab_folder` makes it."""
    # Here, not at the top: setup.py loads this file without fibsem.
    from fibsem import paths

    return paths.state_root() / "knowledge" / kind


def find_root(given: Optional[str] = None, base_skill: str = "fibsem-agent"):
    """The folder whose ``.claude`` holds the shipped skills: the checkout
    given, else the one this module runs from, else the current folder, else
    the copy in the package. None when none has *base_skill*."""
    if given:
        candidates = [Path(given).expanduser()]
    else:
        candidates = [Path(__file__).resolve().parents[3], Path.cwd(), SHIPPED]
    for candidate in candidates:
        if (candidate / CLAUDE / "skills" / base_skill / SKILL_FILE).is_file():
            return candidate.resolve()
    return None


def skill_dir(name: str, root: Optional[Path]) -> Optional[Path]:
    """The folder of skill *name*: the lab's, else the shipped one."""
    for folder in (lab_folder("skills"), root and root / CLAUDE / "skills"):
        if folder and (folder / name / SKILL_FILE).is_file():
            return folder / name
    return None


def lab_skills() -> List[str]:
    """The names of the lab's own skills, sorted."""
    folder = lab_folder("skills")
    if not folder.is_dir():
        return []
    return sorted(p.parent.name for p in folder.glob(f"*/{SKILL_FILE}"))


def agent_file(name: str, root: Optional[Path]) -> Optional[Path]:
    """A subagent's file: the lab's, else the shipped one."""
    for folder in (lab_folder("agents"), root and root / CLAUDE / "agents"):
        if folder and (folder / f"{name}.md").is_file():
            return folder / f"{name}.md"
    return None


def lab_agents() -> List[str]:
    """The names of the lab's own subagents, sorted."""
    folder = lab_folder("agents")
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.md"))


def assemble(
    target: Path, root: Path, skills: Iterable[str], agents: Iterable[str] = ()
) -> List[str]:
    """Make ``target/.claude`` hold *skills* and *agents*, each from where it
    is found, and nothing else: it is rebuilt every start, so nothing left
    there (a settings file with hooks) outlives a session. Answers the names
    of the skills copied from the lab's folder."""
    target.mkdir(parents=True, exist_ok=True)
    claude = target / CLAUDE
    if claude.is_symlink() or claude.is_file():
        claude.unlink()
    elif claude.exists():
        shutil.rmtree(claude)
    (claude / "skills").mkdir(parents=True)
    from_lab = []
    lab = lab_folder("skills")
    for name in skills:
        found = skill_dir(name, root)
        if found is None:
            raise FileNotFoundError(f"No skill {name} in {lab} or {root}")
        shutil.copytree(found, claude / "skills" / name)
        if found.parent == lab:
            from_lab.append(name)
    agents = list(agents)
    if agents:
        (claude / "agents").mkdir()
    for name in agents:
        found = agent_file(name, root)
        if found is None:
            raise FileNotFoundError(
                f"No subagent {name} in {lab_folder('agents')} or {root}"
            )
        shutil.copyfile(found, claude / "agents" / f"{name}.md")
    return from_lab


def ship(source: Path, target: Path) -> None:
    """Copy a checkout's skills, subagents and playbooks to *target*'s
    ``.claude``: what the wheel ships (setup.py's build step)."""
    for kind in ("skills", "agents", "fibsem-agent"):
        found = source / CLAUDE / kind
        if found.is_dir():
            shutil.copytree(
                found,
                target / CLAUDE / kind,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )

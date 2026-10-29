"""``fibsem-agent --code [PATH]``: the agent writes and runs its own code in
PATH, inside Claude Code's sandbox.

Everything it runs is sandboxed: writes stay in PATH, reads outside it are
refused, there is no network (so no App Server either: the fibsem tools are
the only way to the microscope), and secrets are unset. Its code runs with
fibsem-agent's own Python environment, which it can read but not change. The
sandbox must start, or the CLI refuses the command: code never runs
unconfined. There is no sandbox on Windows, so code mode is macOS and Linux
only.

``fibsem-agent --files [PATH]`` is the same without the shell: the agent reads
and writes files in PATH (its notebook, what it fetched) and runs nothing, so
it needs no sandbox and works on Windows too. The folder rules are Claude
Code's permission rules, the same as code mode's file tools.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePath
from typing import List, Mapping, Optional

# The file tools on top of the harness's own, and the code tools: those and a
# shell.
FILE_TOOLS = ["Read", "Write", "Edit", "Glob", "Grep"]
CODE_TOOLS = ["Bash"] + FILE_TOOLS

# Unset in every sandboxed command, with any other variable whose name says
# it is a secret. Some the CLI sets itself, after these settings are written.
SECRETS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "FIBSEM_SERVER_TOKEN",
)
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)

NOTEBOOK = "notebook"
INDEX = "INDEX.md"
INDEX_TEXT = """\
# Lab notebook

One line per entry: what it is, and its file.

## Lessons

## Scripts
"""

CODE_PROMPT = """\
- Your tools are the mcp__fibsem__* tools, Skill, Agent for the \
autolamella-* agents only, and code mode's shell, \
Read, Write, Edit, Glob and Grep, which work in {work} only, in a sandbox with no \
network: code can't reach the App Server or the web, so the mcp__fibsem__* \
tools are the only way to the microscope and its data. \
experiment_fetch_file saves a file under fetched/ for your code. Run Python \
with {python}: it has fibsem and what fibsem-agent's environment has. You \
can't install anything: say what you are missing.
"""
FILES_PROMPT = """\
- Your tools are the mcp__fibsem__* tools, Skill, Agent for the \
autolamella-* agents only, and Read, Write, Edit, Glob and Grep, which work in {work} \
only. There is no shell and no web: you can't run code. Where a skill shows \
curl, python or a script, use the matching mcp__fibsem__ tool, or say what \
you can't do. experiment_fetch_file saves a file under fetched/ for you to \
Read.
"""
PROMPT = """\
- Notes go on the open experiment's record. A note can't cite another \
experiment's file as its source: attach your own image (a plot, a crop) and \
name the experiment and file in the text.
- You can't ask and wait inside a turn. Where your goal leaves a choice open \
and a wrong pick is only a wasted read (which experiment, which file), pick \
a reasonable one and say which and why in your note. Where a wrong pick \
can't be undone, ask the operator with experiment_message_operator and \
end your turn.
- Your notebook is notebook/. Read notebook/INDEX.md before your first \
analysis. Write lessons and scripts worth reusing there, one line each in \
INDEX.md. Only conclusions go on the record: a note and one evidence image."""


def prompt(work: Path, code: bool = True) -> str:
    """What the agent is told about its folder: code mode's or files-only."""
    tools = CODE_PROMPT if code else FILES_PROMPT
    return tools.format(work=work, python=sys.executable) + PROMPT


def tools(code: bool = True) -> List[str]:
    return list(CODE_TOOLS if code else FILE_TOOLS)


def refusal(work: Path, checkout: Path, code: bool = True) -> Optional[str]:
    """Why code mode (files-only mode, without ``code``) can't run in
    ``work``, or None."""
    if code and sys.platform == "win32":
        return (
            "Code mode needs Claude Code's sandbox, which Windows doesn't have: "
            "run fibsem-agent --code on macOS or Linux."
        )
    home = Path.home().resolve()
    if work in (home, Path(work.anchor)) or home.is_relative_to(work):
        return (
            f"The agent would read and write all of {work}: give "
            "it a folder of its own, such as ~/fibsem-work."
        )
    # What runs outside the sandbox must not be the agent's to change.
    for name, path in (("checkout", checkout), ("Python", Path(sys.prefix))):
        path = path.resolve()
        if work.is_relative_to(path) or path.is_relative_to(work):
            return (
                f"{work} holds fibsem-agent's own {name} ({path}), which the "
                "agent must not change: give it a folder of its own, and run "
                "fibsem-agent from another environment."
            )
    return None


def describe(work: Optional[Path], code: bool = True) -> str:
    """Code mode in fibsem-agent's start status."""
    if work is not None and not code:
        return f"Off; reads and writes files in {work} (--files)"
    if work is not None:
        return f"Available in {work} (sandboxed)"
    if sys.platform == "win32":
        return "Not available (no sandbox on Windows)"
    return "Off (--code to turn it on)"


def details() -> List[str]:
    """For the log: what the agent's code reads, and its Python."""
    readonly = ", ".join(str(path) for path in readable())
    return [f"Code reads, read only: {readonly}", f"Code runs with {sys.executable}"]


def _rule(path: PurePath) -> str:
    # "//" starts an absolute path in a permission rule. Rules match Windows
    # paths in POSIX form, its drive a folder: C:\x is /c/x.
    posix = path.as_posix()
    if len(path.drive) == 2 and path.drive[1] == ":":
        posix = f"/{path.drive[0].lower()}{posix[2:]}"
    return f"/{posix}/**"


def secrets(environ: Mapping[str, str]) -> List[str]:
    """The variables a sandboxed command must not see."""
    names = set(SECRETS)
    names.update(name for name in environ if SECRET_NAME.search(name))
    return sorted(names)


def readable() -> List[Path]:
    """What the agent's code reads outside ``work``: this Python environment,
    the Python it is built on, and fibsem (a checkout, if installed editable)."""
    import fibsem

    paths = [Path(sys.prefix), Path(sys.base_prefix)]
    paths.append(Path(fibsem.__file__).resolve().parents[1])
    found: List[Path] = []
    for path in (p.resolve() for p in paths):
        if not any(path.is_relative_to(other) for other in found):
            found = [other for other in found if not other.is_relative_to(path)]
            found.append(path)
    return found


def allowed_tools(work: Path, code: bool = True) -> List[str]:
    """The code tools (the file tools, without ``code``), writes in ``work``
    only."""
    return ["Bash"] * code + [
        "Read",
        "Glob",
        "Grep",
        # Edit rules cover every file-editing tool, Write too.
        f"Edit({_rule(work)})",
    ]


def settings(work: Path, environ: Mapping[str, str], code: bool = True) -> dict:
    """The CLI's settings for code mode, given on its command line, where a
    project's own settings can't loosen them. Without ``code`` nothing runs,
    so there is no sandbox to set."""
    claude = _rule(work / ".claude")
    permissions = {
        # The Read, Glob and Grep tools, and the sandbox: home and other
        # user folders are unreadable outside ``work``.
        "blockReadsOutsideWorkingDirectories": True,
        # Its skills and settings stay as fibsem-agent put them.
        "deny": [f"Edit({claude})"],
    }
    if not code:
        return {"permissions": permissions}
    return {
        "permissions": permissions,
        "sandbox": {
            "enabled": True,
            # No sandbox, no command: never unconfined.
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            # Sandboxed is what makes a command safe to run unasked.
            "autoAllowBashIfSandboxed": True,
            "network": {"allowedDomains": []},
            "filesystem": {
                # Read only: writes stay in ``work``.
                "allowRead": [str(path) for path in readable()],
                # Not even there: the CLI reads its settings and hooks from it.
                "denyWrite": [str(work / ".claude")],
            },
            "credentials": {
                "envVars": [{"name": name, "mode": "deny"} for name in secrets(environ)]
            },
        },
    }


def settings_json(work: Path, environ: Mapping[str, str], code: bool = True) -> str:
    return json.dumps(settings(work, environ, code))


def env(work: Path) -> dict:
    """What the CLI and the sidecar need to know about ``work``."""
    return {
        # The sidecar's experiment_fetch_file saves under work/fetched.
        "FIBSEM_AGENT_WORKDIR": str(work),
        # matplotlib can't write its default under home.
        "MPLCONFIGDIR": str(work / ".cache" / "matplotlib"),
    }


def prepare(
    work: Path, checkout: Path, skills: List[str], agents: List[str] = ()
) -> None:
    """Make ``work`` ready: the agent's skills and specialists copied in from
    the checkout (the CLI reads them from its working folder) and the
    notebook. Its ``.claude`` is rebuilt every start, so nothing else there
    (a settings file with hooks) outlives a session."""
    work.mkdir(parents=True, exist_ok=True)
    claude = work / ".claude"
    if claude.is_symlink() or claude.is_file():
        claude.unlink()
    elif claude.exists():
        shutil.rmtree(claude)
    target = claude / "skills"
    target.mkdir(parents=True, exist_ok=True)
    for name in skills:
        shutil.copytree(checkout / ".claude" / "skills" / name, target / name)
    if agents:
        (work / ".claude" / "agents").mkdir(exist_ok=True)
    for name in agents:
        shutil.copyfile(
            checkout / ".claude" / "agents" / f"{name}.md",
            work / ".claude" / "agents" / f"{name}.md",
        )
    notebook = work / NOTEBOOK
    notebook.mkdir(exist_ok=True)
    if not (notebook / INDEX).exists():
        (notebook / INDEX).write_text(INDEX_TEXT, encoding="utf-8")
    if not (notebook / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(notebook)], check=False)


def workdir(given: str) -> Path:
    return Path(os.path.expanduser(given)).resolve()

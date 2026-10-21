"""``fibsem-agent --code PATH``: the agent writes and runs its own code in
PATH, inside Claude Code's sandbox.

Everything it runs is sandboxed: writes stay in PATH, reads outside it are
refused, there is no network (so no App Server either: the fibsem tools are
the only way to the microscope), and secrets are unset. The sandbox must
start, or the CLI refuses the command: code never runs unconfined. There is
no sandbox on Windows, so code mode is macOS and Linux only.
"""

import configparser
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping, Optional

# The code tools on top of the runner's own.
CODE_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]

# Unset in every sandboxed command, with any other variable whose name says
# it is a secret.
SECRETS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
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

PROMPT = """\
- Your tools are the mcp__fibsem__* tools, Skill, and code mode's shell, \
Read, Write and Edit, which work in {work} only, in a sandbox with no \
network: code can't reach the App Server or the web, so the mcp__fibsem__* \
tools are the only way to the microscope and its data. \
experiment_fetch_file saves a file under fetched/ for your code. Run Python \
with .venv/bin/python; it has numpy, scipy, scikit-image, matplotlib and \
fibsem. You can't install anything: say what you are missing.
- Your notebook is notebook/. Read notebook/INDEX.md before your first \
analysis. Write lessons and scripts worth reusing there, one line each in \
INDEX.md. Only conclusions go on the record: a note and one evidence image."""


def refusal() -> Optional[str]:
    """Why code mode can't run here, or None."""
    if sys.platform == "win32":
        return (
            "Code mode needs Claude Code's sandbox, which Windows doesn't have: "
            "run fibsem-agent --code on macOS or Linux."
        )
    return None


def _rule(path: Path) -> str:
    # "//" starts an absolute path in a permission rule.
    return f"/{path.as_posix()}/**"


def secrets(environ: Mapping[str, str]) -> List[str]:
    """The variables a sandboxed command must not see."""
    names = set(SECRETS)
    names.update(name for name in environ if SECRET_NAME.search(name))
    return sorted(names)


def allowed_tools(work: Path) -> List[str]:
    """The code tools, writes in ``work`` only."""
    return [
        "Bash",
        "Read",
        "Glob",
        "Grep",
        # Edit rules cover every file-editing tool, Write too.
        f"Edit({_rule(work)})",
    ]


def settings(work: Path, environ: Mapping[str, str]) -> dict:
    """The CLI's settings for code mode, given on its command line, where a
    project's own settings can't loosen them."""
    claude = _rule(work / ".claude")
    return {
        "permissions": {
            # The Read, Glob and Grep tools, and the sandbox: home and other
            # user folders are unreadable outside ``work``.
            "blockReadsOutsideWorkingDirectories": True,
            # Its own skills and settings stay as fibsem-agent put them.
            "deny": [f"Edit({claude})"],
        },
        "sandbox": {
            "enabled": True,
            # No sandbox, no command: never unconfined.
            "failIfUnavailable": True,
            "allowUnsandboxedCommands": False,
            # Sandboxed is what makes a command safe to run unasked.
            "autoAllowBashIfSandboxed": True,
            "network": {"allowedDomains": []},
            "credentials": {
                "envVars": [{"name": name, "mode": "deny"} for name in secrets(environ)]
            },
        },
    }


def settings_json(work: Path, environ: Mapping[str, str]) -> str:
    return json.dumps(settings(work, environ))


def env(work: Path) -> dict:
    """What the CLI and the sidecar need to know about ``work``."""
    return {
        # The sidecar's experiment_fetch_file saves under work/fetched.
        "FIBSEM_AGENT_WORKDIR": str(work),
        # matplotlib can't write its default under home.
        "MPLCONFIGDIR": str(work / ".cache" / "matplotlib"),
    }


def prepare(work: Path, checkout: Path, skills: List[str]) -> List[str]:
    """Make ``work`` ready: the agent's skills copied in from the checkout
    (the CLI reads them from its working folder) and the notebook. Returns
    warnings for the operator."""
    work.mkdir(parents=True, exist_ok=True)
    target = work / ".claude" / "skills"
    target.mkdir(parents=True, exist_ok=True)
    for name in skills:
        if (target / name).exists():
            shutil.rmtree(target / name)
        shutil.copytree(checkout / ".claude" / "skills" / name, target / name)
    notebook = work / NOTEBOOK
    notebook.mkdir(exist_ok=True)
    if not (notebook / INDEX).exists():
        (notebook / INDEX).write_text(INDEX_TEXT, encoding="utf-8")
    if not (notebook / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(notebook)], check=False)
    return _venv_warnings(work)


def _venv_warnings(work: Path) -> List[str]:
    config = work / ".venv" / "pyvenv.cfg"
    if not config.is_file():
        return [
            f"No Python environment in {work / '.venv'}: the agent's code can't "
            "run until you make one (python3 -m venv .venv, then pip install "
            "numpy scipy scikit-image matplotlib fibsem)."
        ]
    parser = configparser.ConfigParser()
    try:
        parser.read_string("[venv]\n" + config.read_text(encoding="utf-8"))
    except configparser.Error:
        return []
    base = parser.get("venv", "home", fallback="")
    home = Path.home().resolve()
    if (
        base
        and Path(base).resolve().is_relative_to(home)
        and not Path(base).resolve().is_relative_to(work.resolve())
    ):
        return [
            f"{work / '.venv'} is built on a Python under your home folder "
            f"({base}), which the sandbox can't read: rebuild it on a Homebrew "
            "or python.org Python."
        ]
    return []


def workdir(given: str) -> Path:
    return Path(os.path.expanduser(given)).resolve()

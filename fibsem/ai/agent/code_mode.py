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
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import List, Mapping, Optional, Sequence, Tuple

from fibsem.ai.agent import sources

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
- Your tools are the mcp__fibsem__* tools, Skill, Agent for the subagents the \
playbook names, and code mode's shell, \
Read, Write, Edit, Glob and Grep, which work in {work} only, in a sandbox with no \
network: code can't reach the App Server or the web, so the mcp__fibsem__* \
tools are the only way to the microscope and its data. \
{fetch}Run Python \
with {python}: it has fibsem and what fibsem-agent's environment has. You \
can't install anything: say what you are missing.
"""
FILES_PROMPT = """\
- Your tools are the mcp__fibsem__* tools, Skill, Agent for the subagents the \
playbook names, and Read, Write, Edit, Glob and Grep, which work in {work} \
only. There is no shell and no web: you can't run code. Where a skill shows \
curl, python or a script, use the matching mcp__fibsem__ tool, or say what \
you can't do.{fetch_read}
"""
# What the fibsem sidecar adds: its experiment's files and record. Left out
# when the session has no fibsem server (an agent host's own servers).
FETCH = "experiment_fetch_file saves a file under fetched/ for your code. "
FETCH_READ = " experiment_fetch_file saves a file under fetched/ for you to Read."
EXPERIMENT_PROMPT = """\
- Notes go on the open experiment's record. A note can't cite another \
experiment's file as its source: attach your own image (a plot, a crop) and \
name the experiment and file in the text.
"""
PROMPT = """\
- You can't ask and wait inside a turn. Where your goal leaves a choice open \
and a wrong pick is only a wasted read (which experiment, which file), pick \
a reasonable one and say which and why in your note. Where a wrong pick \
can't be undone, ask the operator with experiment_message_operator and \
end your turn.
- Your notebook is notebook/. Read notebook/INDEX.md before your first \
analysis. Write lessons and scripts worth reusing there, one line each in \
INDEX.md. Only conclusions go on the record: a note and one evidence image."""


READ_PROMPT = """\
- You may also read {folders} with Read, Glob and Grep{code}. You can't \
write there: what you make goes in your own folder.
"""
SITES_PROMPT = """\
- One exception to no network: your code reaches these sites and nothing \
else, never an address in the lab: {sites}.
"""

# A site the sandbox may reach: a public host name, or every host under one
# (``*.example.org``). Never an IP address or a single-label name.
_SITE = re.compile(r"^(\*\.)?([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$")
# Names that stay inside a lab or a PC, whatever they resolve to.
_LOCAL_ENDS = (".localhost", ".local", ".internal", ".lan", ".home.arpa", ".intranet")


@dataclass(frozen=True)
class Grants:
    """What one start gives the agent beyond its own folder: folders it
    reads and never writes (``--read``), and the sites its sandboxed code
    reaches (``--allow-site``, code mode only)."""

    read: Tuple[Path, ...] = ()
    sites: Tuple[str, ...] = ()


def site_refusal(site: str) -> Optional[str]:
    """Why the sandbox must not reach *site*, or None: it is a local address
    (this PC, an IP address, a name with no dot, a lab-only ending), or not a
    host name at all. A name is all that is checked: where it resolves to is
    the network's."""
    name = str(site or "").strip().lower()
    if not name:
        return "A site is a host name, such as pypi.org."
    bare = name[2:] if name.startswith("*.") else name
    if bare == "localhost" or bare.endswith(_LOCAL_ENDS):
        return f"{site} is a local address: the sandbox never reaches one."
    if re.fullmatch(r"[0-9.]+|\[?[0-9a-f:]+\]?", bare):
        return f"{site} is an IP address: a site is given by its name, never a local address."
    if "." not in bare:
        return f"{site} is a name inside this network: the sandbox never reaches one."
    if not _SITE.fullmatch(name):
        return (
            f"{site} is not a host name: give a name such as pypi.org, with "
            "no http://, port or path."
        )
    return None


def grants(
    read: Sequence[str], sites: Sequence[str], work: Optional[Path], code: bool
) -> Grants:
    """The grants a command line asked for, checked. Raises ``ValueError``
    with why not."""
    if (read or sites) and work is None:
        raise ValueError(
            "--read and --allow-site go with a folder of the agent's own: "
            "add --files or --code."
        )
    if sites and not code:
        raise ValueError("--allow-site is for the code sandbox: add --code.")
    folders: List[Path] = []
    home = Path.home().resolve()
    for given in read:
        folder = workdir(given)
        if not folder.is_dir():
            raise ValueError(f"--read {given} is not a folder on this PC.")
        if folder in (home, Path(folder.anchor)) or home.is_relative_to(folder):
            raise ValueError(
                f"--read {folder} would let the agent read everything of "
                "yours: give it one data folder."
            )
        if work is not None and (
            folder.is_relative_to(work) or work.is_relative_to(folder)
        ):
            raise ValueError(
                f"--read {folder} and the agent's own folder ({work}) must be apart."
            )
        if folder not in folders:
            folders.append(folder)
    names: List[str] = []
    for site in sites:
        refused = site_refusal(site)
        if refused:
            raise ValueError(refused)
        name = site.strip().lower()
        if name not in names:
            names.append(name)
    return Grants(tuple(folders), tuple(names))


def prompt(
    work: Path,
    code: bool = True,
    fibsem: bool = True,
    given: Optional[Grants] = None,
) -> str:
    """What the agent is told about its folder: code mode's or files-only;
    with *fibsem*, about the sidecar's experiment files and record too; and
    what the start *given* it beyond the folder."""
    tools = CODE_PROMPT if code else FILES_PROMPT
    text = tools.format(
        work=work,
        python=sys.executable,
        fetch=FETCH if fibsem else "",
        fetch_read=FETCH_READ if fibsem else "",
    )
    if given is not None and given.read:
        text += READ_PROMPT.format(
            folders=", ".join(str(folder) for folder in given.read),
            code=", and your code may read it too" if code else "",
        )
    if given is not None and given.sites and code:
        text += SITES_PROMPT.format(sites=", ".join(given.sites))
    return text + (EXPERIMENT_PROMPT if fibsem else "") + PROMPT


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


def describe(
    work: Optional[Path], code: bool = True, given: Optional[Grants] = None
) -> str:
    """Code mode in fibsem-agent's start status."""
    more = ""
    if given is not None and given.read:
        more += "; reads " + ", ".join(str(folder) for folder in given.read)
    if work is not None and not code:
        return f"Off; reads and writes files in {work} (--files){more}"
    if work is not None:
        sites = given.sites if given is not None else ()
        network = f", network: {', '.join(sites)}" if sites else ""
        return f"Available in {work} (sandboxed{network}){more}"
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


def allowed_tools(
    work: Path, code: bool = True, given: Optional[Grants] = None
) -> List[str]:
    """The code tools (the file tools, without ``code``), writes in ``work``
    only; reads in the folders *given* too."""
    return (
        ["Bash"] * code
        + [
            "Read",
            "Glob",
            "Grep",
            # Edit rules cover every file-editing tool, Write too.
            f"Edit({_rule(work)})",
        ]
        + [f"Read({_rule(folder)})" for folder in (given.read if given else ())]
    )


def settings(
    work: Path,
    environ: Mapping[str, str],
    code: bool = True,
    given: Optional[Grants] = None,
) -> dict:
    """The CLI's settings for code mode, given on its command line, where a
    project's own settings can't loosen them. Without ``code`` nothing runs,
    so there is no sandbox to set. What the start *given* it is added: its
    read folders are working folders it may not edit, and its sites are the
    sandbox's whole network."""
    claude = _rule(work / ".claude")
    read = list(given.read) if given is not None else []
    sites = list(given.sites) if given is not None else []
    permissions = {
        # The Read, Glob and Grep tools, and the sandbox: home and other
        # user folders are unreadable outside ``work``.
        "blockReadsOutsideWorkingDirectories": True,
        # Its skills and settings stay as fibsem-agent put them; a folder it
        # was given to read is never written.
        "deny": [f"Edit({claude})"] + [f"Edit({_rule(folder)})" for folder in read],
    }
    if read:
        # Read as its own folder is, by the file tools.
        permissions["additionalDirectories"] = [str(folder) for folder in read]
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
            # None, but the sites the start named: never a local address
            # (`site_refusal`).
            "network": {"allowedDomains": sites},
            "filesystem": {
                # Read only: writes stay in ``work``.
                "allowRead": [str(path) for path in readable() + read],
                # Not even there: the CLI reads its settings and hooks from it.
                "denyWrite": [str(work / ".claude")] + [str(path) for path in read],
            },
            "credentials": {
                "envVars": [{"name": name, "mode": "deny"} for name in secrets(environ)]
            },
        },
    }


def settings_json(
    work: Path,
    environ: Mapping[str, str],
    code: bool = True,
    given: Optional[Grants] = None,
) -> str:
    return json.dumps(settings(work, environ, code, given))


def env(work: Path) -> dict:
    """What the CLI and the sidecar need to know about ``work``."""
    return {
        # The sidecar's experiment_fetch_file saves under work/fetched.
        "FIBSEM_AGENT_WORKDIR": str(work),
        # matplotlib can't write its default under home.
        "MPLCONFIGDIR": str(work / ".cache" / "matplotlib"),
    }


def prepare(
    work: Path,
    checkout: Path,
    skills: List[str],
    agents: List[str] = (),
    pinned: Optional[Path] = None,
) -> None:
    """Make ``work`` ready: the agent's skills and specialists copied in (the
    CLI reads them from its working folder; `sources.assemble`, the lab's
    winning over the checkout's) and the notebook."""
    sources.assemble(work, checkout, skills, agents, pinned)
    notebook = work / NOTEBOOK
    notebook.mkdir(exist_ok=True)
    if not (notebook / INDEX).exists():
        (notebook / INDEX).write_text(INDEX_TEXT, encoding="utf-8")
    if not (notebook / ".git").exists() and shutil.which("git"):
        subprocess.run(["git", "init", "-q", str(notebook)], check=False)


def workdir(given: str) -> Path:
    return Path(os.path.expanduser(given)).resolve()

"""Tools for reporting issues and submitting bug reports (optionally with data).

Two paths are supported:

* **Public report** — open a pre-filled GitHub issue in the browser. No data
  leaves the machine beyond what the user types + basic environment info.
* **Private data bundle** — build a scrubbed ``.zip`` of the selected experiment
  artifacts (log file, experiment/protocol yaml, optionally screenshots/images)
  for the user to email to the support address themselves.

The bundle is the deliverable, and it is self-contained: ``report.md`` inside it
carries the whole report. Nothing here tries to send it. Instrument PCs commonly
have no mail client, no browser session and sometimes no network, so composing
the mail is the user's step, on whichever machine they actually have email --
the application's job ends at a findable file and an address to send it to.
:func:`open_github_issue` is the one outbound action, and it reports whether it
actually opened anything rather than assuming.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import re
import socket
import webbrowser
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import fibsem
import fibsem.config as fibsem_cfg
from fibsem.ai.agent import manager as agent_sessions
from fibsem.versioning import get_revision

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope

# Support / issue destinations
SUPPORT_EMAIL = "contact@fibsemos.org"
GITHUB_REPO_URL = "https://github.com/fibsem-os/fibsem-os"
GITHUB_NEW_ISSUE_URL = f"{GITHUB_REPO_URL}/issues/new"

# The budget is on the *encoded* URL, never the raw text: percent-encoding turns
# every newline into three characters, so a body that looks comfortably short
# still overflows once quoted -- and a crash report pre-fills the description
# with a full traceback. GitHub itself accepts a longer prefill, but browsers
# and corporate proxies start dropping query strings well before its limit.
GITHUB_MAX_URL_LENGTH = 7000

_TRUNCATION_NOTE = "\n\n[trimmed -- the full text was copied to your clipboard]"

# A user-typed title goes into the URL alongside the body. Capping it keeps a
# runaway title (a pasted traceback, say) from eating the whole budget.
_TITLE_MAX_LENGTH = 120

# Text file extensions whose contents are scrubbed before being added to a bundle.
_SCRUBBED_EXTENSIONS = {
    ".log",
    ".yaml",
    ".yml",
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
}

# Rough per-file globs used to locate optional artifacts within an experiment.
# The generic .tif/.tiff patterns also cover .ome.tif/.ome.tiff.
_SCREENSHOT_GLOB = "**/*.png"
_IMAGE_GLOBS = ("**/*.tif", "**/*.tiff")

# Matches a home-directory prefix (group 1) followed by the username segment.
_HOME_PATH_RE = re.compile(r"([/\\](?:Users|home)[/\\])[^/\\\s]+")

# An absolute path with at least three parts: the folders before the last two
# (group 1) say where a site keeps its data, the last two (group 2) are enough
# to tell which file it was.
_PATH_PART = r"[^/\\\s:\"'<>|]+"
_SEP = r"[/\\]+"
_ABSOLUTE_PATH_RE = re.compile(
    rf"(?<![\w<>/\\.:])((?:[A-Za-z]:)?(?:{_SEP}{_PATH_PART})+?)"
    rf"((?:{_SEP}{_PATH_PART}){{2}})(?![^\s:\"'<>|])"
)

# Wherever Python's packages are installed: a traceback keeps the path inside.
_SITE_PACKAGES_RE = re.compile(
    rf"(?<![\w<>/\\.:])(?:[A-Za-z]:)?(?:{_SEP}{_PATH_PART})*?{_SEP}(?:site|dist)-packages"
)

# Keys of a saved experiment that name a person, a place or a machine.
_IDENTITY_KEYS_RE = re.compile(
    r"^(\s*[\"']?(?:user|email|organisation|organization|project|hostname)[\"']?"
    r"\s*:[ \t]*)(?!null\b|~|[\"']{2})\S.*$",
    re.MULTILINE,
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
# Not this computer's own address, which says nothing about the site.
_IP_RE = re.compile(r"(?<![\w.])(?!127\.|0\.0\.0\.0)\d{1,3}(?:\.\d{1,3}){3}(?![\w.])")
# The values of the keys that name a person or a place, to find them elsewhere.
_NAMED_RE = re.compile(
    r"^\s*[\"']?(?:user|organisation|organization|project)[\"']?\s*:[ \t]*"
    r"[\"']?([^\"'\n#,]+?)[\"']?,?[ \t]*$",
    re.MULTILINE,
)

# Machine names that are also ordinary words: replacing them would hide more of
# a log than it protects.
_GENERIC_HOSTNAMES = {"localhost", "hostname", "microscope", "desktop", "support"}


@dataclass
class BugReportContent:
    """User-entered content and data-inclusion choices for a bug report."""

    title: str = ""
    description: str = ""
    steps: str = ""
    severity: str = "Normal"
    contact_email: str = ""

    include_logfile: bool = True
    include_experiment_yaml: bool = True
    include_protocol: bool = True
    include_events: bool = True
    include_screenshots: bool = False
    include_images: bool = False
    # From this PC, whether or not an experiment is open.
    include_crash_records: bool = True
    include_settings: bool = True
    include_agent_log: bool = False

    system_context: Dict[str, str] = field(default_factory=dict)


@dataclass
class SubmitResult:
    """Outcome of handing a report to an external application.

    ``opened`` is what the platform reported, not proof that the user saw a
    window: a browser that launches but never reaches GitHub still reports
    success. It is trustworthy in the negative, which is the case that matters
    on an instrument PC -- no browser, or none registered, means the report was
    not filed and the user has to be told.

    ``full_text`` is the untruncated body, for the caller to put on the
    clipboard when ``truncated`` is set.
    """

    opened: bool
    full_text: str
    truncated: bool = False


def collect_system_context(
    microscope: Optional["FibsemMicroscope"] = None,
) -> Dict[str, str]:
    """Gather non-identifying environment information for a bug report.

    Deliberately excludes usernames, paths and IP addresses.
    """
    ctx: Dict[str, str] = {
        "fibsem_version": getattr(fibsem, "__version__", "unknown"),
        # The commit, for source installs. Deliberately not the branch: branch
        # names routinely embed usernames, which this function promises to omit
        # and scrub_text cannot catch (it only strips the *local* user).
        # Empty for a wheel install, and dropped by the filter below.
        "fibsem_revision": get_revision() or "",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    try:
        import napari

        ctx["napari"] = str(napari.__version__)
    except Exception:
        pass
    try:
        import qtpy
        from qtpy.QtCore import __version__ as qt_version

        ctx["Qt"] = f"{qt_version} ({qtpy.API_NAME})"
    except Exception:
        pass

    if microscope is not None:
        try:
            info = microscope.system.info
            ctx["microscope_manufacturer"] = str(getattr(info, "manufacturer", ""))
            ctx["microscope_model"] = str(getattr(info, "model", ""))
            ctx["microscope_software_version"] = str(
                getattr(info, "software_version", "")
            )
        except Exception:
            logging.debug("Could not read microscope system info for bug report.")

    return {k: v for k, v in ctx.items() if v}


def _hostname() -> str:
    try:
        return os.environ.get("COMPUTERNAME") or socket.gethostname()
    except Exception:
        return ""


def _known_folders(experiment_path: Optional[str]) -> List[Tuple[str, str]]:
    """Folders whose location is private but whose contents are worth keeping,
    with what each is replaced by. Longest first, so a folder inside another is
    replaced before its parent."""
    folders: List[Tuple[str, str]] = []
    if experiment_path:
        # Either separator: a Windows path is scrubbed on any platform.
        parent = re.sub(r"[/\\]+[^/\\]*[/\\]*$", "", str(experiment_path))
        if len(parent) > 3:
            folders.append((parent, "<data>"))
    try:
        folders.append((os.path.dirname(os.path.dirname(fibsem.__file__)), "<install>"))
    except Exception:
        pass
    try:
        folders.append((os.path.expanduser("~"), "<home>"))
    except Exception:
        pass
    return sorted(folders, key=lambda pair: len(pair[0]), reverse=True)


def _replace_folder(text: str, folder: str, placeholder: str) -> str:
    """Replace ``folder`` with either kind of separator, whatever its case."""
    parts = [re.escape(part) for part in re.split(r"[/\\]", folder)]
    pattern = r"[/\\]+".join(parts) + r"(?![\w.-])"
    return re.sub(pattern, lambda m: placeholder, text, flags=re.IGNORECASE)


def scrub_text(text: str, experiment_path: Optional[str] = None) -> str:
    """Best-effort removal of what names the person, the site or the machine.

    Folders are cut down, not blanked: the experiment keeps its own folder name
    and a traceback keeps its path inside the package, because that is what a
    report is read for. The login is removed only where it is a folder name, so
    an account called ``user`` does not blank the word everywhere.
    """
    if not text:
        return text

    scrubbed = text
    for folder, placeholder in _known_folders(experiment_path):
        scrubbed = _replace_folder(scrubbed, folder, placeholder)

    # The username is the path segment following the prefix, up to the next
    # separator or whitespace (so a trailing username is still redacted).
    scrubbed = _HOME_PATH_RE.sub(lambda m: m.group(1) + "<user>", scrubbed)

    try:
        login = getpass.getuser()
    except Exception:
        login = ""
    if len(login) > 2:
        scrubbed = re.sub(
            rf"(?<=[/\\=]){re.escape(login)}(?![\w.-])", "<user>", scrubbed
        )

    scrubbed = _SITE_PACKAGES_RE.sub("<packages>", scrubbed)
    scrubbed = _ABSOLUTE_PATH_RE.sub(lambda m: "<path>" + m.group(2), scrubbed)

    hostname = _hostname()
    if len(hostname) > 3 and hostname.lower() not in _GENERIC_HOSTNAMES:
        scrubbed = re.sub(
            rf"(?<![\w-]){re.escape(hostname)}(?![\w-])",
            "<host>",
            scrubbed,
            flags=re.IGNORECASE,
        )

    return scrubbed


def _redact_value(match: "re.Match[str]") -> str:
    """The key with its value redacted, still valid where the file is JSON."""
    value = match.group(0)[len(match.group(1)) :]
    if value[:1] == '"':
        return match.group(1) + '"<redacted>"' + ("," if value.endswith(",") else "")
    return match.group(1) + "<redacted>"


def names_in(experiment_path: Optional[str]) -> List[str]:
    """Who and where the experiment says it is: the operator, organisation and
    project typed when it was created."""
    if not experiment_path:
        return []
    try:
        with open(
            os.path.join(str(experiment_path), "experiment.yaml"),
            encoding="utf-8",
            errors="replace",
        ) as f:
            text = f.read()
    except OSError:
        return []
    names = {m.group(1).strip() for m in _NAMED_RE.finditer(text)}
    # Short or empty values would blank ordinary words wherever they appear.
    return sorted(n for n in names if len(n) > 3 and n.lower() not in ("null", "none"))


def scrub_file_text(
    text: str, experiment_path: Optional[str] = None, names: Sequence[str] = ()
) -> str:
    """:func:`scrub_text` for a file out of the experiment or off this PC, which
    also names the operator, their organisation and project, and may hold email
    and network addresses. ``names`` are removed wherever they appear whole."""
    scrubbed = _IDENTITY_KEYS_RE.sub(_redact_value, text)
    for name in names:
        scrubbed = re.sub(
            rf"(?<![\w-]){re.escape(name)}(?![\w-])", "<redacted>", scrubbed
        )
    scrubbed = _EMAIL_RE.sub("<email>", scrubbed)
    scrubbed = _IP_RE.sub("<ip>", scrubbed)
    return scrub_text(scrubbed, experiment_path)


def _collect_files(experiment_path: str, content: BugReportContent) -> List[str]:
    """Resolve the absolute paths of the artifacts selected for inclusion."""
    from glob import glob

    exp_path = str(experiment_path)
    files: List[str] = []

    def _add(path: str) -> None:
        if path and os.path.isfile(path) and path not in files:
            files.append(path)

    if content.include_logfile:
        _add(os.path.join(exp_path, "logfile.log"))
    if content.include_experiment_yaml:
        _add(os.path.join(exp_path, "experiment.yaml"))
    if content.include_protocol:
        _add(os.path.join(exp_path, "protocol.yaml"))
    if content.include_events:
        _add(os.path.join(exp_path, "events.jsonl"))
    if content.include_screenshots:
        for path in glob(os.path.join(exp_path, _SCREENSHOT_GLOB), recursive=True):
            _add(path)
    if content.include_images:
        for pattern in _IMAGE_GLOBS:
            for path in glob(os.path.join(exp_path, pattern), recursive=True):
                _add(path)

    return files


def _files_from_this_pc(content: BugReportContent) -> List[Tuple[str, str]]:
    """What is selected that belongs to the PC, not to an experiment: each
    file's path and its name in the bundle."""
    from fibsem import diagnostics

    files: List[Tuple[str, str]] = []

    def _add(path, folder: str) -> None:
        if path and os.path.isfile(path):
            files.append((str(path), f"{folder}/{os.path.basename(path)}"))

    if content.include_crash_records:
        crashes = diagnostics.crash_folder()
        if crashes.is_dir():
            for path in sorted(crashes.iterdir()):
                _add(path, "crashes")
    if content.include_settings:
        _add(fibsem_cfg.USER_PREFERENCES_PATH, "settings")
        _add(fibsem_cfg.DEFAULT_CONFIGURATION_PATH, "settings")
    if content.include_agent_log:
        sessions = agent_sessions.folders()
        if sessions:
            _add(sessions[0] / agent_sessions.RECORD, "agent")
            _add(sessions[0] / agent_sessions.LOG, "agent")
    return files


def estimate_bundle_size(
    experiment_path: Optional[str], content: BugReportContent
) -> int:
    """Estimate the on-disk size (bytes) of the selected artifacts."""
    paths = [path for path, _ in _files_from_this_pc(content)]
    if experiment_path is not None:
        paths += _collect_files(experiment_path, content)
    total = 0
    for path in paths:
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def _clip_at_line(text: str, size: int) -> str:
    """The first ``size`` characters of ``text``, cut back to a line boundary.

    Falls back to a hard cut when the last newline is in the front half, so a
    single enormous line still gets trimmed rather than emptied.
    """
    head = text[:size]
    newline = head.rfind("\n")
    return head[:newline] if newline > size // 2 else head


def _fit_url(
    build_url: Callable[[str], str], text: str, limit: int
) -> Tuple[str, bool]:
    """Build a URL from ``text``, trimming ``text`` until the URL fits ``limit``.

    Binary-searches the longest prefix that fits once encoded, so the caller
    gets as much of the report as the platform will carry. Returns the URL and
    whether anything was dropped -- the caller owes the user the full text on
    the clipboard when it was.
    """
    if len(build_url(text)) <= limit:
        return build_url(text), False

    def _candidate(size: int) -> str:
        return _clip_at_line(text, size) + _TRUNCATION_NOTE

    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if len(build_url(_candidate(mid))) <= limit:
            low = mid
        else:
            high = mid - 1

    return build_url(_candidate(low)), True


def _render_report_text(content: BugReportContent) -> str:
    """Render the human-readable report body written to ``report.md``."""
    lines = [
        f"# fibsemOS bug report: {content.title or '(no title)'}",
        "",
        f"Severity: {content.severity}",
        f"Contact: {content.contact_email or '(not provided)'}",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Description",
        content.description or "(none)",
        "",
        "## Steps to reproduce",
        content.steps or "(none)",
        "",
        "## Environment",
    ]
    for key, value in content.system_context.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def build_bug_report_bundle(
    content: BugReportContent,
    experiment_path: Optional[str],
    output_dir: Optional[str] = None,
) -> str:
    """Build a scrubbed ``.zip`` bundle for the bug report and return its path.

    Text artifacts (logs, yaml, json, ...) go through :func:`scrub_file_text`
    before being written. Binary artifacts (images) are included verbatim, and
    only when explicitly selected.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    zip_name = f"bug-report-fibsem-{timestamp}.zip"

    if output_dir is None:
        # Prefer alongside the experiment; fall back to the log directory.
        if experiment_path is not None and os.path.isdir(str(experiment_path)):
            output_dir = str(experiment_path)
        else:
            output_dir = fibsem_cfg.LOG_PATH
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, zip_name)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.md", scrub_text(_render_report_text(content)))
        zf.writestr(
            "system_info.json",
            scrub_text(json.dumps(content.system_context, indent=2)),
        )

        members = _files_from_this_pc(content)
        if experiment_path is not None:
            exp_path = str(experiment_path)
            members += [
                (path, os.path.relpath(path, os.path.dirname(exp_path)))
                for path in _collect_files(experiment_path, content)
            ]
        names = names_in(experiment_path)
        for path, arcname in members:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext in _SCRUBBED_EXTENSIONS:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        text = scrub_file_text(f.read(), experiment_path, names)
                    zf.writestr(arcname, text)
                else:
                    zf.write(path, arcname)
            except OSError as e:
                logging.warning("Skipping %s in bug report bundle: %s", path, e)

    logging.info("Created bug report bundle at %s", zip_path)
    return zip_path


def _github_issue_body(content: BugReportContent) -> str:
    """Markdown body for a GitHub issue (no private data)."""
    lines = [
        "### Description",
        content.description or "_(describe the issue)_",
        "",
        "### Steps to reproduce",
        content.steps or "_(steps)_",
        "",
        "### Environment",
    ]
    for key, value in content.system_context.items():
        lines.append(f"- **{key}**: {value}")
    lines += [
        "",
        "---",
        "_Filed from Report an Issue in fibsemOS. "
        "Do not paste private/experiment data into this public issue — "
        "use the email option to submit data privately._",
    ]
    return "\n".join(lines)


def open_github_issue(content: BugReportContent) -> SubmitResult:
    """Open a pre-filled GitHub issue in the default browser.

    The body is trimmed until the encoded URL fits
    :data:`GITHUB_MAX_URL_LENGTH`; the result carries the full text so the
    caller can offer it on the clipboard instead.
    """
    # A public page: what the user typed or pasted is scrubbed like the bundle.
    body = scrub_text(_github_issue_body(content))
    title = scrub_text(content.title or "fibsemOS issue")[:_TITLE_MAX_LENGTH]

    def _build(text: str) -> str:
        return f"{GITHUB_NEW_ISSUE_URL}?{urlencode({'title': title, 'body': text})}"

    url, truncated = _fit_url(_build, body, GITHUB_MAX_URL_LENGTH)

    logging.info("Opening GitHub issue in browser.")
    try:
        opened = bool(webbrowser.open(url))
    except Exception:
        logging.exception("Could not open a browser for the GitHub issue.")
        opened = False

    return SubmitResult(opened=opened, full_text=body, truncated=truncated)

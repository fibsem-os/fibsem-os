"""Tools for reporting issues and submitting bug reports (optionally with data).

Two paths are supported:

* **Public report** — open a pre-filled GitHub issue in the browser. No data
  leaves the machine beyond what the user types + basic environment info.
* **Private data bundle** — build a scrubbed ``.zip`` of the selected experiment
  artifacts (log file, experiment/protocol yaml, optionally screenshots/images)
  and open a pre-filled email to the support address so the user can attach it.

An inert :func:`init_sentry` hook is included so automatic crash reporting can be
enabled later (by installing ``sentry-sdk`` and setting a DSN in preferences)
without any UI changes.
"""

from __future__ import annotations

import getpass
import json
import logging
import os
import platform
import re
import webbrowser
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional
from urllib.parse import quote, urlencode

import fibsem
import fibsem.config as fibsem_cfg

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Experiment
    from fibsem.microscope import FibsemMicroscope

# Support / issue destinations
SUPPORT_EMAIL = "contact@fibsemos.org"
GITHUB_REPO_URL = "https://github.com/fibsem-os/fibsem-os"
GITHUB_NEW_ISSUE_URL = f"{GITHUB_REPO_URL}/issues/new"

# Text file extensions whose contents are scrubbed before being added to a bundle.
_SCRUBBED_EXTENSIONS = {".log", ".yaml", ".yml", ".txt", ".md", ".json", ".csv"}

# Rough per-file globs used to locate optional artifacts within an experiment.
# The generic .tif/.tiff patterns also cover .ome.tif/.ome.tiff.
_SCREENSHOT_GLOB = "**/*.png"
_IMAGE_GLOBS = ("**/*.tif", "**/*.tiff")

# Matches a home-directory prefix (group 1) followed by the username segment.
_HOME_PATH_RE = re.compile(r"([/\\](?:Users|home)[/\\])[^/\\\s]+")


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
    include_screenshots: bool = False
    include_images: bool = False

    system_context: Dict[str, str] = field(default_factory=dict)


def collect_system_context(
    microscope: Optional["FibsemMicroscope"] = None,
) -> Dict[str, str]:
    """Gather non-identifying environment information for a bug report.

    Deliberately excludes usernames, paths and IP addresses.
    """
    ctx: Dict[str, str] = {
        "fibsem_version": getattr(fibsem, "__version__", "unknown"),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
    }

    for mod in ("napari", "PyQt5.QtCore"):
        try:
            imported = __import__(mod, fromlist=["__version__"])
            version = getattr(
                imported, "__version__", getattr(imported, "QT_VERSION_STR", None)
            )
            if version:
                ctx[mod.split(".")[0]] = str(version)
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


def scrub_text(text: str) -> str:
    """Best-effort removal of the user's home directory / username from text."""
    if not text:
        return text

    replacements: List[str] = []
    try:
        replacements.append(os.path.expanduser("~"))
    except Exception:
        pass
    try:
        replacements.append(getpass.getuser())
    except Exception:
        pass

    scrubbed = text
    for value in replacements:
        if value and len(value) > 2:
            scrubbed = scrubbed.replace(value, "<redacted>")

    # Generic home-directory prefixes for the common platforms. The username is
    # the path segment following the prefix, up to the next separator or
    # whitespace (so trailing usernames without a separator are still redacted).
    scrubbed = _HOME_PATH_RE.sub(lambda m: m.group(1) + "<user>", scrubbed)

    return scrubbed


def _collect_files(
    experiment: "Experiment", content: BugReportContent
) -> List[str]:
    """Resolve the absolute paths of the artifacts selected for inclusion."""
    from glob import glob

    exp_path = str(experiment.path)
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
    if content.include_screenshots:
        for path in glob(os.path.join(exp_path, _SCREENSHOT_GLOB), recursive=True):
            _add(path)
    if content.include_images:
        for pattern in _IMAGE_GLOBS:
            for path in glob(os.path.join(exp_path, pattern), recursive=True):
                _add(path)

    return files


def estimate_bundle_size(
    experiment: Optional["Experiment"], content: BugReportContent
) -> int:
    """Estimate the on-disk size (bytes) of the selected artifacts."""
    if experiment is None:
        return 0
    total = 0
    for path in _collect_files(experiment, content):
        try:
            total += os.path.getsize(path)
        except OSError:
            pass
    return total


def _render_report_text(content: BugReportContent) -> str:
    """Render the human-readable report body shared by the bundle and email."""
    lines = [
        f"# AutoLamella Bug Report: {content.title or '(no title)'}",
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
    experiment: Optional["Experiment"],
    output_dir: Optional[str] = None,
) -> str:
    """Build a scrubbed ``.zip`` bundle for the bug report and return its path.

    Text artifacts (logs, yaml, json, ...) are scrubbed of the user's home
    directory / username before being written. Binary artifacts (images) are
    included verbatim only when explicitly selected.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    zip_name = f"bug-report-autolamella-{timestamp}.zip"

    if output_dir is None:
        # Prefer alongside the experiment; fall back to the log directory.
        if experiment is not None and os.path.isdir(str(experiment.path)):
            output_dir = str(experiment.path)
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

        if experiment is not None:
            exp_path = str(experiment.path)
            for path in _collect_files(experiment, content):
                arcname = os.path.relpath(path, os.path.dirname(exp_path))
                ext = os.path.splitext(path)[1].lower()
                try:
                    if ext in _SCRUBBED_EXTENSIONS:
                        with open(path, "r", encoding="utf-8", errors="replace") as f:
                            zf.writestr(arcname, scrub_text(f.read()))
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
        "_Filed from the AutoLamella Report an Issue dialog. "
        "Do not paste private/experiment data into this public issue — "
        "use the email option to submit data privately._",
    ]
    return "\n".join(lines)


def open_github_issue(content: BugReportContent) -> None:
    """Open a pre-filled GitHub issue in the default browser."""
    params = {
        "title": content.title or "AutoLamella issue",
        "body": _github_issue_body(content),
    }
    url = f"{GITHUB_NEW_ISSUE_URL}?{urlencode(params)}"
    logging.info("Opening GitHub issue in browser.")
    webbrowser.open(url)


def compose_support_email(
    content: BugReportContent, attachment_path: Optional[str]
) -> None:
    """Open a pre-filled email to the support address.

    ``mailto:`` cannot attach files, so the bundle path is included in the body
    with an instruction for the user to attach it manually.
    """
    subject = f"[AutoLamella Bug Report] {content.title or 'Untitled'}"
    body_lines = [_render_report_text(content)]
    if attachment_path:
        body_lines += [
            "",
            "---",
            f"A data bundle was created at:\n{attachment_path}",
            "",
            "Please attach this file to this email before sending.",
        ]
    body = "\n".join(body_lines)

    url = f"mailto:{SUPPORT_EMAIL}?" + urlencode(
        {"subject": subject, "body": body}, quote_via=quote
    )
    logging.info("Opening support email to %s", SUPPORT_EMAIL)
    webbrowser.open(url)


def init_sentry() -> bool:
    """Initialise automatic crash reporting, if enabled and available.

    Inert by default: returns ``False`` unless the user has opted in
    (``reporting.crash_reporting_enabled``), provided a DSN, and installed
    ``sentry-sdk``. Safe to call unconditionally at startup.
    """
    try:
        prefs = fibsem_cfg.load_user_preferences()
        reporting = getattr(prefs, "reporting", None)
        if reporting is None or not getattr(reporting, "crash_reporting_enabled", False):
            return False
        dsn = getattr(reporting, "sentry_dsn", "")
        if not dsn:
            return False

        import sentry_sdk  # noqa: F401 - optional dependency
    except ImportError:
        logging.info("Crash reporting enabled but sentry-sdk is not installed.")
        return False
    except Exception as e:
        logging.warning("Could not initialise crash reporting: %s", e)
        return False

    def _before_send(event, hint):
        try:
            return json.loads(scrub_text(json.dumps(event)))
        except Exception:
            return event

    sentry_sdk.init(
        dsn=dsn,
        release=getattr(fibsem, "__version__", "unknown"),
        send_default_pii=False,
        before_send=_before_send,
    )
    logging.info("Crash reporting initialised.")
    return True

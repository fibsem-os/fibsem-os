"""An HTML page to a PDF, printed by a browser already on the machine (FIB-1036).

Edge ships with Windows 10 and 11, and Chrome or Chromium is on most other
machines. Run headless, the browser prints the page through the page's own
print stylesheet, exactly as its Print button would. The PDF is the page:
there's no second renderer to keep in step, and no new dependency.

The browser runs with a throwaway profile, so it never touches anyone's own,
and with its background networking off: the page is a local file with nothing
to fetch. It isn't waited on to exit, since Chrome on macOS writes the PDF and
then doesn't exit. The PDF is taken once it is complete (it ends ``%%EOF``),
and the browser and everything it started are stopped.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

# A browser to use instead of looking for one: a facility's own, or a test's.
BROWSER_ENV = "FIBSEM_PDF_BROWSER"
TIMEOUT_S = 60.0

_FLAGS = (
    "--headless=new",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-default-apps",
    "--disable-extensions",
    "--metrics-recording-only",
    "--no-pdf-header-footer",
)


class PdfExportError(RuntimeError):
    """No PDF was made: no browser, or the browser didn't write one."""


def find_browser() -> Optional[str]:
    """The browser to print with: ``FIBSEM_PDF_BROWSER`` if it names one that
    exists, else Edge, Chrome or Chromium, where each is usually installed or
    on the PATH. None when there is none."""
    named = os.environ.get(BROWSER_ENV)
    if named:
        return named if Path(named).is_file() else None
    for candidate in _installed():
        if Path(candidate).is_file():
            return candidate
    for command in (
        "msedge",
        "microsoft-edge",
        "microsoft-edge-stable",
        "google-chrome",
        "google-chrome-stable",
        "chromium",
        "chromium-browser",
        "chrome",
    ):
        found = shutil.which(command)
        if found:
            return found
    return None


def html_to_pdf(
    html_path: Union[str, Path],
    pdf_path: Union[str, Path, None] = None,
    *,
    browser: Union[str, Sequence[str], None] = None,
    timeout: float = TIMEOUT_S,
) -> Path:
    """Print ``html_path`` to ``pdf_path`` (beside it, as .pdf, by default) and
    return where it went. ``browser`` is the executable, or a command to run
    with the browser's arguments added; found with :func:`find_browser` when
    None. Raises PdfExportError when no PDF was made, and then leaves none."""
    html_path = Path(html_path).resolve()
    pdf_path = Path(pdf_path) if pdf_path is not None else html_path.with_suffix(".pdf")
    if browser is None:
        browser = find_browser()
        if browser is None:
            raise PdfExportError(
                "no Edge, Chrome or Chromium was found to print the page with"
            )
    command: List[str] = [browser] if isinstance(browser, str) else list(browser)
    part = pdf_path.with_name(pdf_path.name + ".part")
    _remove(part)
    profile = tempfile.mkdtemp(prefix="fibsem-pdf-")
    flags = list(_FLAGS)
    if sys.platform.startswith("linux"):
        # Many Linux machines, CI's among them, forbid the unprivileged user
        # namespaces Chrome's sandbox needs, and it then exits at once. The
        # page is a local file this app wrote, with nothing to fetch.
        flags.append("--no-sandbox")
    command += flags + [
        f"--user-data-dir={profile}",
        f"--print-to-pdf={part}",
        html_path.as_uri(),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **_own_process_group(),
    )
    try:
        deadline = time.monotonic() + timeout
        while not _complete(part):
            if process.poll() is not None and not _complete(part):
                raise PdfExportError(
                    f"the browser exited (code {process.returncode}) without "
                    "writing the PDF"
                )
            if time.monotonic() > deadline:
                raise PdfExportError(
                    f"the browser did not write the PDF within {timeout:.0f} s"
                )
            time.sleep(0.1)
    except BaseException:
        _stop(process)
        _remove(part)
        raise
    finally:
        _stop(process)
        shutil.rmtree(profile, ignore_errors=True)
    _replace(part, pdf_path)
    return pdf_path


def _installed() -> List[str]:
    """Where Edge, Chrome and Chromium install themselves."""
    if sys.platform == "win32":
        roots = [
            os.environ.get(name)
            for name in ("ProgramFiles(x86)", "ProgramFiles", "LocalAppData")
        ]
        return [
            str(Path(root) / relative)
            for relative in (
                "Microsoft/Edge/Application/msedge.exe",
                "Google/Chrome/Application/chrome.exe",
                "Chromium/Application/chrome.exe",
            )
            for root in roots
            if root
        ]
    if sys.platform == "darwin":
        return [
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
    return []


def _own_process_group() -> dict:
    """Start the browser in a group of its own, so it and every process it
    starts can be stopped together."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop(process: subprocess.Popen) -> None:
    """Stop the browser and what it started; nothing if it has exited."""
    if process.poll() is None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(process.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                )
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
        process.wait(timeout=5)


def _complete(path: Path) -> bool:
    """Whether the PDF is written to its end: a PDF closes with ``%%EOF``."""
    try:
        size = path.stat().st_size
        if size < 16:
            return False
        with open(path, "rb") as f:
            f.seek(max(size - 1024, 0))
            return b"%%EOF" in f.read()
    except OSError:
        return False


def _replace(part: Path, final: Path) -> None:
    """Move the finished PDF into place. On Windows the browser's handle can
    outlive it for a moment."""
    for attempt in range(10):
        try:
            os.replace(part, final)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass

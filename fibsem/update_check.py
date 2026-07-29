"""Whether a newer fibsemOS release is available on PyPI.

Deliberately Qt-free and stdlib-only: the UI owns the threading (see
``fibsem/ui/qt/threading.py``), and this module only knows how to ask PyPI, cache
the answer, and decide whether asking is appropriate at all.

Two rules shape everything here:

* **Never block the caller for long, and never raise.** :func:`refresh` is
  blocking and must be run on a worker thread; it has an explicit timeout, since
  an instrument PC behind a firewall that blackholes packets would otherwise
  stall until the OS TCP timeout.
* **A failed check is not an error state.** Offline is the normal condition on an
  air-gapped microscope, so it is indistinguishable from "no check has run": both
  leave the cached status as ``None`` and the UI shows nothing.
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from packaging import version

from fibsem.versioning import get_revision

logger = logging.getLogger(__name__)

PYPI_URL = "https://pypi.org/pypi/{package}/json"

# Long enough for a slow-but-working connection, short enough that a wedged
# network does not keep a worker thread alive for the rest of the session.
REQUEST_TIMEOUT_SECONDS = 5.0

_PACKAGE = "fibsem"

# Populated by refresh(); None means "no successful check", which covers both
# "not run yet" and "could not reach PyPI".
_status: Optional["UpdateStatus"] = None


@dataclass(frozen=True)
class UpdateStatus:
    """The outcome of a successful PyPI lookup."""

    current: str
    latest: str
    update_available: bool


def is_supported() -> bool:
    """Whether an update check could say anything meaningful.

    False for source installs, and no preference or explicit request overrides
    it: someone 48 commits past v0.5.1 being told that 0.5.1 is available is
    noise, and once a newer release lands it becomes advice to ``pip install -U``
    over their editable install.
    """
    return get_revision() is None


def is_enabled() -> bool:
    """Whether to check *automatically*, without the user asking.

    **Opt-in.** These run on instrument PCs, sometimes on institutional networks
    where an unannounced connection at startup is a compliance question, so the
    user has to ask for it (``reporting.update_check_enabled``).

    This governs the background check only. A user-initiated check goes through
    ``refresh(force=True)`` — the click is the consent.
    """
    if not is_supported():
        return False
    try:
        from fibsem.config import load_user_preferences

        return bool(load_user_preferences().reporting.update_check_enabled)
    except Exception:
        # Fail closed. An unreadable preferences file must not produce a network
        # call the user never opted into.
        logger.debug("Could not read update-check preference", exc_info=True)
        return False


def get_status() -> Optional[UpdateStatus]:
    """The cached result, or None if no check has succeeded this session."""
    return _status


def get_available_update() -> Optional[str]:
    """The newer version available, or None.

    None whenever there is nothing to act on — no check has run, the check
    failed, updates are disabled, or the installed version is already current.
    The UI shows an indicator if and only if this returns a version.
    """
    status = _status
    if status is not None and status.update_available:
        return status.latest
    return None


def fetch_stable_versions(package: str = _PACKAGE) -> List[str]:
    """Stable versions on PyPI, newest first. Empty list on any failure.

    Uses ``urllib`` rather than ``requests``: requests is not a core dependency
    (it only appears in the ``server`` extra), so importing it would fail for a
    normal install and this check would silently never work.
    """
    try:
        request = urllib.request.Request(
            PYPI_URL.format(package=package),
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        releases = list(data["releases"].keys())
    except Exception as exc:
        # Debug, not warning: being offline is the expected state on an
        # air-gapped instrument PC, and a warning every launch trains operators
        # to ignore the log.
        logger.debug("Could not fetch PyPI versions for %s: %s", package, exc)
        return []

    stable = []
    for release in releases:
        try:
            parsed = version.parse(release)
        except Exception:
            continue
        if not parsed.is_prerelease and not parsed.is_devrelease:
            stable.append((parsed, release))
    return [release for _, release in sorted(stable, reverse=True)]


def refresh(package: str = _PACKAGE, force: bool = False) -> Optional[UpdateStatus]:
    """Ask PyPI and cache the result. Blocking — run me on a worker thread.

    Returns None (leaving any previous result cached) when the check is disabled
    or could not complete.

    Pass ``force=True`` for a user-initiated check, e.g. an explicit "check for
    updates" click: that is not an unexpected network call, so the opt-in
    preference does not apply to it. Source-install suppression still does,
    since the answer would be meaningless either way.
    """
    global _status

    if not is_supported():
        return None
    if not force and not is_enabled():
        return None

    versions = fetch_stable_versions(package)
    if not versions:
        return None

    import fibsem

    current_raw = getattr(fibsem, "__version__", None) or ""
    try:
        current = version.parse(current_raw)
        latest = version.parse(versions[0])
    except Exception as exc:
        logger.debug("Could not compare versions (%r vs %r): %s",
                     current_raw, versions[0], exc)
        return None

    _status = UpdateStatus(
        current=current_raw,
        latest=versions[0],
        update_available=current < latest,
    )
    logger.debug("Update check: %s", _status)
    return _status


def reset() -> None:
    """Drop the cached status. For tests."""
    global _status
    _status = None

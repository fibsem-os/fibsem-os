"""Which version, and which commit, is actually running.

``fibsem.__version__`` comes from installed package metadata, which for a source
install is whatever ``pyproject.toml`` said at install time — or ``"unknown"``
when the package was never pip-installed. Neither identifies the commit actually
running, which is the common case for users who install from a clone and then
``git pull``. This module adds that, by asking git about the checkout the
package was imported from:

* :func:`get_revision` — ``git describe`` of the running checkout, or ``None``
* :func:`get_branch` — the checked-out branch, or ``None``
* :func:`get_version_string` — packaged version plus revision, for display

Never raises, never runs at import, never depends on the process cwd, and never
writes to the user's repository. Results are cached for the lifetime of the
process, so ``-dirty`` reflects the working tree as it was at the *first* call,
i.e. what the session started with.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    from functools import cache
except ImportError:  # Python < 3.9 fallback
    from functools import lru_cache as cache

_logger = logging.getLogger(__name__)

# Small enough that a wedged git cannot visibly stall window construction, large
# enough to survive a cold spawn behind on-access antivirus.
_GIT_TIMEOUT_SECONDS = 2.0

# These redirect git at a *different* repository and take precedence over the
# cwd we pass, so a process launched from a git hook or pre-commit would report
# the wrong revision. GIT_CEILING_DIRECTORIES is deliberately not scrubbed: it
# only restricts git's upward search, which is harmless here (we have already
# confirmed .git is in cwd) and lets the tests stay hermetic.
_GIT_ENV_OVERRIDES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)

# Windows-only: stops a console window flashing up when a GUI running under
# pythonw.exe shells out. The constant is absent on other platforms, and
# subprocess accepts creationflags=0 everywhere, so no platform branch is needed.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _source_checkout_root() -> Optional[Path]:
    """Return the repository root when running from a source checkout, else None.

    Everything is inside the try: Path.exists() propagates PermissionError (it
    only swallows ENOENT/ENOTDIR/EBADF/ELOOP), and __file__ and resolve() can
    both fail under frozen or embedded interpreters.
    """
    try:
        root = Path(__file__).resolve().parent.parent
        # .git is a file in a git worktree and a directory in a normal clone, so
        # test existence rather than is_dir(). pyproject.toml distinguishes our
        # own checkout from an unrelated repo a wheel happens to be installed in.
        if (root / ".git").exists() and (root / "pyproject.toml").exists():
            return root
    except Exception:
        pass
    return None


def _run_git(*args: str) -> Optional[str]:
    """Run ``git <args>`` in the source checkout, returning None on any failure."""
    root = _source_checkout_root()
    if root is None:
        return None

    try:
        env = {k: v for k, v in os.environ.items() if k not in _GIT_ENV_OVERRIDES}
        # --dirty refreshes the index, which normally takes .git/index.lock and
        # writes the index back: a read-only version lookup would mutate the
        # user's repository and collide with any git they run concurrently.
        env["GIT_OPTIONAL_LOCKS"] = "0"
        result = subprocess.run(
            ["git", *args],
            cwd=str(root),
            capture_output=True,
            # Not text=True: that decodes with the locale's preferred encoding,
            # which raises UnicodeDecodeError on a legacy Windows code page when
            # a branch name is non-ASCII.
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SECONDS,
            env=env,
            creationflags=_NO_WINDOW,
        )
    except Exception as exc:
        # git missing, spawn blocked, timeout, cwd vanished, ... Debug rather
        # than warning: for a normal wheel install this path is the expected one,
        # and a startup warning on every launch trains operators to ignore logs.
        _logger.debug("git %s failed: %s", " ".join(args), exc)
        return None

    if result.returncode != 0:
        _logger.debug(
            "git %s exited %s: %s",
            " ".join(args),
            result.returncode,
            result.stderr.strip(),
        )
        return None

    return result.stdout.strip() or None


@cache
def _describe() -> Optional[str]:
    # --always falls back to a bare short sha in a shallow or tagless clone,
    # which is what CI produces (actions/checkout defaults to fetch-depth 1).
    # Not --broken: that needs git >= 2.13, and on older git the whole command
    # fails, losing the revision entirely.
    #
    # Cached because FibsemImageMetadata builds a FibsemExperimentRef for every
    # acquired image — without this that would be one git fork per image.
    return _run_git("describe", "--tags", "--always", "--dirty")


@cache
def _branch() -> Optional[str]:
    # symbolic-ref rather than `rev-parse --abbrev-ref HEAD`: on a detached HEAD
    # this exits non-zero with empty output, which _run_git already handles,
    # instead of printing the literal string "HEAD".
    return _run_git("symbolic-ref", "--quiet", "--short", "HEAD")


def get_revision() -> Optional[str]:
    """Return ``git describe`` of the running checkout, or None.

    For example ``"v0.5.1-48-g4cd11d9c"``, ``"v0.5.1-48-g4cd11d9c-dirty"``, or a
    bare ``"4cd11d9c"`` in a clone with no tags. None for a wheel install, or on
    any failure. Cached: the first call forks git, later calls are free.
    """
    return _describe()


def get_branch() -> Optional[str]:
    """Return the checked-out branch, or None (detached HEAD, or not a checkout).

    Deliberately not part of :func:`get_version_string`: it costs a second fork,
    adds nothing to the identity of the running code, and branch names routinely
    embed usernames — so it must not reach the bug report or Sentry.
    """
    return _branch()


def get_version_string() -> str:
    """Return the version for display. Never raises, never returns None.

    ``"0.5.2.dev0 (v0.5.1-48-g4cd11d9c)"`` from a source checkout, plain
    ``"0.5.2.dev0"`` otherwise. The parenthesised form is deliberately not
    PEP 440 — a reader should not be tempted to hand it to
    ``packaging.version.parse``.
    """
    try:
        import fibsem

        version = getattr(fibsem, "__version__", None) or "unknown"
    except Exception:
        version = "unknown"

    # Independent guard: a failure in the git half must never cost the caller the
    # package version it would otherwise have got.
    try:
        revision = _describe()
    except Exception:
        revision = None

    return "{} ({})".format(version, revision) if revision else version

"""The wheel must not ship files the application writes at runtime.

A release that carries the build machine's own ``user-preferences.yaml`` or
``saved-positions.yaml`` does not just add a stray file. Shipped files are listed
in the distribution's ``RECORD``, so pip replaces every user's copy with the build
machine's on the next upgrade, and deletes it on uninstall. Measured on a real
wheel built from this tree: before the fix, a ``user-preferences.yaml`` sitting in
``fibsem/config`` was picked up and shipped.

The mechanism was a ``"*" = ["*.yaml"]`` entry in ``package-data``. A glob there is
matched against whatever is on the build machine's disk, not against what git
tracks, so it swept up exactly the files ``.gitignore`` exists to keep out. There is
no glob in ``package-data`` now -- setuptools_scm's file finder already offers every
tracked file to the build -- and these tests are what keep it that way.

The check is on the *patterns* rather than on what a build happens to produce,
because a clean checkout has none of the runtime files on disk. Tagged releases are
built by the publish workflow from exactly such a checkout, so a reintroduced glob
would sail through there and misfire only on a working tree that has run the
application -- a wheel built by hand for a site, a partner or a demo, which is the
one build path where nobody is watching.
"""

import fnmatch
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Set

import pytest

from fibsem import config as cfg

REPO_ROOT = Path(__file__).resolve().parents[1]


def _git_tracked_names() -> Set[str]:
    """Basenames of every file git tracks, or skip if this is not a checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout, so there is nothing to compare against")
    return {os.path.basename(line) for line in out.splitlines() if line}


def _declared_package_data() -> Dict[str, List[str]]:
    tomllib = pytest.importorskip("tomllib", reason="needs Python 3.11+ to read TOML")
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        config = tomllib.load(fh)
    return config["tool"]["setuptools"].get("package-data", {})


def _runtime_written_names() -> Set[str]:
    """Filenames fibsem writes into its config directory at runtime.

    Read off the ``*_PATH`` constants in fibsem.config rather than listed here, so
    that a path added there is covered without anyone remembering to update this
    test. Tracked files are removed: ``microscope-configuration.yaml`` and
    ``tescan_manipulator.yaml`` are shipped *and* written at runtime, which is its
    own problem (FIB-512) and not one package-data can fix.
    """
    tracked = _git_tracked_names()
    config_dir = os.path.normpath(cfg.CONFIG_PATH)
    names = set()
    for attr in dir(cfg):
        if not attr.endswith("_PATH"):
            continue
        value = getattr(cfg, attr)
        if not isinstance(value, str):
            continue
        if os.path.normpath(os.path.dirname(value)) != config_dir:
            continue
        names.add(os.path.basename(value))
    return names - tracked


def test_runtime_config_files_were_found():
    """Guard the guard: the two tests below are vacuous if this comes back empty."""
    assert _runtime_written_names(), (
        "found no runtime-written paths in fibsem.config, which is not credible -- "
        "the constants were probably renamed away from the *_PATH suffix, and the "
        "packaging checks below are now passing without checking anything"
    )


def test_no_package_data_pattern_matches_a_runtime_file():
    runtime_names = _runtime_written_names()
    offenders = []
    for package, patterns in _declared_package_data().items():
        for pattern in patterns:
            matched = sorted(n for n in runtime_names if fnmatch.fnmatch(n, pattern))
            if matched:
                offenders.append((package, pattern, matched))

    assert not offenders, (
        "package-data patterns that would ship runtime state:\n"
        + "\n".join(
            '  "{}" = ["{}"]  would ship {}'.format(
                package, pattern, ", ".join(matched)
            )
            for package, pattern, matched in offenders
        )
    )


@pytest.mark.skipif(
    not os.environ.get("FIBSEM_TEST_WHEEL"),
    reason="set FIBSEM_TEST_WHEEL=<dir with a built wheel> to check a real artifact",
)
def test_built_wheel_contains_only_tracked_files():
    """Check a built wheel, which is the only place some of this is visible.

    A stale ``build/lib`` re-ships a file that has already been deleted from the
    source tree, with its old contents, so inspecting the source tree cannot tell
    you what a wheel actually carries.
    """
    wheel_dir = Path(os.environ["FIBSEM_TEST_WHEEL"])
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert wheels, "no wheel found in {}".format(wheel_dir)

    tracked = _git_tracked_names()
    with zipfile.ZipFile(wheels[-1]) as wheel:
        shipped = [
            name
            for name in wheel.namelist()
            if not name.endswith("/") and ".dist-info/" not in name
        ]

    untracked = sorted(
        name for name in shipped if os.path.basename(name) not in tracked
    )
    assert not untracked, "wheel ships files git does not track:\n" + "\n".join(
        "  " + name for name in untracked
    )

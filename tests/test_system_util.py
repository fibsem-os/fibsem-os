"""What `fibsem.util.system` reports about the machine it is running on.

The interesting cases are the ones a Windows site hits and a developer's laptop does
not: a directory that has not been created yet (every new experiment), and a volume
that cannot be read at all (a mapped drive that is not mapped today). Both are forced
here rather than waited for.

Run directly:
    python -m pytest tests/test_system_util.py
"""

import os
import shutil

import pytest

from fibsem.util.system import (
    CRITICAL_FREE_SPACE_BYTES,
    LOW_FREE_SPACE_BYTES,
    DiskSpace,
    FreeSpaceLevel,
    classify_free_space,
    directory_size,
    disk_space,
    nearest_existing,
)

# ── nearest_existing ─────────────────────────────────────────────────────


def test_nearest_existing_returns_the_path_itself_when_it_is_there(tmp_path):
    assert nearest_existing(tmp_path) == os.path.abspath(str(tmp_path))


def test_nearest_existing_walks_up_to_the_first_real_directory(tmp_path):
    """The case every new experiment is: the directory is named but not yet made."""
    missing = tmp_path / "AutoLamella-2026-09-21" / "01-tough-goose" / "Alignment"
    assert nearest_existing(missing) == os.path.abspath(str(tmp_path))


def test_nearest_existing_stops_at_the_root_rather_than_looping():
    """An unmapped `Z:` has no existing ancestor at all. The walk has to terminate."""
    root = os.path.abspath(os.sep)
    assert (
        nearest_existing(os.path.join(root, "no-such-directory-here", "nested")) == root
    )


# ── disk_space ───────────────────────────────────────────────────────────


def test_disk_space_reads_a_real_volume(tmp_path):
    space = disk_space(tmp_path)
    assert space is not None
    assert space.total > 0
    assert space.free >= 0
    # used + free need not equal total (reserved blocks), but neither may exceed it.
    assert space.used <= space.total
    assert 0.0 <= space.fraction_used <= 1.0


def test_disk_space_answers_for_a_directory_that_does_not_exist_yet(tmp_path):
    """The new-experiment path. The answer is about the volume, not the directory."""
    space = disk_space(tmp_path / "not-created-yet" / "deeper")
    assert space is not None
    assert space.path == os.path.abspath(str(tmp_path))
    assert space.total == disk_space(tmp_path).total


def test_disk_space_is_none_when_the_volume_cannot_be_read(tmp_path, monkeypatch):
    """A disconnected mapped drive. Every caller is a label; None lets it say nothing."""

    def refuse(path):
        raise OSError(5, "device not ready")

    monkeypatch.setattr(shutil, "disk_usage", refuse)
    assert disk_space(tmp_path) is None


def test_disk_space_is_none_when_no_ancestor_of_the_path_exists(monkeypatch):
    """`Z:\\data\\runs` with no `Z:` at all: nothing up the chain to measure.

    Forced rather than reproduced -- on POSIX the root always exists, so the branch is
    unreachable from a real path here and would only ever run on Windows.
    """
    monkeypatch.setattr("fibsem.util.system.nearest_existing", lambda path: None)
    assert disk_space("Z:\\data\\runs") is None


# ── the bands ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "free, expected",
    [
        (0, FreeSpaceLevel.CRITICAL),
        (CRITICAL_FREE_SPACE_BYTES - 1, FreeSpaceLevel.CRITICAL),
        (CRITICAL_FREE_SPACE_BYTES, FreeSpaceLevel.LOW),
        (LOW_FREE_SPACE_BYTES - 1, FreeSpaceLevel.LOW),
        (LOW_FREE_SPACE_BYTES, FreeSpaceLevel.AMPLE),
        (4_000_000_000_000, FreeSpaceLevel.AMPLE),
    ],
)
def test_classify_free_space_bands(free, expected):
    """10 GB and 20 GB are the boundaries, and each belongs to the band above it."""
    assert classify_free_space(free) is expected


def test_disk_space_level_uses_its_own_free_figure():
    space = DiskSpace(
        path="Z:\\", total=500_000_000_000, used=494_000_000_000, free=6_000_000_000
    )
    assert space.level is FreeSpaceLevel.CRITICAL


def test_fraction_used_of_a_volume_reporting_no_size():
    """Some network shares answer zero for everything rather than failing."""
    assert DiskSpace(path="Z:\\", total=0, used=0, free=0).fraction_used == 0.0


# ── directory_size ───────────────────────────────────────────────────────


def test_directory_size_sums_a_nested_tree(tmp_path):
    (tmp_path / "lamella" / "Alignment").mkdir(parents=True)
    (tmp_path / "ref_start_eb.tif").write_bytes(b"x" * 100)
    (tmp_path / "lamella" / "ref_final_ib.tif").write_bytes(b"x" * 250)
    (tmp_path / "lamella" / "Alignment" / "align.tif").write_bytes(b"x" * 33)
    assert directory_size(tmp_path) == 383


def test_directory_size_of_an_empty_directory(tmp_path):
    assert directory_size(tmp_path) == 0


def test_directory_size_of_a_path_that_is_not_there(tmp_path):
    """Skipped rather than raised: the number is for a label."""
    assert directory_size(tmp_path / "gone") == 0


def test_directory_size_does_not_follow_a_symlink_out_of_the_tree(tmp_path):
    """A link out of the experiment must not charge the experiment for its target."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "big.tif").write_bytes(b"x" * 10_000)

    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "small.tif").write_bytes(b"x" * 7)
    try:
        (experiment / "link").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available here (Windows without developer mode)")

    # The link itself is counted as the entry it is, not as the 10 kB behind it.
    assert directory_size(experiment) < 1_000

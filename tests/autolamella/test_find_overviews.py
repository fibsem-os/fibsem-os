"""Which overviews an experiment keeps, and in what order.

The overview plot dialog opened ``glob("*overview*.tif")[-1]``: the last file the
filesystem happened to list, which is not the most recent, and it never looked under
``grids/``. The PDF report drew the same unsorted list. ``find_overviews`` was already
the experiment replay's answer; these pin what the three now share.
"""

import os
import time

from fibsem.applications.autolamella.structures import find_overviews


def _touch(path, age_s: float = 0.0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb"):
        pass
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))
    return path


def _names(root, found):
    return [os.path.relpath(p, root) for p in found]


def test_oldest_first_by_time_not_by_name(tmp_path):
    """A re-acquired overview whose name sorts first is still the most recent."""
    _touch(tmp_path / "overview-image-2026-07-29_18-03-35.tif", age_s=3600)
    _touch(tmp_path / "overview-image-2026-07-29_09-00-00-reacquired.tif")

    assert _names(tmp_path, find_overviews(tmp_path)) == [
        "overview-image-2026-07-29_18-03-35.tif",
        "overview-image-2026-07-29_09-00-00-reacquired.tif",
    ]


def test_grid_overviews_are_found(tmp_path):
    _touch(tmp_path / "grids" / "grid-a" / "overview_sem" / "overview-13-40-51.tif")

    assert _names(tmp_path, find_overviews(tmp_path)) == [
        os.path.join("grids", "grid-a", "overview_sem", "overview-13-40-51.tif")
    ]


def test_fluorescence_overviews_are_not_beam_overviews(tmp_path):
    """They are a different view of the sample and are drawn by a different reader."""
    _touch(tmp_path / "overview-19-07-45.ome.tiff")

    assert find_overviews(tmp_path) == []


def test_files_with_the_same_time_come_out_in_a_fixed_order(tmp_path):
    """Copying an experiment can give every file the same time."""
    for name in ("overview-b.tif", "overview-a.tif", "overview-c.tif"):
        _touch(tmp_path / name, age_s=10)
    stamp = time.time() - 10
    for name in ("overview-b.tif", "overview-a.tif", "overview-c.tif"):
        os.utime(tmp_path / name, (stamp, stamp))

    assert _names(tmp_path, find_overviews(tmp_path)) == [
        "overview-a.tif",
        "overview-b.tif",
        "overview-c.tif",
    ]


def test_a_string_path_and_an_empty_experiment(tmp_path):
    assert find_overviews(str(tmp_path)) == []

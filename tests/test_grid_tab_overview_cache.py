"""Overview-cache logic for the Grids tab: freshness by (path, mtime) + bound.

The overview load is expensive (full-res median/gaussian filter), so the tab
caches loaded images and re-opens the same one instantly. These tests exercise
the cache helpers directly, without building the (heavy) tab UI.
"""

import os

from fibsem.applications.autolamella.ui.grid_tab import (
    GridTabWidget,
    _OVERVIEW_CACHE_SIZE,
)


def _bare_widget():
    # skip __init__ (which builds the full UI); only the cache helpers are used
    w = GridTabWidget.__new__(GridTabWidget)
    w._overview_cache = {}
    return w


def test_cache_hit_returns_same_image(tmp_path):
    w = _bare_widget()
    p = str(tmp_path / "overview.tif")
    open(p, "wb").write(b"x")
    img = object()
    w._cache_overview(p, img)
    assert w._cached_overview(p) is img


def test_cache_invalidated_when_file_changes(tmp_path):
    w = _bare_widget()
    p = str(tmp_path / "overview.tif")
    open(p, "wb").write(b"x")
    w._cache_overview(p, object())
    st = os.stat(p)
    os.utime(p, (st.st_atime, st.st_mtime + 10))  # a re-stitched overview
    assert w._cached_overview(p) is None


def test_cache_miss_for_unknown_path(tmp_path):
    w = _bare_widget()
    assert w._cached_overview(str(tmp_path / "nope.tif")) is None


def test_cache_is_bounded(tmp_path):
    w = _bare_widget()
    paths = []
    for i in range(_OVERVIEW_CACHE_SIZE + 2):
        p = str(tmp_path / f"ov{i}.tif")
        open(p, "wb").write(b"x")
        w._cache_overview(p, object())
        paths.append(p)
    assert len(w._overview_cache) == _OVERVIEW_CACHE_SIZE
    assert w._cached_overview(paths[-1]) is not None  # most recent kept
    assert w._cached_overview(paths[0]) is None        # oldest evicted

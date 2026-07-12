"""Tests for the concurrent _ImageLoader (pooled background image loading)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402

import fibsem.ui.widgets.lamella_task_image_widget as m  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _fake_load(path, width):
    return np.zeros((2, 2), dtype=np.uint8), 1.0


def test_loads_all_paths_concurrently(qapp, monkeypatch):
    monkeypatch.setattr(m, "_load_and_resize", _fake_load)
    loader = m._ImageLoader(target_width=100, max_threads=4)
    assert loader._pool.maxThreadCount() >= 1
    got = []
    loader.image_loaded.connect(lambda path, arr, px: got.append(path))

    paths = [f"/img{i}.tif" for i in range(8)]
    loader.load(paths)
    loader._pool.waitForDone(5000)  # let all tasks run (without setting _stopped)
    qapp.processEvents()  # deliver any queued emits

    assert sorted(got) == sorted(paths)


def test_failed_loads_are_skipped_not_fatal(qapp, monkeypatch):
    def boom(path, width):
        if "bad" in path:
            raise RuntimeError("decode failed")
        return _fake_load(path, width)

    monkeypatch.setattr(m, "_load_and_resize", boom)
    loader = m._ImageLoader(target_width=100)
    got = []
    loader.image_loaded.connect(lambda path, arr, px: got.append(path))

    loader.load(["/ok1.tif", "/bad.tif", "/ok2.tif"])
    loader._pool.waitForDone(5000)
    qapp.processEvents()

    assert sorted(got) == ["/ok1.tif", "/ok2.tif"]  # bad one skipped, others fine


def test_cancel_is_nonblocking_and_safe(qapp, monkeypatch):
    monkeypatch.setattr(m, "_load_and_resize", _fake_load)
    loader = m._ImageLoader(target_width=100)
    loader.load([f"/x{i}.tif" for i in range(50)])

    loader.cancel()  # drops queued tasks, must return promptly (no wait)
    loader.wait(2000)  # teardown drains the few in-flight tasks


def test_wait_stops_emitting(qapp, monkeypatch):
    monkeypatch.setattr(m, "_load_and_resize", _fake_load)
    loader = m._ImageLoader(target_width=100)
    got = []
    loader.image_loaded.connect(lambda path, arr, px: got.append(path))

    loader.wait()  # set _stopped before any task can emit
    loader.load([f"/y{i}.tif" for i in range(8)])
    loader._pool.waitForDone(5000)
    qapp.processEvents()

    assert got == []  # stopped loader emits nothing

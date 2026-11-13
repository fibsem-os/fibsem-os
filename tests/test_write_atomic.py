"""A file written whole, or not at all."""

import os

import pytest

from fibsem.util import files
from fibsem.util.files import write_atomic


def test_it_writes_the_text_and_leaves_nothing_else(tmp_path):
    path = tmp_path / "experiment.yaml"

    write_atomic(path, "name: one\n")
    write_atomic(path, "name: two\n")

    assert path.read_text() == "name: two\n"
    assert os.listdir(tmp_path) == ["experiment.yaml"]


def test_a_write_cut_short_leaves_the_file_that_was_there(tmp_path, monkeypatch):
    path = tmp_path / "experiment.yaml"
    write_atomic(path, "name: one\n")

    def cut(fd):
        raise OSError("disk full")

    monkeypatch.setattr(files.os, "fsync", cut)
    with pytest.raises(OSError):
        write_atomic(path, "name: two\n")

    assert path.read_text() == "name: one\n"
    assert os.listdir(tmp_path) == ["experiment.yaml"]


def test_a_file_being_read_is_replaced_once_the_reader_lets_go(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    replace, calls = os.replace, []

    def busy_twice(src, dst):
        calls.append(src)
        if len(calls) < 3:
            raise PermissionError("in use")
        replace(src, dst)

    monkeypatch.setattr(files.os, "replace", busy_twice)
    monkeypatch.setattr(files, "REPLACE_WAIT_S", 0)

    write_atomic(path, "{}")

    assert path.read_text() == "{}" and len(calls) == 3


def test_a_file_that_stays_in_use_raises_and_keeps_the_old_one(tmp_path, monkeypatch):
    path = tmp_path / "session.json"
    path.write_text("old")

    def busy(src, dst):
        raise PermissionError("in use")

    monkeypatch.setattr(files.os, "replace", busy)
    monkeypatch.setattr(files, "REPLACE_WAIT_S", 0)
    monkeypatch.setattr(files, "REPLACE_TRIES", 3)

    with pytest.raises(PermissionError):
        write_atomic(path, "new")

    assert path.read_text() == "old"
    assert os.listdir(tmp_path) == ["session.json"]

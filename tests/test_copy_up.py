"""Copying an experiment's folder up to a store, and the folder store."""

import os

import pytest

from fibsem.copy_up import copy_up
from fibsem.storage import FolderStore, valid_name


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def experiment(tmp_path):
    folder = tmp_path / "data" / "grid-1"
    write(folder / "experiment.yaml", "name: grid-1")
    write(folder / "lamella-1" / "ref.tif", "image")
    return folder


def test_a_copy_sends_everything_once_and_then_only_what_changed(tmp_path, experiment):
    store = FolderStore(tmp_path / "shared")
    first = copy_up(experiment, store, "fib-1/grid-1")
    assert (first.sent, first.same, first.complete) == (2, 0, True)
    assert store.read("fib-1/grid-1/lamella-1/ref.tif") == b"image"

    assert copy_up(experiment, store, "fib-1/grid-1").sent == 0

    write(experiment / "experiment.yaml", "name: grid-1\nmore: true")
    write(experiment / "lamella-2" / "ref.tif", "new")
    again = copy_up(experiment, store, "fib-1/grid-1")
    assert (again.sent, again.same) == (2, 1)
    assert store.read("fib-1/grid-1/experiment.yaml") == b"name: grid-1\nmore: true"


def test_a_copy_never_removes_from_the_store(tmp_path, experiment):
    store = FolderStore(tmp_path / "shared")
    copy_up(experiment, store, "grid-1")
    (experiment / "lamella-1" / "ref.tif").unlink()
    copy_up(experiment, store, "grid-1")
    assert store.exists("grid-1/lamella-1/ref.tif")


def test_a_copy_that_is_stopped_carries_on_where_it_stopped(tmp_path, experiment):
    store = FolderStore(tmp_path / "shared")
    asked = []
    stopped = copy_up(
        experiment, store, "grid-1", stop=lambda: asked.append(1) or len(asked) > 1
    )
    assert stopped.stopped and stopped.sent == 1 and not stopped.complete
    rest = copy_up(experiment, store, "grid-1")
    assert (rest.sent, rest.same, rest.complete) == (1, 1, True)


def test_a_file_that_cannot_be_sent_is_named_and_the_rest_still_goes(
    tmp_path, experiment
):
    class Flaky(FolderStore):
        def save(self, name, source):
            if name.endswith("ref.tif"):
                raise OSError("the drive dropped")
            return super().save(name, source)

    done = copy_up(experiment, Flaky(tmp_path / "shared"), "grid-1")
    assert done.failed == ["grid-1/lamella-1/ref.tif"] and done.sent == 1
    # Nothing half-written is left where the file goes.
    assert not list((tmp_path / "shared").rglob("*.part"))


def test_a_copy_keeps_the_modified_time_so_the_next_one_sees_it_as_the_same(
    tmp_path, experiment
):
    os.utime(experiment / "experiment.yaml", (1_700_000_000, 1_700_000_000))
    store = FolderStore(tmp_path / "shared")
    copy_up(experiment, store, "grid-1")
    assert store.stat("grid-1/experiment.yaml").mtime == 1_700_000_000
    assert [kept.name for kept in store.list("grid-1")] == [
        "grid-1/experiment.yaml",
        "grid-1/lamella-1/ref.tif",
    ]


@pytest.mark.parametrize("name", ["../x", "/x", "a/../b", "c:/x", "a\\b", ""])
def test_a_store_refuses_a_name_that_leaves_it(tmp_path, name):
    assert not valid_name(name)
    with pytest.raises(ValueError):
        FolderStore(tmp_path).exists(name)


def test_a_shared_folder_that_is_not_there_says_so(tmp_path):
    assert not FolderStore(tmp_path / "unmounted").available()

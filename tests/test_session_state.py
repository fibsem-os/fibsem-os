"""The session-state store: one file per instrument configuration.

Nothing is wired to it yet. These pin the contract the sections moving into it will
rely on: scoped by configuration file name, read-only unless asked, each writer
replaces only its own section, an unreadable file is never overwritten, and an old
file is imported once.
"""

import os
import stat
import threading
from pathlib import Path

import numpy as np
import pytest
import yaml

from fibsem.session_state import (
    SESSION_STATE_VERSION,
    SessionState,
    session_state_for,
    session_state_path,
)


@pytest.fixture
def store(tmp_path) -> SessionState:
    return SessionState(
        "/somewhere/tfs-hydra-configuration.yaml",
        writable=True,
        directory=str(tmp_path),
    )


def _on_disk(store: SessionState) -> dict:
    return yaml.safe_load(store.path.read_text())


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_the_file_is_named_for_the_configuration_file(tmp_path):
    path = session_state_path("/a/b/bay-2.yml", directory=str(tmp_path))
    assert path == tmp_path / "bay-2.yaml"


def test_two_configurations_do_not_share_state(tmp_path):
    hydra = SessionState("hydra.yaml", writable=True, directory=str(tmp_path))
    arctis = SessionState("arctis.yaml", writable=True, directory=str(tmp_path))

    hydra.save_section("holder_occupancy", {"Slot-01": {"name": "G1"}})

    assert arctis.load_section("holder_occupancy") is None


def test_a_session_not_started_from_a_file_reads_empty_and_writes_nothing(tmp_path):
    store = SessionState(None, writable=True, directory=str(tmp_path))

    assert store.load_section("fm", default={}) == {}
    assert store.save_section("fm", {"x": 1}) is False
    assert list(tmp_path.iterdir()) == []


def test_the_microscope_s_configuration_path_picks_the_file(tmp_path, monkeypatch):
    import fibsem.session_state as module

    monkeypatch.setattr(module, "SESSION_STATE_DIRECTORY", str(tmp_path))

    class _Microscope:
        configuration_path = "/x/site.yaml"

    assert session_state_for(_Microscope()).path == tmp_path / "site.yaml"


# ---------------------------------------------------------------------------
# Scripts read, the application writes
# ---------------------------------------------------------------------------


def test_a_store_is_read_only_unless_asked(tmp_path):
    reader = SessionState("site.yaml", directory=str(tmp_path))

    assert reader.save_section("fm", {"x": 1}) is False
    assert not (tmp_path / "site.yaml").exists()


def test_a_read_only_store_still_reads(tmp_path):
    SessionState("site.yaml", writable=True, directory=str(tmp_path)).save_section(
        "saved_positions", [{"name": "p1"}]
    )
    reader = SessionState("site.yaml", directory=str(tmp_path))

    assert reader.load_section("saved_positions") == [{"name": "p1"}]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_a_missing_section_reads_as_the_default(store):
    assert store.load_section("fm") is None
    assert store.load_section("fm", default={}) == {}


def test_a_section_round_trips_and_the_version_is_written(store):
    store.save_section("fm", {"working": {"exposure": 0.1}})

    assert store.load_section("fm") == {"working": {"exposure": 0.1}}
    assert _on_disk(store)["version"] == SESSION_STATE_VERSION
    assert list(_on_disk(store))[0] == "version"


def test_writing_one_section_leaves_the_others(store):
    store.save_section("fm", {"a": 1})
    store.save_section("holder_occupancy", {"Slot-01": None})
    store.save_section("fm", {"a": 2})

    assert _on_disk(store) == {
        "version": SESSION_STATE_VERSION,
        "fm": {"a": 2},
        "holder_occupancy": {"Slot-01": None},
    }


def test_a_section_this_version_does_not_know_is_carried_across(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text(yaml.safe_dump({"version": 1, "from_the_future": [1, 2]}))

    store.save_section("fm", {"a": 1})

    assert _on_disk(store)["from_the_future"] == [1, 2]


def test_numpy_values_are_written_as_plain_yaml(store):
    """`safe_load` refuses numpy tags; one would make the whole file unreadable."""
    store.save_section("fm", {"exposure": np.float64(0.25), "shape": np.array([2, 3])})

    assert "numpy" not in store.path.read_text()
    assert store.load_section("fm") == {"exposure": 0.25, "shape": [2, 3]}


def test_writers_on_different_threads_keep_each_other_s_sections(store):
    """The FM autosave and the occupancy writer share the file."""

    def write(name):
        for i in range(50):
            store.save_section(name, {"i": i})

    threads = [threading.Thread(target=write, args=(n,)) for n in ("fm", "occupancy")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert _on_disk(store)["fm"] == {"i": 49}
    assert _on_disk(store)["occupancy"] == {"i": 49}


def test_a_dotted_name_writes_one_key_and_keeps_its_siblings(store):
    """The FM autosave and the recent-channels writer share the `fm` section."""
    store.save_section("fm.working", {"channels": ["GFP"]})
    store.save_section("fm.recent_channels", [{"name": "DAPI"}])
    store.save_section("fm.working", {"channels": ["RFP"]})

    assert _on_disk(store)["fm"] == {
        "working": {"channels": ["RFP"]},
        "recent_channels": [{"name": "DAPI"}],
    }
    assert store.load_section("fm.recent_channels") == [{"name": "DAPI"}]
    assert store.load_section("fm.nothing_here", default=[]) == []


def test_a_dotted_name_under_a_non_mapping_replaces_it(store):
    store.save_section("fm", "not a mapping")
    store.save_section("fm.working", {"a": 1})

    assert store.load_section("fm") == {"working": {"a": 1}}


# ---------------------------------------------------------------------------
# An unreadable file
# ---------------------------------------------------------------------------


def test_an_unreadable_file_reads_empty_and_is_set_aside_not_overwritten(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("fm: [unclosed\n")

    assert store.load_section("fm", default={}) == {}

    store.save_section("fm", {"a": 1})

    kept = [p for p in store.path.parent.iterdir() if ".unreadable-" in p.name]
    assert len(kept) == 1
    assert kept[0].read_text() == "fm: [unclosed\n"
    assert _on_disk(store)["fm"] == {"a": 1}


def test_a_file_that_is_not_a_mapping_is_treated_as_unreadable(store):
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text("- just\n- a list\n")

    assert store.load_section("fm") is None


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_the_file_is_readable_by_the_next_operator(store):
    """Created like any other file, under the umask -- not private to the account
    that happened to write it, as a `mkstemp` file would be."""
    umask = os.umask(0)
    os.umask(umask)

    store.save_section("fm", {"a": 1})

    assert stat.S_IMODE(store.path.stat().st_mode) == 0o666 & ~umask


def test_no_temporary_file_is_left_behind(store):
    store.save_section("fm", {"a": 1})

    assert sorted(p.name for p in store.path.parent.iterdir()) == [store.path.name]


# ---------------------------------------------------------------------------
# Importing from the files that came before
# ---------------------------------------------------------------------------


def test_an_absent_section_is_imported_once_and_kept(store):
    calls = []

    def migrate():
        calls.append(1)
        return {"from": "old file"}

    assert store.load_section("fm", migrate=migrate) == {"from": "old file"}
    assert store.load_section("fm", migrate=migrate) == {"from": "old file"}
    assert calls == [1]
    assert _on_disk(store)["fm"] == {"from": "old file"}


def test_a_read_only_store_imports_without_writing(tmp_path):
    reader = SessionState("site.yaml", directory=str(tmp_path))

    assert reader.load_section("fm", migrate=lambda: {"a": 1}) == {"a": 1}
    assert not (tmp_path / "site.yaml").exists()


def test_nothing_to_import_gives_the_default_and_writes_nothing(store):
    assert store.load_section("fm", default={}, migrate=lambda: None) == {}
    assert not store.path.exists()


def test_an_import_that_fails_is_treated_as_finding_nothing(store):
    def broken():
        raise OSError("old file unreadable")

    assert store.load_section("fm", default={}, migrate=broken) == {}


def test_a_present_section_is_never_re_imported(store):
    store.save_section("fm", {"current": True})

    def migrate():
        raise AssertionError("imported over an existing section")

    assert store.load_section("fm", migrate=migrate) == {"current": True}


def test_the_session_directory_is_not_tracked_by_git():
    """The files are written inside the checkout as the operator works; without the
    ignore rule every site's checkout would show them as changes."""
    import subprocess

    import fibsem.config as cfg

    probe = Path(cfg.CONFIG_PATH) / "session" / "any-configuration.yaml"
    result = subprocess.run(
        ["git", "check-ignore", "-q", str(probe)],
        cwd=cfg.BASE_PATH,
        capture_output=True,
    )
    if result.returncode == 128:
        pytest.skip("not a git checkout")
    assert result.returncode == 0

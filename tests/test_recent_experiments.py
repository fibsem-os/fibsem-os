"""Tests for the recent-experiments quick-select config helpers."""

import os

import pytest
import yaml

from fibsem import config as cfg


@pytest.fixture
def prefs_env(tmp_path, monkeypatch):
    """Point the preferences helpers at an isolated temp file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(config_dir / "user-preferences.yaml"))
    return tmp_path


def _write_experiment(dir_path, name="exp", created_at=1000.0, num_positions=0) -> str:
    """Create an experiment.yaml on disk and return its path."""
    os.makedirs(dir_path, exist_ok=True)
    yaml_path = os.path.join(dir_path, "experiment.yaml")
    with open(yaml_path, "w") as f:
        yaml.safe_dump(
            {
                "name": name,
                "created_at": created_at,
                "positions": [{"id": i} for i in range(num_positions)],
            },
            f,
        )
    return yaml_path


def test_record_persists_and_dedups_most_recent_first(prefs_env):
    a = _write_experiment(prefs_env / "a")
    b = _write_experiment(prefs_env / "b")

    cfg.record_recent_experiment(a)
    cfg.record_recent_experiment(b)
    cfg.record_recent_experiment(a)  # re-record moves a back to front

    recent = cfg.load_user_preferences().experiment.recent_experiments
    assert recent == [os.path.normpath(a), os.path.normpath(b)]


def test_record_truncates_to_max(prefs_env):
    for i in range(cfg.MAX_RECENT_EXPERIMENTS + 5):
        cfg.record_recent_experiment(_write_experiment(prefs_env / f"e{i}"))

    recent = cfg.load_user_preferences().experiment.recent_experiments
    assert len(recent) == cfg.MAX_RECENT_EXPERIMENTS
    # Most recently recorded is first
    assert recent[0] == os.path.normpath(str(prefs_env / f"e{cfg.MAX_RECENT_EXPERIMENTS + 4}" / "experiment.yaml"))


def test_record_ignores_empty_path(prefs_env):
    cfg.record_recent_experiment("")
    assert cfg.load_user_preferences().experiment.recent_experiments == []


def test_peek_reads_display_metadata(prefs_env):
    path = _write_experiment(prefs_env / "meta", name="cool-experiment", created_at=1234.0, num_positions=3)
    info = cfg._peek_experiment_yaml(path)
    assert info.name == "cool-experiment"
    assert info.created_at == 1234.0
    assert info.num_lamella == 3
    assert info.exists is True
    assert info.available is True


def test_peek_missing_file(prefs_env):
    path = str(prefs_env / "gone" / "experiment.yaml")
    info = cfg._peek_experiment_yaml(path)
    assert info.exists is False
    assert info.available is False
    assert info.name == "gone"  # falls back to parent dir name


def test_peek_corrupt_file_is_unavailable_but_exists(prefs_env):
    dir_path = prefs_env / "corrupt"
    os.makedirs(dir_path)
    yaml_path = os.path.join(dir_path, "experiment.yaml")
    with open(yaml_path, "w") as f:
        f.write("{ this is: not: valid yaml ]")

    info = cfg._peek_experiment_yaml(yaml_path)
    assert info.exists is True       # kept, not silently pruned
    assert info.available is False   # flagged so the UI can grey it out


def test_get_recent_prunes_missing_and_persists(prefs_env):
    good = _write_experiment(prefs_env / "good")
    missing = str(prefs_env / "missing" / "experiment.yaml")
    cfg.record_recent_experiment(good)
    cfg.record_recent_experiment(missing)

    infos = cfg.get_recent_experiments(prune_missing=True)

    assert [i.path for i in infos] == [os.path.normpath(good)]
    # Pruned list was persisted back to disk
    assert cfg.load_user_preferences().experiment.recent_experiments == [os.path.normpath(good)]


def test_get_recent_keeps_corrupt_entries(prefs_env):
    dir_path = prefs_env / "corrupt"
    os.makedirs(dir_path)
    yaml_path = os.path.join(dir_path, "experiment.yaml")
    with open(yaml_path, "w") as f:
        f.write(": : :")
    cfg.record_recent_experiment(yaml_path)

    infos = cfg.get_recent_experiments(prune_missing=True)

    assert len(infos) == 1
    assert infos[0].available is False
    # corrupt-but-present entry is not pruned from preferences
    assert cfg.load_user_preferences().experiment.recent_experiments == [os.path.normpath(yaml_path)]


def test_add_recent_experiment_mutates_without_saving(prefs_env):
    prefs = cfg.load_user_preferences()
    a = _write_experiment(prefs_env / "a")

    cfg.add_recent_experiment(prefs, a)

    assert prefs.experiment.recent_experiments == [os.path.normpath(a)]
    # Nothing was written to disk (no save called)
    assert not os.path.exists(cfg.USER_PREFERENCES_PATH)


def test_backward_compat_missing_key(prefs_env):
    # Legacy prefs file without the recent_experiments key
    with open(cfg.USER_PREFERENCES_PATH, "w") as f:
        yaml.safe_dump({"experiment": {"user": "someone"}}, f)

    prefs = cfg.load_user_preferences()
    assert prefs.experiment.user == "someone"
    assert prefs.experiment.recent_experiments == []

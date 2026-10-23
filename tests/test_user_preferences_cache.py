"""load_user_preferences parses the file once until it changes."""

import os

import pytest
import yaml

from fibsem import config as cfg


@pytest.fixture
def prefs_path(tmp_path, monkeypatch):
    path = tmp_path / "user-preferences.yaml"
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(path))
    return path


def _count_parses(monkeypatch) -> list:
    calls = []
    real = cfg.yaml.safe_load

    def counting(stream):
        calls.append(stream)
        return real(stream)

    monkeypatch.setattr(cfg.yaml, "safe_load", counting)
    return calls


def test_an_unchanged_file_is_parsed_once(prefs_path, monkeypatch):
    prefs_path.write_text(yaml.safe_dump({"experiment": {"user": "someone"}}))
    calls = _count_parses(monkeypatch)

    first = cfg.load_user_preferences()
    second = cfg.load_user_preferences()

    assert first.shared.user == second.shared.user == "someone"
    assert len(calls) == 1


def test_each_call_gets_its_own_copy(prefs_path):
    prefs_path.write_text(yaml.safe_dump({"experiment": {"user": "someone"}}))

    cfg.load_user_preferences().shared.user = "changed but not saved"

    assert cfg.load_user_preferences().shared.user == "someone"


def test_a_save_is_seen_by_the_next_load(prefs_path):
    preferences = cfg.load_user_preferences()
    preferences.shared.user = "saved"
    cfg.save_user_preferences(preferences)

    assert cfg.load_user_preferences().shared.user == "saved"


def test_an_edit_from_outside_is_seen(prefs_path):
    prefs_path.write_text(yaml.safe_dump({"experiment": {"user": "before"}}))
    assert cfg.load_user_preferences().shared.user == "before"

    prefs_path.write_text(yaml.safe_dump({"experiment": {"user": "after, longer"}}))
    stat = os.stat(prefs_path)
    os.utime(prefs_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))

    assert cfg.load_user_preferences().shared.user == "after, longer"


def test_a_missing_file_reads_as_defaults(prefs_path):
    assert cfg.load_user_preferences() == cfg.UserPreferences()

"""The preferences file by owner (FIB-1538): shared, qt and install.

An older file, sectioned by topic, must keep loading with every setting it chose:
each key moves to its new section, what the application remembered (recents, the
guided setup dismissed, the History filter) moves to the per-account session state,
and the retired keys are dropped.
"""

import yaml

from fibsem import config as cfg
from fibsem.session_state import user_state

OLDER_FILE = {
    "display": {
        "sound_enabled": True,
        "border_enabled": False,
        "toast_problems_only": True,
        "dev_mode": True,
        "lamella_card_mode": "compact",
        "guided_setup_dismissed": True,
        "info_bar_fields": {"SEM": ["hfw"]},
        "history_filter": {"status": "failed"},
    },
    "features": {"coincidence_milling_enabled": True, "connection_chip": True},
    "movement": {
        "acquire_sem_after_stage_movement": False,
        "acquire_fib_after_stage_movement": False,
    },
    "experiment": {
        "default_experiment_directory": "/data/experiments",
        "default_protocol_path": "/data/protocol.yaml",
        "last_experiment_path": "/data/experiments/last",
        "recent_experiments": ["/data/experiments/last/experiment.yaml"],
        "user": "user",
        "project": "project",
        "organisation": "organisation",
    },
    "reporting": {
        "contact_email": "someone@example.com",
        "crash_reporting_enabled": True,
        "sentry_dsn": "https://dsn.example.com/1",
        "update_check_enabled": True,
    },
    "app_server": {"enabled": True, "microscope": True, "allow_local_agents": True},
    "hooks": [{"type": "NotificationHook", "name": "custom"}],
}


def test_an_older_file_keeps_every_choice():
    prefs = cfg.UserPreferences.from_dict(OLDER_FILE)

    assert prefs.shared == cfg.SharedPreferences(
        info_bar_fields={"SEM": ["hfw"]},
        default_experiment_directory="/data/experiments",
        default_protocol_path="/data/protocol.yaml",
        user="user",
        project="project",
        organisation="organisation",
    )
    assert prefs.qt == cfg.QtPreferences(
        sound_enabled=True,
        border_enabled=False,
        toast_problems_only=True,
        dev_mode=True,
        lamella_card_mode="compact",
        acquire_sem_after_stage_movement=False,
        acquire_fib_after_stage_movement=False,
        contact_email="someone@example.com",
    )
    assert prefs.install.update_check_enabled is True
    assert prefs.install.app_server == cfg.AppServerPreferences(
        enabled=True, microscope=True, allow_local_agents=True
    )


def test_the_retired_keys_are_not_written_back():
    """Never reused either (FIB-781): a name an older file carries means what it
    meant then."""
    written = str(cfg.UserPreferences.from_dict(OLDER_FILE).to_dict())

    for retired in (
        "coincidence_milling_enabled",
        "connection_chip",
        "last_experiment_path",
        "crash_reporting_enabled",
        "sentry_dsn",
        "hooks",
        "recent_experiments",
        "guided_setup_dismissed",
        "history_filter",
    ):
        assert retired not in written, retired


def test_this_shape_wins_over_an_older_one():
    both = {
        "display": {"sound_enabled": False},
        "qt": {"sound_enabled": True},
        "app_server": {"enabled": False},
        "install": {"app_server": {"enabled": True}},
    }

    prefs = cfg.UserPreferences.from_dict(both)

    assert prefs.qt.sound_enabled is True
    assert prefs.install.app_server.enabled is True


def test_the_oldest_flat_file_still_loads():
    flat = {"acquire_sem_after_stage_movement": False, "experiment_directory": "/d"}

    prefs = cfg.UserPreferences.from_dict(flat)

    assert prefs.qt.acquire_sem_after_stage_movement is False
    assert prefs.shared.default_experiment_directory == "/d"


def test_what_was_remembered_moves_to_the_session_state(tmp_path, monkeypatch):
    path = tmp_path / "user-preferences.yaml"
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(path))
    path.write_text(yaml.safe_dump(OLDER_FILE))

    prefs = cfg.load_user_preferences()

    state = user_state()
    assert state.load_section("recent_experiments") == [
        "/data/experiments/last/experiment.yaml"
    ]
    assert state.load_section("guided_setup_dismissed") is True
    assert state.load_section("history_filter") == {"status": "failed"}
    # The next save drops them from the preferences file, and the state keeps them.
    cfg.save_user_preferences(prefs)
    assert "recent_experiments" not in path.read_text()
    assert user_state().load_section("guided_setup_dismissed") is True


def test_the_session_state_is_not_overwritten_by_an_older_file(tmp_path, monkeypatch):
    """A section already there was written since; the older file's copy is stale."""
    path = tmp_path / "user-preferences.yaml"
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(tmp_path))
    monkeypatch.setattr(cfg, "USER_PREFERENCES_PATH", str(path))
    user_state(writable=True).save_section("recent_experiments", ["/newer.yaml"])
    path.write_text(yaml.safe_dump(OLDER_FILE))

    cfg.load_user_preferences()

    assert user_state().load_section("recent_experiments") == ["/newer.yaml"]

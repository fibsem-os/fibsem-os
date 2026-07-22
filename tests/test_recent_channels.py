"""Tests for the recently-used FM channel settings config helpers."""

import os

import pytest
import yaml

from fibsem import config as cfg
from fibsem.fm import config as fm_config
from fibsem.fm.structures import ChannelSettings


@pytest.fixture
def recents_env(tmp_path, monkeypatch):
    """Point the recent-channels helpers at an isolated temp file."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(cfg, "CONFIG_PATH", str(config_dir))
    monkeypatch.setattr(
        cfg, "FM_RECENT_CHANNELS_PATH", str(config_dir / "fm-recent-channels.yaml")
    )
    return tmp_path


def _channel(name="DAPI", excitation=405.0, emission=450.0, **kwargs) -> ChannelSettings:
    return ChannelSettings(
        name=name, excitation_wavelength=excitation, emission_wavelength=emission, **kwargs
    )


def test_load_missing_file_returns_empty(recents_env):
    assert fm_config.load_recent_channels() == []


def test_record_inserts_most_recent_first(recents_env):
    fm_config.record_recent_channels(_channel("DAPI", 405.0))
    fm_config.record_recent_channels(_channel("GFP", 488.0, 520.0))

    recents = fm_config.load_recent_channels()
    assert [ch.name for ch in recents] == ["GFP", "DAPI"]


def test_record_list_preserves_channel_order(recents_env):
    fm_config.record_recent_channels(_channel("Old", 550.0))
    fm_config.record_recent_channels(
        [_channel("DAPI", 405.0), _channel("GFP", 488.0, 520.0)]
    )

    recents = fm_config.load_recent_channels()
    assert [ch.name for ch in recents] == ["DAPI", "GFP", "Old"]


def test_reuse_replaces_entry_and_moves_to_front(recents_env):
    fm_config.record_recent_channels(_channel("DAPI", 405.0, power=0.01))
    fm_config.record_recent_channels(_channel("GFP", 488.0, 520.0))
    # same name with tweaked power -> replace, move to front
    fm_config.record_recent_channels(_channel("DAPI", 405.0, power=0.05))

    recents = fm_config.load_recent_channels()
    assert [ch.name for ch in recents] == ["DAPI", "GFP"]
    assert recents[0].power == 0.05


def test_same_name_different_wavelength_replaces(recents_env):
    # keyed on name only: re-using a name overwrites its stored settings
    fm_config.record_recent_channels(_channel("DAPI", 405.0))
    fm_config.record_recent_channels(_channel("DAPI", 488.0))

    recents = fm_config.load_recent_channels()
    assert len(recents) == 1
    assert recents[0].excitation_wavelength == 488.0


def test_different_name_is_separate_entry(recents_env):
    fm_config.record_recent_channels(_channel("DAPI", 405.0))
    fm_config.record_recent_channels(_channel("Hoechst", 405.0))

    assert len(fm_config.load_recent_channels()) == 2


def test_record_truncates_to_max(recents_env):
    for i in range(fm_config.MAX_RECENT_CHANNELS + 5):
        fm_config.record_recent_channels(_channel(f"Channel-{i}", 400.0 + i))

    recents = fm_config.load_recent_channels()
    assert len(recents) == fm_config.MAX_RECENT_CHANNELS
    assert recents[0].name == f"Channel-{fm_config.MAX_RECENT_CHANNELS + 4}"


def test_identical_record_skips_rewrite(recents_env):
    channel = _channel("DAPI", 405.0)
    fm_config.record_recent_channels(channel)
    mtime = os.path.getmtime(cfg.FM_RECENT_CHANNELS_PATH)
    os.utime(cfg.FM_RECENT_CHANNELS_PATH, (mtime - 100, mtime - 100))

    fm_config.record_recent_channels(channel)
    assert os.path.getmtime(cfg.FM_RECENT_CHANNELS_PATH) == mtime - 100


def test_corrupted_file_returns_empty(recents_env):
    with open(cfg.FM_RECENT_CHANNELS_PATH, "w") as f:
        f.write("{not: valid: yaml: [")

    assert fm_config.load_recent_channels() == []


def test_malformed_entry_is_skipped(recents_env):
    with open(cfg.FM_RECENT_CHANNELS_PATH, "w") as f:
        yaml.safe_dump(
            [
                _channel("DAPI", 405.0).to_dict(),
                {"bogus_field": 1},
                "not-a-dict",
            ],
            f,
        )

    recents = fm_config.load_recent_channels()
    assert [ch.name for ch in recents] == ["DAPI"]


def test_round_trip_preserves_emission_types(recents_env):
    fm_config.record_recent_channels(
        [
            _channel("Reflection", 550.0, emission=None),
            _channel("Filter", 488.0, emission="LP510"),
        ]
    )

    recents = fm_config.load_recent_channels()
    assert recents[0].emission_wavelength is None
    assert recents[1].emission_wavelength == "LP510"

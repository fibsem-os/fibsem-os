"""Round-trip for the FM configuration persistence (fibsem/fm/config.py)."""

from pathlib import Path

import pytest
import yaml

import fibsem.config as fibsem_cfg
from fibsem.fm.config import load_fm_configuration, save_fm_configuration
from fibsem.fm.structures import (
    CameraImageTransform,
    CameraSettings,
    ChannelSettings,
    FluorescenceConfiguration,
    OverviewParameters,
    ZParameters,
)
from fibsem.session_state import SessionState


def _make_config() -> FluorescenceConfiguration:
    return FluorescenceConfiguration(
        channel_settings=[ChannelSettings(name="GFP", exposure_time=0.05)],
        z_parameters=ZParameters(),
        overview_parameters=OverviewParameters(),
        camera_settings=CameraSettings(transform=CameraImageTransform.FLIP_XY),
        focus_position=1.1e-3,
        limit_position=3.3e-3,
    )


@pytest.fixture
def store(tmp_path, monkeypatch) -> SessionState:
    """A writable store in a temp directory; the old working-state file points at a
    temp path, so nothing is imported from the real one."""
    monkeypatch.setattr(
        fibsem_cfg, "FM_CONFIGURATION_PATH", str(tmp_path / "fm-configuration.yaml")
    )
    return SessionState("site.yaml", writable=True, directory=str(tmp_path / "session"))


def test_load_configuration_missing_returns_none(store):
    assert load_fm_configuration(store) is None


def test_configuration_roundtrip(store):
    save_fm_configuration(_make_config(), store)
    loaded = load_fm_configuration(store)

    assert loaded is not None
    # camera transform + objective limit are the fields the earlier save path dropped
    assert loaded.camera_settings.transform == CameraImageTransform.FLIP_XY
    assert loaded.limit_position == 3.3e-3
    assert loaded.focus_position == 1.1e-3
    assert loaded.channel_settings[0].name == "GFP"
    assert loaded.channel_settings[0].exposure_time == 0.05


def test_the_working_state_lives_under_fm_in_the_session_file(store):
    save_fm_configuration(_make_config(), store)

    on_disk = yaml.safe_load(store.path.read_text())
    assert set(on_disk["fm"]) == {"working"}
    assert on_disk["fm"]["working"]["channel_settings"][0]["name"] == "GFP"


def test_the_working_state_and_recent_channels_do_not_overwrite_each_other(store):
    """Two writers of the one `fm` section: the autosave and the acquisition."""
    from fibsem.fm.config import load_recent_channels, record_recent_channels

    save_fm_configuration(_make_config(), store)
    record_recent_channels(ChannelSettings(name="DAPI"), store)
    save_fm_configuration(_make_config(), store)

    assert [c.name for c in load_recent_channels(store)] == ["DAPI"]
    assert load_fm_configuration(store) is not None


def test_the_old_working_file_is_imported_once_and_kept(store, tmp_path):
    old = tmp_path / "fm-configuration.yaml"
    _make_config().export(str(old))

    loaded = load_fm_configuration(store)

    assert loaded.channel_settings[0].name == "GFP"
    assert old.exists(), "the old file is never deleted"
    assert "working" in yaml.safe_load(store.path.read_text())["fm"]


def test_an_old_working_file_that_is_not_a_configuration_is_not_imported(
    store, tmp_path
):
    (tmp_path / "fm-configuration.yaml").write_text("channel_settings: nonsense\n")

    assert load_fm_configuration(store) is None
    assert not store.path.exists()


def test_a_script_reads_the_working_state_but_never_writes_it(store, tmp_path):
    save_fm_configuration(_make_config(), store)
    before = store.path.read_text()
    reader = SessionState("site.yaml", directory=str(tmp_path / "session"))

    assert load_fm_configuration(reader).channel_settings[0].name == "GFP"
    assert save_fm_configuration(_make_config(), reader) is False
    assert store.path.read_text() == before


def test_each_configuration_has_its_own_working_state(store, tmp_path):
    other = SessionState(
        "other.yaml", writable=True, directory=str(tmp_path / "session")
    )
    save_fm_configuration(_make_config(), store)

    assert load_fm_configuration(other) is None


def test_loading_a_configuration_loads_its_fm_working_state(tmp_path, monkeypatch):
    """`load_microscope_configuration` reads it, from that configuration's file."""
    import fibsem.session_state as session_state
    from fibsem import utils

    monkeypatch.setattr(session_state, "SESSION_STATE_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(
        fibsem_cfg, "FM_CONFIGURATION_PATH", str(tmp_path / "no-old-file.yaml")
    )
    config = tmp_path / "bench.yaml"
    config.write_text(
        (Path(fibsem_cfg.CONFIG_PATH) / "microscope-configuration.yaml").read_text()
    )
    save_fm_configuration(
        _make_config(),
        SessionState(str(config), writable=True, directory=str(tmp_path)),
    )

    settings = utils.load_microscope_configuration(str(config))

    assert settings.fm is not None
    assert settings.fm.channel_settings[0].name == "GFP"


def test_reading_a_configuration_dict_does_not_read_the_disk(monkeypatch):
    """`MicroscopeSettings.from_dict` is a function of the dict it is given."""
    from fibsem import fm
    from fibsem.structures import MicroscopeSettings

    def refuse(*args, **kwargs):
        raise AssertionError("from_dict read session state")

    monkeypatch.setattr(fm.config, "load_fm_configuration", refuse)

    assert MicroscopeSettings.from_dict({}).fm is None

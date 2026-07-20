"""Round-trip for the FM configuration persistence (fibsem/fm/config.py)."""

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


def _make_config() -> FluorescenceConfiguration:
    return FluorescenceConfiguration(
        channel_settings=[ChannelSettings(name="GFP", exposure_time=0.05)],
        z_parameters=ZParameters(),
        overview_parameters=OverviewParameters(),
        camera_settings=CameraSettings(transform=CameraImageTransform.ROTATE_90_CW),
        focus_position=1.1e-3,
        limit_position=3.3e-3,
    )


def test_load_configuration_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fibsem_cfg, "FM_CONFIGURATION_PATH", str(tmp_path / "fm-configuration.yaml")
    )
    assert load_fm_configuration() is None


def test_configuration_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fibsem_cfg, "FM_CONFIGURATION_PATH", str(tmp_path / "fm-configuration.yaml")
    )

    save_fm_configuration(_make_config())
    loaded = load_fm_configuration()

    assert loaded is not None
    # camera transform + objective limit are the fields the earlier save path dropped
    assert loaded.camera_settings.transform == CameraImageTransform.ROTATE_90_CW
    assert loaded.limit_position == 3.3e-3
    assert loaded.focus_position == 1.1e-3
    assert loaded.channel_settings[0].name == "GFP"
    assert loaded.channel_settings[0].exposure_time == 0.05

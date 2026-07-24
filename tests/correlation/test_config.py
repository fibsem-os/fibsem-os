"""Tests for CorrelationConfig (persistence phase 1, FIB-298).

Pure dataclasses, no Qt. Covers round-trip and — the load-bearing part —
back-compat: a config dict missing keys (or a whole nested block, or the whole
thing) must fill in defaults, so a protocol saved before this field loads clean.
"""
from fibsem.correlation.config import (
    CorrelationConfig,
    FitSettings,
    InterpolationSettings,
    RISettings,
)


def test_correlation_config_round_trips():
    cfg = CorrelationConfig(
        fit=FitSettings(fib_method="Gaussian", fm_poi_channel="GFP", reflection_cutout=3),
        ri=RISettings(na=0.9, wavelength_um=0.488, mode="post"),
        interpolation=InterpolationSettings(enabled=True, isotropic=False, target_z_nm=120.0),
        load_spot_burns=False,
    )
    back = CorrelationConfig.from_dict(cfg.to_dict())
    assert back == cfg


def test_defaults_match_the_current_hardcoded_behaviour():
    """The default config must reproduce today's behaviour so nothing changes
    for a project that never sets it."""
    cfg = CorrelationConfig()
    assert cfg.fit.fib_method == "Hole"
    assert cfg.fit.fm_fiducial_method == "None"
    assert cfg.fit.fm_poi_method == "Gaussian"
    assert (cfg.fit.reflection_cutout, cfg.fit.fluorescence_cutout) == (2, 5)
    # RI defaults mirror DEFAULT_ZETA_PARAMS in the RI widget
    assert (cfg.ri.tilt_deg, cfg.ri.depth_um, cfg.ri.na, cfg.ri.n2, cfg.ri.wavelength_um) == \
        (15.0, 4.0, 0.8, 1.4, 0.515)
    assert cfg.ri.mode == "pre"
    assert cfg.load_spot_burns is True
    assert cfg.interpolation.enabled is False


def test_channels_default_to_none_stored_by_name():
    """Channels are stored by name (resolved against the image later), so they
    start unset rather than pinned to an index."""
    fit = FitSettings()
    assert fit.fm_fiducial_channel is None
    assert fit.fm_poi_channel is None
    fit.fm_poi_channel = "RFP"
    assert FitSettings.from_dict(fit.to_dict()).fm_poi_channel == "RFP"


def test_empty_dict_yields_defaults():
    """A missing 'correlation' block on a legacy protocol -> a default config."""
    assert CorrelationConfig.from_dict({}) == CorrelationConfig()
    assert CorrelationConfig.from_dict(None) == CorrelationConfig()


def test_partial_dict_fills_missing_keys():
    """A hand-edited or older config with only some keys keeps the rest at default,
    rather than raising on a missing key."""
    cfg = CorrelationConfig.from_dict({"fit": {"fib_method": "None"}})
    assert cfg.fit.fib_method == "None"           # the provided key
    assert cfg.fit.fm_poi_method == "Gaussian"    # a sibling default
    assert cfg.ri == RISettings()                 # a missing nested block
    assert cfg.load_spot_burns is True


def test_nested_blocks_independently_tolerate_absence():
    cfg = CorrelationConfig.from_dict({"ri": {"na": 0.7}, "load_spot_burns": False})
    assert cfg.ri.na == 0.7
    assert cfg.ri.mode == "pre"                   # sibling default within ri
    assert cfg.fit == FitSettings()               # whole missing block
    assert cfg.interpolation == InterpolationSettings()
    assert cfg.load_spot_burns is False

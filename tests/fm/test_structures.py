import pytest

from fibsem.fm.structures import ChannelSettings, ZParameters, FluorescenceImage
import numpy as np


def test_channel_settings():
    channel_settings = ChannelSettings(
        name="test-channel",
        excitation_wavelength=488.0,
        emission_wavelength=520.0,
        exposure_time=0.1,
        power=0.3,
    )

    # test to_dict method
    cs_dict = channel_settings.to_dict()
    assert cs_dict["name"] == "test-channel"
    assert cs_dict["excitation_wavelength"] == 488.0
    assert cs_dict["emission_wavelength"] == 520.0
    assert cs_dict["exposure_time"] == 0.1
    assert cs_dict["power"] == 0.3

    # test from_dict method
    cs_from_dict = ChannelSettings.from_dict(cs_dict)
    assert cs_from_dict.name == "test-channel"
    assert cs_from_dict.excitation_wavelength == 488.0
    assert cs_from_dict.emission_wavelength == 520.0
    assert cs_from_dict.exposure_time == 0.1
    assert cs_from_dict.power == 0.3

    # test with emission set to None (reflection)
    channel_settings.emission_wavelength = None
    cs_dict_none = channel_settings.to_dict()
    assert cs_dict_none["emission_wavelength"] is None
    cs_from_dict_none = ChannelSettings.from_dict(cs_dict_none)
    assert cs_from_dict_none.emission_wavelength is None


def test_z_parameters():
    zinit = 0.0
    zmin, zmax, zstep = -10e-6, 10e-6, 1e-6
    zparams = ZParameters(zmin=zmin, zmax=zmax, zstep=zstep)

    # test to_dict method
    zparams_dict = zparams.to_dict()
    assert zparams_dict["zmin"] == zmin
    assert zparams_dict["zmax"] == zmax
    assert zparams_dict["zstep"] == zstep
    # test from_dict method
    zparams_from_dict = ZParameters.from_dict(zparams_dict)
    assert zparams_from_dict.zmin == zparams.zmin
    assert zparams_from_dict.zmax == zparams.zmax
    assert zparams_from_dict.zstep == zparams.zstep

    # test generate_z_positions
    z_positions = zparams.generate_positions(z_init=zinit)
    assert len(z_positions) == 21
    assert np.isclose(z_positions[0], zmin)
    assert np.isclose(z_positions[-1], zmax)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with even step
    zparams.zstep = 2.0e-6
    z_positions_even = zparams.generate_positions(z_init=zinit)
    assert len(z_positions_even) == 11
    assert np.isclose(z_positions_even[0], zmin)
    assert np.isclose(z_positions_even[-1], zmax)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with asymmetric range
    zparams.zmin = -10e-6
    zparams.zmax = 5e-6
    zparams.zstep = 1.0e-6
    z_positions = zparams.generate_positions(z_init=zinit)
    assert len(z_positions) == 16
    assert np.isclose(z_positions[0], -10e-6)
    assert np.isclose(z_positions[-1], 5e-6)
    assert any([np.isclose(zinit, pos) for pos in z_positions])

    # test with zero step
    zparams.zstep = 0.0
    with pytest.raises(ValueError):
        zparams.generate_positions(z_init=zinit)

    # test with negative step
    zparams.zstep = -1.0e-6
    with pytest.raises(ValueError):
        zparams.generate_positions(z_init=zinit)


def test_fluorescence_image():
    pass

"""A channel's emission is an EmissionFilter (FIB-1439): matching it against a
filter set, loading channels saved as a value, and writing it to OME."""

import numpy as np
import pytest

from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
    matching_emission_filter,
)

GFP = EmissionFilter.band("GFP", 510, 560)
RFP = EmissionFilter.band("RFP", 600, 650)
DUAL = EmissionFilter("Dual", bands=[(450, 490), (600, 650)])
FLUORESCENCE = EmissionFilter("Fluorescence", multi_band=True)
QUAD = EmissionFilter("Quad-band", bands=[(414, 450), (499.5, 530), (580, 611)])


@pytest.mark.parametrize(
    "wanted, choices, expected",
    [
        (GFP, [REFLECTION, GFP, RFP], GFP),
        (EmissionFilter("GFP"), [REFLECTION, GFP], GFP),
        (EmissionFilter.band("510 nm", 512), [REFLECTION, GFP, RFP], GFP),
        (REFLECTION, [REFLECTION, FLUORESCENCE], REFLECTION),
        (GFP, [REFLECTION, FLUORESCENCE], FLUORESCENCE),
        (DUAL, [REFLECTION, FLUORESCENCE], FLUORESCENCE),
        (FLUORESCENCE, [REFLECTION, GFP, FLUORESCENCE], FLUORESCENCE),
        # a channel saved as "Fluorescence" finds a multi-band filter of known bands
        (FLUORESCENCE, [REFLECTION, QUAD], QUAD),
        (GFP, [REFLECTION, QUAD], QUAD),
    ],
)
def test_matching_emission_filter(wanted, choices, expected):
    assert matching_emission_filter(wanted, choices) == expected


@pytest.mark.parametrize(
    "wanted, choices",
    [(REFLECTION, [GFP]), (FLUORESCENCE, [REFLECTION, GFP]), (GFP, [REFLECTION])],
)
def test_no_matching_emission_filter(wanted, choices):
    with pytest.raises(ValueError, match="No emission filter"):
        matching_emission_filter(wanted, choices)


@pytest.mark.parametrize(
    "old, expected",
    [
        (None, REFLECTION),
        ("Fluorescence", FLUORESCENCE),
        (510, EmissionFilter("510 nm", bands=[(510, None)])),
    ],
)
def test_channel_saved_with_an_emission_wavelength_loads(old, expected):
    channel = ChannelSettings.from_dict({"name": "c", "emission_wavelength": old})
    assert channel.emission_filter == expected
    assert "emission_wavelength" not in channel.to_dict()


def test_channel_round_trips_its_filter():
    channel = ChannelSettings(name="c", emission_filter=DUAL)
    loaded = ChannelSettings.from_dict(channel.to_dict())
    assert loaded.emission_filter == DUAL


def _channel(name, emission_filter):
    return FluorescenceChannelMetadata(
        name=name,
        excitation_wavelength=488,
        power=0.1,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        emission_filter=emission_filter,
    )


def test_ome_file_keeps_each_channels_filter(tmp_path):
    filters = [REFLECTION, GFP, DUAL, FLUORESCENCE]
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-10-16T12:00:00",
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        resolution=(8, 8),
        channels=[_channel(f"c{i}", f) for i, f in enumerate(filters)],
    )
    data = np.zeros((len(filters), 1, 8, 8), dtype=np.uint16)
    path = FluorescenceImage(data=data, metadata=metadata).save(
        str(tmp_path / "image.ome.tiff")
    )

    loaded = FluorescenceImage.load(path)

    assert [c.emission_filter for c in loaded.metadata.channels] == filters

    # Read from the OME alone (another program's view), a filter has one range.
    ome = loaded.get_ome_metadata()
    wavelengths = [c.emission_wavelength for c in ome.images[0].pixels.channels]
    assert wavelengths == [None, 535.0, None, None]
    from_ome = FluorescenceImageMetadata.from_ome(ome)
    assert [c.emission_filter for c in from_ome.channels] == [
        REFLECTION,
        GFP,
        EmissionFilter("Dual", bands=[(450, 650)], multi_band=True),
        FLUORESCENCE,
    ]


def _default_channel(excitations, filters):
    from types import SimpleNamespace

    from fibsem.fm.microscope import FluorescenceMicroscope

    fm = SimpleNamespace(
        filter_set=SimpleNamespace(
            available_excitation_wavelengths=tuple(excitations),
            available_emission_filters=tuple(filters),
        )
    )
    return FluorescenceMicroscope.default_channel(fm)


def test_the_default_channel_is_on_wavelengths_the_fm_has():
    """B10: a METEOR has no reflection position and no 550 nm LED, so the plain
    default (550 nm, Reflection) could not be set and the live view never started."""
    rfp = EmissionFilter.band("RFP", 579.5, 610.5)
    channel = _default_channel([365.0, 470.0, 500.0, 625.0], [GFP, rfp])

    assert channel.excitation_wavelength == 500.0
    assert channel.emission_filter == GFP  # the lowest band above 500 nm


def test_the_default_channel_keeps_reflection_where_the_fm_has_it():
    channel = _default_channel([550.0], [REFLECTION, GFP])

    assert channel.excitation_wavelength == 550.0
    assert channel.emission_filter == REFLECTION


def test_an_fm_that_reports_no_choices_keeps_the_plain_default():
    channel = _default_channel([], [])

    assert channel == ChannelSettings()

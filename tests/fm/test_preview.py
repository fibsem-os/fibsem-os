"""Projecting a fluorescence z-stack to one displayable RGB image."""

import numpy as np
import pytest

from fibsem.fm.composite import composite_fm_layers
from fibsem.fm.preview import (
    composite_projection,
    is_fluorescence_image,
    load_projection,
    projection_layers,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


def _channel(name: str, color: str) -> FluorescenceChannelMetadata:
    return FluorescenceChannelMetadata(
        name=name,
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
        color=color,
    )


def _stack(data: np.ndarray, colors, pixel_size: float = 1e-7) -> FluorescenceImage:
    """A (C, Z, Y, X) stack with one metadata channel per given colour."""
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-01-01T00:00:00",
        pixel_size_x=pixel_size,
        pixel_size_y=pixel_size,
        resolution=(data.shape[3], data.shape[2]),
        channels=[_channel(f"Channel-{i:02d}", c) for i, c in enumerate(colors)],
    )
    return FluorescenceImage(data=data, metadata=metadata)


@pytest.fixture
def stack_on_disk(monkeypatch):
    """Hand a prepared FluorescenceImage to load_projection as if read from disk.

    Deliberately does not go through save()/load(). Those transpose C and Z for
    most real shapes (FIB-279), so a fixture built that way would test the
    serialiser's bug rather than this module's contract, which is: given what load
    returns, project over z and composite.
    """

    def _install(image: FluorescenceImage) -> str:
        monkeypatch.setattr(
            "fibsem.fm.preview.FluorescenceImage.load",
            staticmethod(lambda _path: image),
        )
        return "stack.ome.tiff"

    return _install


# is_fluorescence_image


@pytest.mark.parametrize(
    "name, expected",
    [
        ("a.ome.tiff", True),
        ("a.ome.tif", True),
        ("A.OME.TIFF", True),
        ("ref_MillRough_final_res_01_eb.tif", False),
        ("a.png", False),
    ],
)
def test_recognises_fluorescence_by_extension(name, expected):
    assert is_fluorescence_image(name) is expected


# load_projection


def test_projects_over_z_not_over_channels(stack_on_disk):
    """The single most consequential axis choice here.

    Projecting axis 0 instead of axis 1 would collapse the *channels* and leave the
    z-slices to be treated as channels — the image would still render, plausibly,
    and be completely wrong. Channel 0 is bright in one z-slice only; channel 1 is
    dark everywhere. A correct z-projection is red; collapsing channels is not.
    """
    data = np.zeros((2, 3, 8, 8), dtype=np.uint16)
    data[0, 1] = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    path = stack_on_disk(_stack(data, ["red", "green"]))

    rgb, _ = load_projection(path)

    assert rgb.shape == (8, 8, 3)
    assert rgb[..., 0].max() == 255  # red channel survived the projection
    assert rgb[..., 1].max() == 0  # the dark channel contributes nothing


def test_returns_uint8_rgb_and_the_pixel_size(stack_on_disk):
    data = np.zeros((1, 2, 8, 8), dtype=np.uint16)
    data[0, 0] = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    path = stack_on_disk(_stack(data, ["red"], pixel_size=1.3e-7))

    rgb, pixel_size = load_projection(path)

    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert pixel_size == pytest.approx(1.3e-7)


def test_a_channel_missing_from_metadata_is_still_shown(stack_on_disk):
    """Metadata should describe every plane, but a plane must never vanish because
    it doesn't — an unnamed channel is better than a silently missing one."""
    ramp = np.linspace(0, 4095, 64, dtype=np.uint16).reshape(8, 8)
    data = np.zeros((2, 2, 8, 8), dtype=np.uint16)
    data[0, 0] = ramp
    data[1, 0] = ramp

    image = _stack(data, ["red"])  # metadata describes one of two planes
    rgb, _ = load_projection(stack_on_disk(image))

    assert rgb[..., 0].max() == 255  # the described channel is red
    assert rgb.sum(axis=2).max() > 255  # the undescribed one still contributes


def test_a_stack_with_no_channels_raises(stack_on_disk):
    """Better a clear error the loader logs than a blank tile with no explanation.

    Built with data that has no planes rather than metadata with no channels —
    FluorescenceImageMetadata rejects the latter outright.
    """
    image = _stack(np.zeros((0, 1, 8, 8), dtype=np.uint16), ["red"])

    with pytest.raises(ValueError, match="no displayable channels"):
        load_projection(stack_on_disk(image))


# projection_layers


def test_each_channel_is_its_own_projected_layer():
    """What the composite is blended from, kept apart so it can be re-blended:
    one layer per channel, projected over z, in the metadata's colour."""
    data = np.zeros((2, 3, 8, 8), dtype=np.uint16)
    data[0, 2, 1, 1] = 900
    data[1, 0, 5, 5] = 700
    layers = projection_layers(_stack(data, ["red", "cyan"]))

    assert [(layer.name, layer.color) for layer in layers] == [
        ("Channel-00", "red"),
        ("Channel-01", "cyan"),
    ]
    assert layers[0].data.shape == (8, 8)
    assert layers[0].data[1, 1] == 900 and layers[1].data[5, 5] == 700


def test_the_composite_is_the_blend_of_the_layers():
    rng = np.random.default_rng(0)
    data = (rng.random((2, 2, 8, 8)) * 4000).astype(np.uint16)
    image = _stack(data, ["green", "magenta"])
    assert np.array_equal(
        composite_projection(image), composite_fm_layers(projection_layers(image))
    )

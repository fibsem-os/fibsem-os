"""Projecting a fluorescence z-stack to one displayable RGB image."""

import numpy as np
import pytest

from fibsem.fm.composite import composite_fm_layers
from fibsem.fm.preview import (
    composite_projection,
    is_fluorescence_image,
    load_projection,
    load_stack_preview,
    projection_layers,
    render_stack,
    stack_facts,
    stack_layers,
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
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=pixel_size,
        pixel_size_y=pixel_size,
        resolution=(data.shape[3], data.shape[2]),
        channels=[_channel(f"Channel-{i:02d}", c) for i, c in enumerate(colors)],
    )
    return FluorescenceImage(data=data, metadata=metadata)


class _StackInMemory:
    """A StackFile over a FluorescenceImage already in hand."""

    def __init__(self, path, image: FluorescenceImage):
        self.path, self.metadata = path, image.metadata
        self._data = image.data
        self.shape = image.data.shape
        self.channels, self.planes = self.shape[:2]

    def plane(self, channel: int, z: int) -> np.ndarray:
        return self._data[channel, z]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


@pytest.fixture
def stack_on_disk(monkeypatch, tmp_path):
    """Hand a prepared FluorescenceImage to the preview as if read from disk.

    Deliberately does not go through save(): metadata that doesn't describe
    every channel, or a stack with no channels, can't be written, and the
    preview still has to cope with them.
    """
    images = {}
    monkeypatch.setattr(
        "fibsem.fm.stack_file.StackFile",
        lambda path: _StackInMemory(path, images[path]),
    )

    def _install(image: FluorescenceImage) -> str:
        path = str(tmp_path / f"stack-{len(images)}.ome.tiff")
        open(path, "wb").close()
        images[path] = image
        return path

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


def test_a_stack_says_its_channels_planes_and_step_for_its_legend(stack_on_disk):
    data = np.zeros((2, 5, 8, 8), dtype=np.uint16)
    image = _stack(data, ["green"])  # metadata describes one of two channels
    image.metadata.z_positions = [0.0, 2e-7, 4e-7, 6e-7, 8e-7]

    _, _, facts = load_stack_preview(stack_on_disk(image))

    assert facts["z_planes"] == 5
    assert facts["z_step"] == pytest.approx(2e-7)
    first, second = facts["channels"]
    assert first["name"] == "Channel-00" and first["color"] == "green"
    assert first["excitation_wavelength"] == 488.0
    assert first["exposure_time"] == 0.1  # for the preview's metadata line
    assert facts["acquired_at"].startswith("2026-01-01T00:00:00")
    # drawn as the projection draws it, though the metadata doesn't name it
    assert second["name"] == "Channel-02" and second["excitation_wavelength"] is None
    assert [layer.color for layer in projection_layers(image)] == [
        c["color"] for c in facts["channels"]
    ]


def test_a_recorded_z_step_is_the_step(stack_on_disk):
    image = _stack(np.zeros((1, 3, 8, 8), dtype=np.uint16), ["red"])
    image.metadata.pixel_size_z = 5e-7

    assert stack_facts(image)["z_step"] == pytest.approx(5e-7)


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


def test_gray_channels_in_a_stack_take_the_palette():
    """Gray is the settings' default, so a stack of gray channels would draw as
    one gray blur (FIB-1517): each takes its palette colour instead, and the
    legend says so. A lone gray channel stays gray."""
    data = np.zeros((2, 3, 8, 8), dtype=np.uint16)
    image = _stack(data, ["gray", "gray"])
    colors = [layer.color for layer in projection_layers(image)]
    assert colors == ["violet", "blue"]
    assert [c["color"] for c in stack_facts(image)["channels"]] == colors
    lone = _stack(np.zeros((1, 3, 8, 8), dtype=np.uint16), ["gray"])
    assert projection_layers(lone)[0].color == "gray"


def test_the_composite_is_the_blend_of_the_layers():
    rng = np.random.default_rng(0)
    data = (rng.random((2, 2, 8, 8)) * 4000).astype(np.uint16)
    image = _stack(data, ["green", "magenta"])
    assert np.array_equal(
        composite_projection(image), composite_fm_layers(projection_layers(image))
    )


# a plane, some channels, and how sharp each plane is


def _spots(z_planes: int = 5) -> FluorescenceImage:
    """Two channels: a point in focus at plane 1 in the first, plane 3 in the
    second, blurring and dimming away from it."""
    yy, xx = np.mgrid[:64, :64]
    data = np.zeros((2, z_planes, 64, 64), dtype=np.float32)
    for c, (focus, x) in enumerate(((1, 16), (3, 48))):
        for z in range(z_planes):
            width = 2.0 + 3.0 * abs(z - focus)
            data[c, z] = (
                1000
                / (1 + abs(z - focus))
                * np.exp(-((xx - x) ** 2 + (yy - 32) ** 2) / (2 * width**2))
            )
    return _stack(data.astype(np.uint16), ["green", "red"])


def test_a_plane_shows_that_plane_not_the_projection():
    image = _spots()

    def spread(rgb):  # how many pixels the green point lights
        return int((rgb[..., 1] > 128).sum())

    # the green point is tightest in its own plane, and blurred far from it
    assert spread(render_stack(image, z=1)) < spread(render_stack(image, z=4))
    assert spread(render_stack(image, z=4)) != spread(render_stack(image))


def test_channels_shows_only_those_named():
    image = _spots()

    red_only = render_stack(image, channels=["Channel-01"])

    assert red_only[32, 16, 1] == 0  # the green point is hidden
    assert red_only[32, 48, 0] > 0
    assert [l.visible for l in stack_layers(image, channels=["Channel-01"])] == [
        False,
        True,
    ]


def test_a_plane_or_channel_the_stack_lacks_is_named():
    image = _spots()

    with pytest.raises(ValueError, match="plane 5 is outside the stack's 0 to 4"):
        render_stack(image, z=5)
    with pytest.raises(ValueError, match="no channel GFP .* Channel-00, Channel-01"):
        render_stack(image, channels=["GFP"])


def test_each_channel_says_how_sharp_each_plane_is():
    image = _spots()
    image.metadata.z_positions = [0.0, 5e-7, 1e-6, 1.5e-6, 2e-6]

    facts = stack_facts(image)

    green, red = facts["channels"]
    assert green["sharpest_plane"] == 1 and red["sharpest_plane"] == 3
    assert green["sharpness"][1] == 1.0 and len(green["sharpness"]) == 5
    assert facts["z_positions"] == pytest.approx(image.metadata.z_positions)


def test_a_single_plane_has_no_sharpness_to_compare():
    facts = stack_facts(_stack(np.ones((1, 1, 8, 8), dtype=np.uint16), ["red"]))

    assert "sharpness" not in facts["channels"][0]
    assert facts["z_positions"] is None

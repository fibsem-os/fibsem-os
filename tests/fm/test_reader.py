"""Reading a fluorescence image from other software for import (FIB-1030).

The reader must keep the pixels as stored and suggest only what the file actually
says: which axis is what, a pixel size in a physical unit (never a print resolution),
channel names and colours. The builder must then turn confirmed answers into an image
that saves and loads back as it was built. No Qt, no network.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from fibsem.fm.reader import (
    ImportSource,
    _ome_colours,
    arrange,
    build_image,
    check_roles,
    default_channels,
    nearest_colour,
    read_source,
    suggest_roles,
)
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import CameraImageTransform, FibsemStagePosition

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def planes(nc, nz, h=6, w=7):
    data = np.zeros((nc, nz, h, w), dtype=np.uint16)
    for c in range(nc):
        for z in range(nz):
            data[c, z] = (10 * c + z + 1) * 97 + np.arange(h * w).reshape(h, w) % 3
    return data


# ── roles ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "axes, roles",
    [
        ("YX", "YX"),
        ("ZYX", "ZYX"),
        ("CZYX", "CZYX"),
        ("ZCYX", "ZCYX"),  # ImageJ: kept as stored, arranged later
        ("IYX", "ZYX"),  # a page sequence is z, as the loader has it
        ("QQYX", "CZYX"),
        ("YXS", "YXC"),  # colour samples are channels
        ("IYXS", "ZYXC"),
        ("CYXS", "CYXZ"),  # channels named too: the samples take a free role
        ("TZCYX", "TZCYX"),
        ("QQQYX", "CZTYX"),  # one unnamed axis too many becomes time
    ],
)
def test_roles_are_suggested_from_what_the_file_says(axes, roles):
    assert suggest_roles(axes) == roles


@pytest.mark.parametrize(
    "roles, shape, message",
    [
        ("CZY", (2, 3, 4, 5), "4 axes"),
        ("CCYX", (2, 3, 4, 5), "only one axis can be Channel"),
        ("CZYY", (2, 3, 4, 5), "exactly one axis must be Y"),
        ("CZ?X", (2, 3, 4, 5), "is not a role"),
    ],
)
def test_roles_that_cannot_be_used_say_why(roles, shape, message):
    with pytest.raises(ValueError, match=message):
        check_roles(roles, shape)


def test_arranging_follows_the_roles_given_not_the_file():
    """The user corrects a file: what was read as z is channels, and back."""
    data = planes(2, 3)  # stored C, Z
    np.testing.assert_array_equal(arrange(data, "CZYX"), data)
    np.testing.assert_array_equal(arrange(data, "ZCYX"), data.transpose(1, 0, 2, 3))
    stored = np.stack([data, data + 1])  # T, C, Z
    np.testing.assert_array_equal(arrange(stored, "TCZYX"), data)  # first point


# ── reading ───────────────────────────────────────────────────────────────────


def test_a_plain_stack_is_z_with_no_pixel_size(tmp_path):
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, planes(1, 5)[0], photometric="minisblack")

    source = read_source(path)

    assert source.roles == "ZYX"
    assert source.pixel_size is None
    assert (source.channel_names, source.channel_colors) == (["Channel-01"], ["gray"])
    assert source.described_by == "no metadata"


def test_a_resolution_in_inches_is_not_a_pixel_size(tmp_path):
    """Zeiss exports say 300 dpi: a print setting."""
    path = str(tmp_path / "export.tif")
    tifffile.imwrite(path, planes(1, 1)[0, 0], resolution=(300, 300), resolutionunit=2)
    assert read_source(path).pixel_size is None


def test_a_resolution_in_centimetres_is_read(tmp_path):
    path = str(tmp_path / "cm.tif")
    per_cm = 1 / 1.3e-5  # 0.13 um pixels, as a METEOR export states them
    tifffile.imwrite(
        path, planes(1, 1)[0, 0], resolution=(per_cm, per_cm), resolutionunit=3
    )
    assert read_source(path).pixel_size == pytest.approx(0.13e-6)


def test_an_implausible_pixel_size_is_not_suggested(tmp_path):
    path = str(tmp_path / "odd.tif")
    tifffile.imwrite(path, planes(1, 1)[0, 0], resolution=(1, 1), resolutionunit=3)
    assert read_source(path).pixel_size is None  # 1 cm pixels


@pytest.mark.parametrize(
    "unit, per_unit, expected",
    [("micron", 1 / 0.325, 0.325e-6), ("nm", 1 / 130, 130e-9)],
)
def test_an_imagej_hyperstack_gives_its_axes_and_pixel_size(
    tmp_path, unit, per_unit, expected
):
    data = planes(3, 4)
    path = str(tmp_path / "stack.ij.tiff")
    tifffile.imwrite(
        path,
        data.transpose(1, 0, 2, 3),
        imagej=True,
        resolution=(per_unit, per_unit),
        metadata={"axes": "ZCYX", "unit": unit},
    )

    source = read_source(path)

    assert source.axes == "ZCYX" and source.roles == "ZCYX"
    assert source.described_by == "ImageJ metadata"
    assert source.pixel_size == pytest.approx(expected)
    assert source.channel_names == ["Channel-01", "Channel-02", "Channel-03"]
    np.testing.assert_array_equal(arrange(source.data, source.roles), data)


def test_an_ome_tiff_from_other_software_gives_its_names_and_pixel_size(tmp_path):
    path = str(tmp_path / "foreign.ome.tif")
    tifffile.imwrite(
        path,
        planes(2, 3),
        ome=True,
        photometric="minisblack",
        metadata={
            "axes": "CZYX",
            "PhysicalSizeX": 0.2,
            "PhysicalSizeY": 0.2,
            "Channel": {"Name": ["DAPI", "GFP"]},
        },
    )

    source = read_source(path)

    assert source.described_by == "OME metadata"
    assert source.pixel_size == pytest.approx(0.2e-6)
    assert source.channel_names == ["DAPI", "GFP"]
    assert source.roles == "CZYX"


def test_our_own_file_gives_its_channels_and_colours(tmp_path):
    """A FibsemOS image missing only its placement, say one taken before geometry
    was recorded, keeps what it knows."""
    data = planes(2, 1)
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-09-23T10:00:00",
        pixel_size_x=1.1e-7,
        pixel_size_y=1.1e-7,
        channels=[
            FluorescenceChannelMetadata(
                name=n,
                excitation_wavelength=wl,
                power=0.5,
                exposure_time=0.2,
                gain=1.0,
                offset=0.0,
                color=c,
            )
            for n, wl, c in (("GFP", 488.0, "green"), ("mCherry", 561.0, "red"))
        ],
    )
    path = str(tmp_path / "ours.ome.tiff")
    FluorescenceImage(data=data, metadata=metadata).save(path)

    source = read_source(path)

    assert source.described_by == "FibsemOS metadata"
    assert (source.channel_names, source.channel_colors) == (
        ["GFP", "mCherry"],
        ["green", "red"],
    )
    assert source.pixel_size == pytest.approx(1.1e-7)
    assert [c.excitation_wavelength for c in source.channel_metadata] == [488.0, 561.0]


def test_an_rgb_tiff_is_red_green_and_blue(tmp_path):
    rgb = np.zeros((6, 7, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    path = str(tmp_path / "export.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")

    source = read_source(path)

    assert source.is_rgb and source.roles == "YXC"
    assert list(zip(source.channel_names, source.channel_colors)) == [
        ("Red", "red"),
        ("Green", "green"),
        ("Blue", "blue"),
    ]
    arranged = arrange(source.data, source.roles)
    assert arranged.shape == (3, 1, 6, 7)
    assert arranged[1].max() == 200 and arranged[0].max() == 0


@pytest.mark.parametrize("suffix", [".png", ".jpg"])
def test_a_png_or_jpeg_screenshot_is_read_as_rgb(tmp_path, suffix):
    rgb = np.zeros((10, 12, 3), dtype=np.uint8)
    rgb[..., 0] = 250
    path = str(tmp_path / f"shot{suffix}")
    Image.fromarray(rgb).save(path)

    source = read_source(path)

    assert source.axes == "YXS" and source.roles == "YXC"
    assert source.pixel_size is None
    assert source.data.shape == (10, 12, 3)
    assert source.data[..., 0].min() > 240  # JPEG is lossy; red is still red


def test_a_16_bit_grey_png_keeps_its_depth(tmp_path):
    grey = (np.arange(10 * 12).reshape(10, 12) * 500).astype(np.uint16)
    path = str(tmp_path / "grey.png")
    Image.fromarray(grey).save(path)

    source = read_source(path)

    assert source.roles == "YX"
    np.testing.assert_array_equal(source.data, grey)


def test_other_files_are_refused(tmp_path):
    path = tmp_path / "image.czi"
    path.write_bytes(b"ZISRAWFILE")
    with pytest.raises(ValueError, match="can import"):
        read_source(str(path))


# ── colours ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "colour, expected",
    [("#04ff00", "green"), ("#ff1800", "red"), ("#ffffff", "gray"), ("cyan", "cyan")],
)
def test_a_files_colour_becomes_the_nearest_fm_colour(colour, expected):
    assert nearest_colour(colour) == expected


def _ome(*rgbs):
    channels = [
        SimpleNamespace(color=SimpleNamespace(as_rgb_tuple=lambda rgb=rgb: rgb))
        for rgb in rgbs
    ]
    return SimpleNamespace(
        images=[SimpleNamespace(pixels=SimpleNamespace(channels=channels))]
    )


def test_white_is_kept_beside_real_colours_but_all_white_says_nothing():
    """A METEOR reflection channel is white among coloured ones; OME's default is
    white for every channel."""
    assert _ome_colours(_ome((255, 0, 0), (255, 255, 255))) == ["#ff0000", "#ffffff"]
    assert _ome_colours(_ome((255, 255, 255), (255, 255, 255))) == [None, None]


def test_default_channels_are_grey_alone_and_coloured_together():
    assert default_channels(1) == (["Channel-01"], ["gray"])
    names, colours = default_channels(3)
    assert names == ["Channel-01", "Channel-02", "Channel-03"]
    assert len(set(colours)) == 3


# ── building ──────────────────────────────────────────────────────────────────


def _source(data, axes, **kw):
    return ImportSource(
        path="/data/import.tif", data=data, axes=axes, roles=suggest_roles(axes), **kw
    )


def test_the_built_image_has_the_confirmed_axes_pixel_size_and_channels():
    stored = planes(2, 3).transpose(1, 0, 2, 3)  # read as ZCYX
    geometry = SimpleNamespace(kind="stand-in")
    base = FibsemStagePosition(x=1e-4, y=-2e-4, z=0.0, r=0.0, t=0.0)

    image = build_image(
        _source(stored, "ZCYX"),
        "ZCYX",
        pixel_size=0.325e-6,
        channel_names=["GFP", "RFP"],
        channel_colors=["green", "red"],
        geometry=geometry,
        stage_position=base,
    )

    np.testing.assert_array_equal(image.data, planes(2, 3))
    md = image.metadata
    assert md.pixel_size_x == md.pixel_size_y == pytest.approx(0.325e-6)
    assert [(c.name, c.color) for c in md.channels] == [
        ("GFP", "green"),
        ("RFP", "red"),
    ]
    assert md.geometry is geometry and md.stage_position is base
    assert md.resolution == (7, 6)
    assert "import.tif" in md.description


def test_a_flip_mirrors_left_to_right_only():
    data = planes(1, 1)
    image = build_image(
        _source(data[0, 0], "YX"), "YX", 1e-6, ["c"], ["gray"], flip=True
    )
    np.testing.assert_array_equal(image.data, data[..., ::-1])
    assert "mirrored" in image.metadata.description


def test_the_files_own_channel_records_are_kept():
    known = [
        FluorescenceChannelMetadata(
            name="EGFP",
            excitation_wavelength=470.0,
            power=0.2,
            exposure_time=0.05,
            gain=1.1,
            offset=0.0,
        )
    ]
    image = build_image(
        _source(planes(1, 2), "CZYX", channel_metadata=known),
        "CZYX",
        1e-7,
        ["GFP (renamed)"],
        ["green"],
    )
    channel = image.metadata.channels[0]
    assert (channel.name, channel.color) == ("GFP (renamed)", "green")
    assert (channel.excitation_wavelength, channel.exposure_time) == (470.0, 0.05)


@pytest.mark.parametrize(
    "names, pixel_size, message",
    [(["one"], 1e-6, "2 channels need"), (["a", "b"], 0.0, "pixel size")],
)
def test_building_refuses_what_it_cannot_place(names, pixel_size, message):
    with pytest.raises(ValueError, match=message):
        build_image(
            _source(planes(2, 1), "CZYX"),
            "CZYX",
            pixel_size,
            names,
            ["gray"] * len(names),
        )


@pytest.mark.parametrize("compression", [None, "zlib"])
def test_the_built_image_saves_and_loads_back_as_built(tmp_path, compression):
    """What goes into the grid folder is our own OME-TIFF: it must come back with
    its pixels, channels, pixel size and placement intact."""
    from fibsem.structures import FibsemHardwareGeometry

    geometry = FibsemHardwareGeometry(
        column_tilt=52.0,
        fib_column_tilt=52.0,
        shuttle_pre_tilt=0.0,
        rotation_reference=0.0,
        rotation_180=180.0,
        is_compustage=True,
        camera_tilt=0.0,
        transform=CameraImageTransform.NONE,
    )
    base = FibsemStagePosition(x=1e-4, y=-2e-4, z=0.0, r=0.0, t=0.0)
    image = build_image(
        _source(planes(3, 2).transpose(1, 0, 2, 3), "ZCYX"),
        "ZCYX",
        0.13e-6,
        ["Reflection", "EGFP", "RFP"],
        ["gray", "green", "red"],
        flip=True,
        geometry=geometry,
        stage_position=base,
    )
    path = str(tmp_path / "imported.ome.tiff")
    image.save(path, compression=compression)
    with tifffile.TiffFile(path) as tif:
        page = tif.pages.first
        assert page.compression.name == ("ADOBE_DEFLATE" if compression else "NONE")
        predictor = tifffile.PREDICTOR(page.predictor).name
        assert predictor == ("HORIZONTAL" if compression else "NONE")

    back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, image.data)
    assert [(c.name, c.color) for c in back.metadata.channels] == [
        ("Reflection", "gray"),
        ("EGFP", "green"),
        ("RFP", "red"),
    ]
    assert back.metadata.pixel_size_x == pytest.approx(0.13e-6)
    assert back.metadata.geometry == geometry
    assert (back.metadata.stage_position.x, back.metadata.stage_position.y) == (
        pytest.approx(1e-4),
        pytest.approx(-2e-4),
    )


# ── how an imported image is assumed to have been taken ────────────────────────


class _NoFM:
    """A FIB/SEM with no fluorescence microscope: the case a truly external image is."""

    def __init__(self):
        from fibsem.structures import FibsemHardwareGeometry

        self.geometry = FibsemHardwareGeometry(
            column_tilt=52.0,
            fib_column_tilt=52.0,
            shuttle_pre_tilt=35.0,
            rotation_reference=0.0,
            rotation_180=180.0,
            is_compustage=False,
            camera_tilt=12.0,
            transform=CameraImageTransform.FLIP_XY,
        )

    def fm_image_geometry(self):
        raise ValueError("Fluorescence microscope is not available.")

    def hardware_geometry(self):
        return self.geometry

    def get_orientation(self, name):
        if name == "FM":
            raise ValueError(f"Orientation {name} not supported.")
        return FibsemStagePosition(r=0.1, t=0.2)


def test_without_an_fm_the_image_is_taken_as_seen_straight_down():
    from fibsem.fm.reader import assumed_geometry, assumed_pose

    scope = _NoFM()
    geometry = assumed_geometry(scope)
    assert geometry.transform == CameraImageTransform.NONE
    assert geometry.camera_tilt == 0.0
    assert geometry.shuttle_pre_tilt == 35.0  # the instrument's own terms kept
    pose = assumed_pose(scope)
    assert (pose.r, pose.t) == (0.1, 0.2)  # the SEM's


def test_a_meteor_imagej_export_reads_without_an_error_in_the_log(tmp_path, caplog):
    """The METEOR `.ij.tiff` carries OME text tifffile cannot parse; read as
    ImageJ, its axes come through and nothing is logged as an error."""
    import logging

    data = planes(2, 3)  # C, Z
    description = (
        "ImageJ=1.11a\nimages=6\nchannels=2\nslices=3\nframes=1\nhyperstack=true\n"
        'unit=micron\n<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/'
        'Schemas/OME/2016-06"><Image ID="Image:0"><Pixels ID="Pixels:0" '
        'DimensionOrder="XYZTC" Type="uint16" SizeX="7" SizeY="6" SizeZ="3" SizeC="2" '
        'SizeT="1"/></Image></OME>\n'
    )
    path = str(tmp_path / "feature.ij.tiff")
    with tifffile.TiffWriter(path) as writer:
        for z in range(3):
            for c in range(2):
                writer.write(
                    data[c, z],
                    photometric="minisblack",
                    description=description if (z, c) == (0, 0) else None,
                    metadata=None,
                    contiguous=False,
                )

    with caplog.at_level(logging.ERROR, logger="tifffile"):
        source = read_source(path)

    assert source.axes == "ZCYX" and source.described_by == "ImageJ metadata"
    np.testing.assert_array_equal(arrange(source.data, source.roles), data)
    assert [r.getMessage() for r in caplog.records if r.name == "tifffile"] == []

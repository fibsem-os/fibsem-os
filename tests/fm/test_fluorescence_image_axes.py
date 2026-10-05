"""Which axis of a fluorescence file is channels and which is z (FIB-279).

`FluorescenceImage.load` used to guess the order by comparing the array's sizes to the
channel count and the number of z positions in its metadata. With as many channels as
z-slices the guess cannot tell them apart, and a save-and-load round trip handed back
the channels as z. The order now comes from the file: tifffile assembles an OME-TIFF
from its per-plane mapping and names the axes it produced.

Three sources of files, each covered across the shapes that matter:

* **What FibsemOS wrote before the fix** -- real bytes from the old `save`, committed
  under tests/fixtures/fm_legacy. They must load exactly, the C == Z ones included.
* **What `save` writes now** -- every combination of 1-4 channels and 1-5 z-slices.
* **Other software** -- OME-TIFFs in several axis orders without our annotation, ImageJ
  hyperstacks, plain and RGB TIFFs.

Every plane holds a distinct value, so any swap or shuffle shows. No Qt, no network.
"""

import json
import os

import numpy as np
import pytest
import tifffile

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
    to_czyx,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

LEGACY = os.path.join(os.path.dirname(__file__), "..", "fixtures", "fm_legacy")
HEIGHT, WIDTH = 6, 7


def planes(nc, nz, dtype="uint16"):
    """(C, Z, Y, X) with every plane distinct: `(10 c + z + 1) * step` plus a ramp.
    The same values tests/fixtures/fm_legacy/make_fixtures.py wrote."""
    step = 3 if dtype == "uint8" else 97
    data = np.zeros((nc, nz, HEIGHT, WIDTH), dtype=dtype)
    ramp = np.arange(HEIGHT * WIDTH).reshape(HEIGHT, WIDTH) % 3
    for c in range(nc):
        for z in range(nz):
            data[c, z] = (10 * c + z + 1) * step + ramp
    return data


def _channel(name):
    return FluorescenceChannelMetadata(
        name=name,
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
    )


def _image(data, nc=None, nz=None):
    nc = data.shape[0] if nc is None else nc
    nz = data.shape[1] if nz is None else nz
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-09-23T10:00:00",
        pixel_size_x=2e-7,
        pixel_size_y=2e-7,
        resolution=(WIDTH, HEIGHT),
        channels=[_channel(f"ch{c}") for c in range(nc)],
        z_positions=[z * 5e-7 for z in range(nz)] if nz > 1 else None,
    )
    return FluorescenceImage(data=data, metadata=metadata)


# ── files FibsemOS wrote before the fix ────────────────────────────────────────


@pytest.mark.parametrize(
    "nc, nz", [(1, 4), (4, 1), (2, 3), (3, 2), (2, 2), (3, 3), (4, 4)]
)
def test_a_file_written_before_the_fix_loads_exactly(nc, nz):
    path = os.path.join(LEGACY, f"c{nc}z{nz}.ome.tiff")
    with tifffile.TiffFile(path) as tif:
        # Guard: this is the old layout -- declared channel-fastest, read z-outer.
        assert 'DimensionOrder="XYCZT"' in tif.pages[0].description
        if nc > 1 and nz > 1:
            assert tif.series[0].axes == "ZCYX"

    back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, planes(nc, nz))
    assert [c.name for c in back.metadata.channels] == [f"ch{c}" for c in range(nc)]
    assert len(back.metadata.z_positions or [0]) == nz


def test_an_overview_tile_written_before_the_fix_loads_exactly():
    back = FluorescenceImage.load(os.path.join(LEGACY, "tile.ome.tiff"))
    np.testing.assert_array_equal(back.data, planes(1, 1))


# ── files save writes now ─────────────────────────────────────────────────────


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("nz", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("nc", [1, 2, 3, 4])
def test_a_save_and_load_round_trip_keeps_every_plane_in_place(tmp_path, nc, nz, dtype):
    data = planes(nc, nz, dtype)
    path = str(tmp_path / "stack.ome.tiff")
    _image(data.copy()).save(path)

    back = FluorescenceImage.load(path)

    assert back.data.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(back.data, data)
    assert [c.name for c in back.metadata.channels] == [f"ch{c}" for c in range(nc)]


def test_a_2d_image_round_trips(tmp_path):
    """An overview tile is handed to save as a 2D array."""
    data = planes(1, 1)[0, 0]
    path = str(tmp_path / "tile.ome.tiff")
    _image(data.copy(), nc=1, nz=1).save(path)
    np.testing.assert_array_equal(FluorescenceImage.load(path).data, data[None, None])


@pytest.mark.parametrize("nc, nz", [(2, 3), (2, 2), (1, 3), (1, 4), (3, 1)])
def test_a_saved_file_declares_the_order_it_is_written_in(tmp_path, nc, nz):
    """Planes go channel by channel, every z of one before the next, so the file
    says z varies fastest and is greyscale. Then any reader -- an older FibsemOS
    comparing sizes among them -- gets channels first, and three or four z-slices
    are never taken for the samples of an RGB image."""
    path = str(tmp_path / "stack.ome.tiff")
    _image(planes(nc, nz)).save(path)
    with tifffile.TiffFile(path) as tif:
        assert 'DimensionOrder="XYZCT"' in tif.pages[0].description
        assert tif.series[0].keyframe.photometric.name == "MINISBLACK"
        assert len(tif.pages) == nc * nz
        expected = "".join(a for a, n in (("C", nc), ("Z", nz)) if n > 1) + "YX"
        assert tif.series[0].axes == expected


# ── files from other software ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "axes, nc, nz",
    [
        ("CZYX", 2, 3),
        ("ZCYX", 2, 3),
        ("CZYX", 2, 2),
        ("ZCYX", 3, 3),
        ("CYX", 3, 1),
        ("ZYX", 1, 4),
        ("YX", 1, 1),
    ],
)
def test_an_ome_tiff_from_other_software_loads_in_any_axis_order(
    tmp_path, axes, nc, nz
):
    """No annotation of ours: the axes come from the file and the pixel size and
    channel names from its own OME. This raised UnboundLocalError before: the
    annotation search found nothing, raised nothing, and left nothing to return."""
    data = planes(nc, nz)
    stored = {
        "CZYX": data,
        "ZCYX": data.transpose(1, 0, 2, 3),
        "CYX": data[:, 0],
        "ZYX": data[0],
        "YX": data[0, 0],
    }[axes]
    path = str(tmp_path / "foreign.ome.tif")
    metadata = {"axes": axes, "PhysicalSizeX": 0.325, "PhysicalSizeY": 0.325}
    if "C" in axes:
        metadata["Channel"] = {"Name": [f"dye{c}" for c in range(nc)]}
    tifffile.imwrite(
        path, stored, ome=True, photometric="minisblack", metadata=metadata
    )

    back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, data)
    assert back.metadata.pixel_size_x == pytest.approx(0.325e-6)
    if "C" in axes:
        assert [c.name for c in back.metadata.channels] == [
            f"dye{c}" for c in range(nc)
        ]
    assert back.metadata.geometry is None  # nothing says how it was taken


def test_an_ome_tiff_with_several_time_points_keeps_them(tmp_path):
    data = np.stack([planes(2, 3), planes(2, 3) + 1])  # T, C, Z, Y, X
    path = str(tmp_path / "timelapse.ome.tif")
    tifffile.imwrite(
        path, data, ome=True, photometric="minisblack", metadata={"axes": "TCZYX"}
    )
    np.testing.assert_array_equal(FluorescenceImage.load(path).data, data)


@pytest.mark.parametrize(
    "axes, nc, nz",
    [("ZCYX", 2, 3), ("ZCYX", 3, 2), ("ZCYX", 2, 2), ("ZCYX", 3, 3), ("CYX", 3, 1)],
)
def test_an_imagej_hyperstack_loads_channels_first(tmp_path, axes, nc, nz):
    """ImageJ stores z outside channels; the file says so, and it is followed. A
    METEOR `.ij.tiff` export is this: 3 channels by 21 z-slices."""
    data = planes(nc, nz)
    stored = data.transpose(1, 0, 2, 3) if axes == "ZCYX" else data[:, 0]
    path = str(tmp_path / "hyperstack.tif")
    tifffile.imwrite(path, stored, imagej=True, metadata={"axes": axes})

    np.testing.assert_array_equal(FluorescenceImage.load(path).data, data)


def _meteor_like(path, data):
    """An ImageJ hyperstack whose description also carries OME text that is not
    valid OME, as a METEOR `.ij.tiff` export does. Stored z outside channels."""
    nc, nz, height, width = data.shape
    description = (
        f"ImageJ=1.11a\nimages={nc * nz}\nchannels={nc}\nslices={nz}\nframes=1\n"
        "hyperstack=true\nmode=grayscale\nunit=micron\n"
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        f'<Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZTC" '
        f'Type="uint16" SizeX="{width}" SizeY="{height}" SizeZ="{nz}" '
        f'SizeC="{nc}" SizeT="1"/></Image></OME>\n'
    )
    with tifffile.TiffWriter(path) as writer:
        for z in range(nz):
            for c in range(nc):
                writer.write(
                    data[c, z],
                    photometric="minisblack",
                    description=description if (z, c) == (0, 0) else None,
                    metadata=None,
                    contiguous=False,
                )


def test_a_meteor_imagej_export_loads_without_an_error_in_the_log(tmp_path, caplog):
    """Read as OME first, such a file makes tifffile log an ERROR on every open
    before it falls back to ImageJ. It is read as ImageJ, and the log stays quiet."""
    import logging

    data = planes(2, 3)
    path = str(tmp_path / "feature.ij.tiff")
    _meteor_like(path, data)
    with caplog.at_level(logging.ERROR, logger="tifffile"):
        with tifffile.TiffFile(path) as tif:
            tif.series[0].asarray()
    # Recent tifffile tries the OME series first and logs; the one Python 3.8 gets
    # (2023.7.10) reads it as ImageJ and does not. The load must be right and quiet
    # on both; only on the first is there an error for it to have avoided.
    plain_open_logged = any(r.name == "tifffile" for r in caplog.records)
    caplog.clear()

    with caplog.at_level(logging.ERROR, logger="tifffile"):
        back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, data)
    assert [r.getMessage() for r in caplog.records if r.name == "tifffile"] == []
    if not plain_open_logged:
        pytest.skip("this tifffile reads the file as ImageJ first: no error to avoid")


def test_an_imagej_z_stack_is_one_channel(tmp_path):
    data = planes(1, 5)
    path = str(tmp_path / "zstack.tif")
    tifffile.imwrite(path, data[0], imagej=True, metadata={"axes": "ZYX"})
    np.testing.assert_array_equal(FluorescenceImage.load(path).data, data)


def test_an_rgb_tiff_is_three_channels(tmp_path):
    rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    path = str(tmp_path / "export.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")

    back = FluorescenceImage.load(path)

    assert back.data.shape == (3, 1, HEIGHT, WIDTH)
    assert [int(back.data[c, 0, 0, 0]) for c in range(3)] == [10, 20, 30]
    assert len(back.metadata.channels) == 3


def test_a_stack_of_rgb_pages_is_colour_channels_by_z(tmp_path):
    rgb = np.zeros((4, HEIGHT, WIDTH, 3), dtype=np.uint8)
    for z in range(4):
        rgb[z] = [10 + z, 20 + z, 30 + z]
    path = str(tmp_path / "export-stack.tif")
    tifffile.imwrite(path, rgb, photometric="rgb")

    back = FluorescenceImage.load(path)

    assert back.data.shape == (3, 4, HEIGHT, WIDTH)
    assert int(back.data[2, 3, 0, 0]) == 33  # blue, fourth page


def test_a_plain_stack_without_metadata_is_still_a_z_stack(tmp_path):
    data = planes(1, 5)[0]
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, data, photometric="minisblack")

    back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, data[None])


def test_a_broken_annotation_of_ours_falls_back_to_the_ome(tmp_path):
    path = str(tmp_path / "stack.ome.tiff")
    _image(planes(2, 3)).save(path)
    with tifffile.TiffFile(path) as tif:
        xml = tif.pages[0].description
    marker = json.dumps("FluorescenceImageMetadata")[1:-1]
    assert marker in xml
    # Corrupt the JSON the annotation carries but leave the OME itself valid.
    start = xml.index(">", xml.index(marker)) + 1
    tifffile.tiffcomment(path, xml[:start] + "{not json" + xml[start:])

    back = FluorescenceImage.load(path)

    np.testing.assert_array_equal(back.data, planes(2, 3))
    assert len(back.metadata.channels) == 2


# ── the axis arrangement on its own ─────────────────────────────────────────


@pytest.mark.parametrize(
    "axes, shape, expected",
    [
        ("YX", (4, 5), (1, 1, 4, 5)),
        ("ZYX", (3, 4, 5), (1, 3, 4, 5)),
        ("CYX", (2, 4, 5), (2, 1, 4, 5)),
        ("ZCYX", (3, 2, 4, 5), (2, 3, 4, 5)),
        ("CZYX", (2, 3, 4, 5), (2, 3, 4, 5)),
        ("YXS", (4, 5, 3), (3, 1, 4, 5)),
        ("IYXS", (2, 4, 5, 3), (3, 2, 4, 5)),
        ("QYX", (3, 4, 5), (1, 3, 4, 5)),  # unnamed: z, as always
        ("IYX", (3, 4, 5), (1, 3, 4, 5)),
        ("QQYX", (2, 3, 4, 5), (2, 3, 4, 5)),  # two unnamed: channel then z
        ("TZCYX", (1, 3, 2, 4, 5), (2, 3, 4, 5)),  # one time point: dropped
        ("TCZYX", (2, 2, 3, 4, 5), (2, 2, 3, 4, 5)),  # several: kept first
    ],
)
def test_axes_are_arranged_as_czyx(axes, shape, expected):
    data = np.arange(int(np.prod(shape))).reshape(shape)
    out = to_czyx(data, axes)
    assert out.shape == expected
    # Same values, only moved: a plane picked by its role is the same plane.
    if axes == "ZCYX":
        np.testing.assert_array_equal(out[1, 2], data[2, 1])


@pytest.mark.parametrize(
    "axes, shape",
    [
        ("QQQYX", (2, 2, 2, 4, 5)),  # more unnamed axes than roles to give them
        ("CYXS", (2, 4, 5, 3)),  # channels and colour samples at once
        ("ZYX", (4, 5)),  # the axes do not describe the array
        ("CZY", (2, 3, 4)),  # no x
    ],
)
def test_axes_that_cannot_be_arranged_refuse(axes, shape):
    with pytest.raises(ValueError):
        to_czyx(np.zeros(shape), axes)

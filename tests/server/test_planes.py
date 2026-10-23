"""An image's raw planes, as the web viewer reads them to draw itself."""

import os
import shutil

import numpy as np
import pytest

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.server import planes
from fibsem.server.files import FileRefusal

V0 = os.path.join(os.path.dirname(__file__), "..", "fixtures", "image_files_v0")


def _stack(path, planes_=5, size=64):
    channels = [
        FluorescenceChannelMetadata(
            name=name,
            excitation_wavelength=nm,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=color,
        )
        for name, nm, color in (("GFP", 488.0, "green"), ("RFP", 561.0, "red"))
    ]
    rng = np.random.default_rng(3)
    data = rng.integers(0, 4000, (2, planes_, size, size), dtype=np.uint16)
    metadata = FluorescenceImageMetadata(
        pixel_size_x=100e-9,
        pixel_size_y=100e-9,
        pixel_size_z=500e-9,
        channels=channels,
        z_positions=[i * 500e-9 for i in range(planes_)],
    )
    FluorescenceImage(data=data, metadata=metadata).save(path)
    return data


def test_a_plane_is_the_values_as_saved(tmp_path):
    data = _stack(str(tmp_path / "s.ome.tiff"))
    header, out = planes.read(str(tmp_path), "s.ome.tiff", z=2)
    assert header["shape"] == [2, 1, 64, 64]
    assert header["dtype"] == "uint16"
    assert header["channels"] == ["GFP", "RFP"]
    assert header["full_shape"] == [64, 64]
    np.testing.assert_array_equal(out[:, 0], data[:, 2])


def test_projections_over_a_range_of_planes(tmp_path):
    data = _stack(str(tmp_path / "s.ome.tiff"))
    root = str(tmp_path)
    _, top = planes.read(root, "s.ome.tiff")
    np.testing.assert_array_equal(top[:, 0], data.max(axis=1))
    _, low = planes.read(root, "s.ome.tiff", z="min", first=1, last=3)
    np.testing.assert_array_equal(low[:, 0], data[:, 1:4].min(axis=1))
    header, mean = planes.read(root, "s.ome.tiff", z="mean", channels=["RFP"])
    assert header["dtype"] == "float32" and header["channels"] == ["RFP"]
    np.testing.assert_allclose(mean[0, 0], data[1].mean(axis=0), rtol=1e-5)


def test_every_plane_small_from_the_files_own_copies(tmp_path):
    data = _stack(str(tmp_path / "s.ome.tiff"), size=128)
    header, out = planes.read(str(tmp_path), "s.ome.tiff", factor=8, z="all")
    assert header["shape"] == [2, 5, 16, 16] and header["factor"] == 8
    assert header["full_shape"] == [128, 128]
    # a block mean of the full plane, as the file's copy is made
    block = data[0, 4, :8, :8].mean()
    assert out[0, 4, 0, 0] == pytest.approx(block, abs=1)


def test_an_old_stack_gets_its_small_copies_made(tmp_path):
    shutil.copy(os.path.join(V0, "fm_v0_stack.ome.tiff"), tmp_path / "old.ome.tiff")
    _, whole = planes.read(str(tmp_path), "old.ome.tiff", z="all")
    header, small = planes.read(str(tmp_path), "old.ome.tiff", factor=4, z="all")
    c, z, y, x = whole.shape
    assert header["shape"] == [c, z, -(-y // 4), -(-x // 4)]
    plane = planes.read(str(tmp_path), "old.ome.tiff", factor=4, z=1)[1]
    np.testing.assert_array_equal(plane[:, 0], small[:, 1])


def test_a_beam_image_is_one_grey_layer(tmp_path):
    shutil.copy(os.path.join(V0, "fib_v0_ion_uint16.tif"), tmp_path / "ref_ib.tif")
    import tifffile

    data = tifffile.imread(tmp_path / "ref_ib.tif")
    header, out = planes.read(str(tmp_path), "ref_ib.tif")
    assert header["shape"] == [1, 1, *data.shape]
    np.testing.assert_array_equal(out[0, 0], data)


@pytest.mark.parametrize(
    "ask",
    [
        {"factor": 2},
        {"z": 9},
        {"z": "median"},
        {"z": "max", "first": 3, "last": 1},
        {"channels": ["DAPI"]},
    ],
)
def test_what_is_not_in_the_stack_is_refused(tmp_path, ask):
    _stack(str(tmp_path / "s.ome.tiff"))
    with pytest.raises(FileRefusal) as refused:
        planes.read(str(tmp_path), "s.ome.tiff", **ask)
    assert refused.value.status == 422


def test_a_beam_image_has_no_planes_to_pick(tmp_path):
    shutil.copy(os.path.join(V0, "fib_v0_ion_uint16.tif"), tmp_path / "ref_ib.tif")
    with pytest.raises(FileRefusal):
        planes.read(str(tmp_path), "ref_ib.tif", z=3)

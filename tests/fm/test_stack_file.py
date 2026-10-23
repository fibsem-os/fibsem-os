"""Reading a fluorescence stack a plane at a time, and the preview built on it."""

import os

import numpy as np
import pytest

from fibsem.fm import preview
from fibsem.fm.preview import load_stack_preview, render_stack
from fibsem.fm.stack_file import StackFile
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)


def _image(shape) -> FluorescenceImage:
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, shape).astype(np.uint16)
    channels = [
        FluorescenceChannelMetadata(
            name=f"Channel-{i:02d}",
            excitation_wavelength=488.0,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color=color,
        )
        for i, color in zip(range(shape[0]), ["red", "green", "cyan", "magenta"])
    ]
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-01-01T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(shape[3], shape[2]),
        channels=channels,
    )
    return FluorescenceImage(data=data, metadata=metadata)


def _save_before_0_60(image: FluorescenceImage, path: str) -> None:
    """As `FluorescenceImage.save` wrote a stack before 0.60: contiguous, no small copies."""
    import tifffile

    nc, nz, ny, nx = image.data.shape
    with tifffile.TiffWriter(path) as tif:
        tif.write(image.data.reshape(1, nc, nz, ny, nx), contiguous=True, photometric="minisblack")
        tif.overwrite_description(image.get_ome_metadata().to_xml())


@pytest.fixture(params=["0.60", "before 0.60"])
def saved(request, tmp_path):
    def _save(shape):
        image = _image(shape)
        path = str(tmp_path / f"stack-{'x'.join(map(str, shape))}.ome.tiff")
        if request.param == "0.60":
            image.save(path)
        else:
            _save_before_0_60(image, path)
        return path, image

    return _save


@pytest.mark.parametrize("shape", [(3, 5, 16, 20), (1, 4, 16, 20), (2, 1, 16, 20)])
def test_every_plane_is_the_plane_load_reads(saved, shape):
    path, _ = saved(shape)
    whole = FluorescenceImage.load(path)
    with StackFile(path) as stack:
        assert stack.shape == whole.data.shape
        assert [c.name for c in stack.metadata.channels] == [
            c.name for c in whole.metadata.channels
        ]
        for c in range(stack.channels):
            for z in range(stack.planes):
                np.testing.assert_array_equal(stack.plane(c, z), whole.data[c, z])


def test_the_preview_is_the_render_of_the_whole_stack(saved):
    path, _ = saved((3, 5, 16, 20))
    whole = FluorescenceImage.load(path)
    for z, channels in [
        (None, None),
        (2, None),
        (None, ["Channel-01"]),
        (4, ["Channel-00", "Channel-02"]),
    ]:
        rgb, _, _ = load_stack_preview(path, z=z, channels=channels)
        np.testing.assert_array_equal(rgb, render_stack(whole, z=z, channels=channels))


def _counting_reads(monkeypatch):
    reads = []
    real = StackFile.plane

    def plane(self, channel, z):
        reads.append((channel, z))
        return real(self, channel, z)

    monkeypatch.setattr(StackFile, "plane", plane)
    return reads


def test_a_plane_reads_only_its_shown_channels_once_the_stack_is_summarised(
    saved, monkeypatch
):
    path, _ = saved((3, 5, 16, 20))
    reads = _counting_reads(monkeypatch)
    load_stack_preview(path)
    assert len(reads) == 15  # one pass for the projection and the sharpness

    reads.clear()
    load_stack_preview(path, z=3, channels=["Channel-01"])
    assert reads == [(1, 3)]


def test_a_changed_file_is_summarised_again(saved, monkeypatch):
    path, image = saved((2, 3, 16, 20))
    reads = _counting_reads(monkeypatch)
    load_stack_preview(path)
    image.data[:] = 0
    image.save(path)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    reads.clear()

    rgb, _, _ = load_stack_preview(path)

    assert len(reads) == 6
    assert rgb.max() == 0


def test_an_unknown_channel_or_plane_names_the_file(saved):
    path, _ = saved((2, 3, 16, 20))
    with pytest.raises(ValueError, match="no channel.*stack-2x3"):
        load_stack_preview(path, channels=["DAPI"])
    with pytest.raises(ValueError, match="outside the stack.*stack-2x3"):
        load_stack_preview(path, z=3)


def test_only_the_last_few_stacks_are_kept(saved, monkeypatch):
    monkeypatch.setattr(preview, "_SUMMARIES_KEPT", 2)
    monkeypatch.setattr(preview, "_SUMMARIES", type(preview._SUMMARIES)())
    for planes in (2, 3, 4):
        load_stack_preview(saved((1, planes, 8, 8))[0])
    assert len(preview._SUMMARIES) == 2

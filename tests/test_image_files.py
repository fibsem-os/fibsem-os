"""Image files: what 0.60 writes, and that every file written before it still reads
exactly as it did (tests/fixtures/image_files_v0, saved by the pre-0.60 savers)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from ome_types import from_xml

from fibsem import image_format
from fibsem.fm.structures import FluorescenceImage
from fibsem.structures import FibsemImage, FibsemImageMetadata
from fibsem.workflows.image_positions import read_image_position

V0 = Path(__file__).parent / "fixtures" / "image_files_v0"
EXPECTED = json.loads((V0 / "expected.json").read_text())


def _digest(data: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(data).tobytes()).hexdigest()


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_a_file_saved_before_0_60_loads_as_it_was_saved(name):
    expected = EXPECTED[name]
    loader = FluorescenceImage if name.startswith("fm_") else FibsemImage
    image = loader.load(str(V0 / name))

    assert list(image.data.shape) == expected["shape"]
    assert image.data.dtype.name == expected["dtype"]
    assert _digest(image.data) == expected["sha256"]
    metadata = None if image.metadata is None else image.metadata.to_dict()
    assert json.loads(json.dumps(metadata, default=str)) == expected["metadata"]


@pytest.mark.parametrize("name", [n for n in sorted(EXPECTED) if n.startswith("fib_")])
def test_a_beam_image_saved_before_0_60_gives_its_metadata_and_position(name):
    expected = EXPECTED[name]["metadata"]
    read = FibsemImageMetadata.read(str(V0 / name))

    assert (read and read.to_dict()) == (
        expected and FibsemImageMetadata.from_dict(expected).to_dict()
    )
    position = read_image_position(str(V0 / name)).stage_position
    if expected is None:
        assert position is None
    else:
        assert position.x == pytest.approx(
            expected["microscope_state"]["stage_position"]["x"]
        )


def _beam_image(shape, dtype=np.uint16) -> FibsemImage:
    template = FibsemImage.load(str(V0 / "fib_v0_ion_uint16.tif"))
    rng = np.random.default_rng(1)
    data = rng.integers(0, np.iinfo(dtype).max, shape, dtype=dtype)
    return FibsemImage(data=data, metadata=template.metadata)


@pytest.mark.parametrize("shape", [(64, 96), (2100, 3000)], ids=["small", "large"])
def test_a_beam_image_saves_and_loads_back_unchanged(tmp_path, shape):
    image = _beam_image(shape)

    path = image.save(str(tmp_path / "image.tif"))
    back = FibsemImage.load(path)

    np.testing.assert_array_equal(back.data, image.data)
    assert back.metadata.to_dict() == image.metadata.to_dict()
    assert FibsemImageMetadata.read(path).to_dict() == image.metadata.to_dict()
    assert (
        read_image_position(path).stage_position
        == image.metadata.microscope_state.stage_position
    )


def test_a_large_beam_image_is_tiled_and_carries_its_small_copies(tmp_path):
    image = _beam_image((2100, 3000))

    path = image.save(str(tmp_path / "image.tif"))

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        assert series.shape == (2100, 3000)
        assert [level.shape for level in series.levels] == [
            (2100, 3000),
            (1050, 1500),
            (525, 750),
            (263, 375),
        ]
        page = tif.pages.first
        assert page.is_tiled and page.tilewidth == image_format.TILE
        assert page.compression.name == "ADOBE_DEFLATE"
        np.testing.assert_array_equal(
            series.levels[3].asarray(), _block_mean(image.data, 8)
        )


def _block_mean(data: np.ndarray, factor: int) -> np.ndarray:
    from fibsem.imaging.reduce import reduce_by

    out = data
    for _ in range(int(np.log2(factor))):
        out = reduce_by(out, 2)
    return out


def test_a_small_beam_image_is_compressed_without_copies(tmp_path):
    path = _beam_image((64, 96)).save(str(tmp_path / "image.tif"))

    with tifffile.TiffFile(path) as tif:
        assert len(tif.series[0].levels) == 1
        assert not tif.pages.first.is_tiled
        assert tif.pages.first.compression.name == "ADOBE_DEFLATE"


def test_a_beam_image_carries_ome_other_readers_understand(tmp_path):
    image = _beam_image((64, 96))

    path = image.save(str(tmp_path / "image.tif"))

    with tifffile.TiffFile(path) as tif:
        assert tif.is_ome
        ome = from_xml(tif.pages.first.description)
    pixels = ome.images[0].pixels
    assert (pixels.size_x, pixels.size_y) == (96, 64)
    assert pixels.physical_size_x == pytest.approx(image.metadata.pixel_size.x)
    assert pixels.planes[0].position_x == pytest.approx(
        image.metadata.microscope_state.stage_position.x
    )
    assert pixels.channels[0].name == "Ion ETD SecondaryElectrons"
    assert ome.instruments[0].microscope.model == "DemoMicroscope"
    values = ome.structured_annotations.map_annotations[0].value
    assert values[image_format.FORMAT_KEY] == "1"
    assert json.loads(values[image_format.METADATA_KEY]) == image.metadata.to_dict()


def test_a_beam_image_with_no_metadata_saves_and_loads_with_none(tmp_path):
    image = FibsemImage(data=np.zeros((8, 8), dtype=np.uint8), metadata=None)

    back = FibsemImage.load(image.save(str(tmp_path / "bare.tif")))

    assert back.metadata is None
    np.testing.assert_array_equal(back.data, image.data)


def test_an_fm_stack_carries_its_small_copies_and_format(tmp_path):
    image = FluorescenceImage.load(str(V0 / "fm_v0_stack.ome.tiff"))

    path = image.save(str(tmp_path / "stack.ome.tiff"))

    back = FluorescenceImage.load(path)
    np.testing.assert_array_equal(back.data, image.data)
    assert back.metadata.to_dict() == image.metadata.to_dict()
    with tifffile.TiffFile(path) as tif:
        assert [level.shape[-2:] for level in tif.series[0].levels] == [
            (48, 64),
            (24, 32),
            (12, 16),
            (6, 8),
        ]
        assert (
            image_format.annotation(
                tif.pages.first.description, image_format.FORMAT_KEY
            )
            == "1"
        )


def test_a_single_small_fm_image_has_no_copies(tmp_path):
    image = FluorescenceImage.load(str(V0 / "fm_v0_single.ome.tiff"))

    path = image.save(str(tmp_path / "single.ome.tiff"))

    with tifffile.TiffFile(path) as tif:
        assert len(tif.series[0].levels) == 1


@pytest.mark.parametrize(
    "description", [None, "", "not json", "[1, 2]", "<OME/>", "<not xml"]
)
def test_a_description_with_no_beam_metadata_reads_as_none(description):
    assert image_format.metadata_dict(description) is None


def test_an_fm_stack_records_its_channels_in_the_colours_the_apps_draw(tmp_path):
    image = FluorescenceImage.load(str(V0 / "fm_v0_stack.ome.tiff"))
    assert [c.color for c in image.metadata.channels] == ["gray", "gray"]

    path = image.save(str(tmp_path / "stack.ome.tiff"))

    with tifffile.TiffFile(path) as tif:
        channels = from_xml(tif.pages.first.description).images[0].pixels.channels
    assert [c.color.as_rgb_tuple()[:3] for c in channels] == [(128, 128, 128), (0, 128, 0)]
    assert [c.color for c in FluorescenceImage.load(path).metadata.channels] == ["gray", "gray"]

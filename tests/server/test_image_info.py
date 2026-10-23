"""The bar under an image, as the server answers it: what the Qt view bar shows."""

import os

import numpy as np
import pytest

from fibsem import acquire, utils
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.imaging.export import BAR_DEFAULT_FIELDS, bar_selection, image_fields
from fibsem.server import files
from fibsem.server.image_payloads import fields_info, image_info
from fibsem.structures import BeamType, FibsemImage, ImageSettings


@pytest.fixture(scope="module")
def fib_image():
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    settings = ImageSettings(
        resolution=(96, 64), hfw=80e-6, beam_type=BeamType.ION, save=False
    )
    return acquire.acquire_image(microscope, settings)


def _stack(planes=3):
    channels = [
        FluorescenceChannelMetadata(
            name="GFP",
            excitation_wavelength=488.0,
            power=0.5,
            exposure_time=0.1,
            gain=1.0,
            offset=0.0,
            color="green",
            objective_magnification=100,
            objective_numerical_aperture=0.75,
        )
    ]
    metadata = FluorescenceImageMetadata(
        acquisition_datetime="2026-10-23T03:40:00",
        pixel_size_x=110e-9,
        pixel_size_y=110e-9,
        pixel_size_z=500e-9,
        channels=channels,
        z_positions=[i * 500e-9 for i in range(planes)],
    )
    data = np.zeros((1, planes, 48, 64), np.uint16)
    return FluorescenceImage(data=data, metadata=metadata)


def test_a_beam_image_says_what_the_qt_bar_says(fib_image):
    info = image_info(fib_image)
    header, labelled, time = bar_selection(image_fields(fib_image), BAR_DEFAULT_FIELDS)
    assert info["kind"] == "FIB"
    assert [f["value"] for f in info["header"]] == [f.value for f in header]
    assert [(f["label"], f["value"]) for f in info["fields"]] == [
        (f.label, f.value) for f in labelled
    ]
    assert info["time"] == time
    assert {f["key"]: f["title"] for f in info["fields"]}["hfw"] == (
        "Horizontal field width"
    )


def test_the_bar_follows_the_fields_someone_chose(fib_image, monkeypatch):
    monkeypatch.setattr(
        "fibsem.imaging.export.stored_bar_keys",
        lambda kind: ("dwell_time", "hfw") if kind == "FIB" else None,
    )
    info = image_info(fib_image)
    assert [f["key"] for f in info["fields"]] == ["dwell_time", "hfw"]
    assert info["header"] == [] and info["time"] is None


def test_the_scalebar_is_a_round_fifth_of_the_field(fib_image):
    # 80 µm across: a fifth is 16 µm, the nearest round length 20 µm.
    scalebar = image_info(fib_image)["scalebar"]
    assert scalebar["length"] == pytest.approx(20e-6)
    assert scalebar["label"].startswith("20 ")


def test_without_a_pixel_size_there_is_no_scalebar(fib_image):
    assert fields_info(image_fields(fib_image), None, 96)["scalebar"] is None


def test_an_fm_stack_names_the_plane_shown(tmp_path):
    path = str(tmp_path / "stack.ome.tiff")
    _stack().save(path)
    whole = files.read(str(tmp_path), "stack.ome.tiff")["info"]
    plane = files.read(str(tmp_path), "stack.ome.tiff", z=1)["info"]
    assert whole["kind"] == "FM"
    assert [f["value"] for f in whole["header"]] == ["100× · 0.75 NA"]
    z = {f["key"]: f["value"] for f in whole["fields"]}["z"]
    assert z.startswith("MIP · 3 × 500")
    assert {f["key"]: f["value"] for f in plane["fields"]}["z"].startswith("2 of 3")


def test_a_saved_beam_image_answers_its_bar(tmp_path, fib_image):
    path = os.path.join(str(tmp_path), "ref_ib.tif")
    fib_image.save(path)
    answer = files.read(str(tmp_path), "ref_ib.tif")
    assert answer["info"] == image_info(FibsemImage.load(path))


def test_a_plain_picture_has_no_bar(tmp_path):
    from PIL import Image

    Image.new("L", (32, 16)).save(tmp_path / "thumb.png")
    assert files.read(str(tmp_path), "thumb.png")["info"] is None

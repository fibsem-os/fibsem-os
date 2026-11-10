"""One provenance view across image kinds: a beam image and an FM stack answer
the same `ImageProvenance`, built from the fields each already saves."""

import numpy as np
import pytest

from fibsem import acquire, utils
from fibsem.fm.preview import _facts
from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImageMetadata,
)
from fibsem.server.images import image_metadata
from fibsem.structures.image import FibsemImageMetadata
from fibsem.structures.provenance import ImageProvenance, InstrumentRef
from fibsem.structures.settings import BeamType, FibsemStagePosition, ImageSettings
from fibsem.structures.system import FibsemExperimentRef

REF = FibsemExperimentRef(
    id="e1", name="run", item_name="Lamella-01", task_name="Rough Milling"
)


@pytest.fixture(scope="module")
def beam_metadata() -> FibsemImageMetadata:
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    microscope.experiment = REF
    settings = ImageSettings(
        resolution=(96, 64), hfw=80e-6, beam_type=BeamType.ION, save=False
    )
    image = acquire.acquire_image(microscope, settings)
    return image.metadata


def _fm_metadata(**fields) -> FluorescenceImageMetadata:
    channel = FluorescenceChannelMetadata(
        name="GFP",
        excitation_wavelength=488.0,
        power=0.5,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
    )
    return FluorescenceImageMetadata(
        pixel_size_x=110e-9, pixel_size_y=110e-9, channels=[channel], **fields
    )


def test_a_beam_image_answers_its_provenance(beam_metadata):
    provenance = beam_metadata.provenance
    assert isinstance(provenance, ImageProvenance)
    assert provenance.instrument.manufacturer == "Demo"
    assert provenance.instrument.serial_number == (
        beam_metadata.system_info.serial_number
    )
    assert (provenance.item, provenance.task) == ("Lamella-01", "Rough Milling")
    assert provenance.acquired_at == beam_metadata.acquisition_date
    assert provenance.stage_position == beam_metadata.stage_position
    assert provenance.geometry == beam_metadata.hardware_geometry
    assert provenance.user is beam_metadata.user


def test_an_fm_stack_answers_the_same_shape():
    position = FibsemStagePosition(x=1e-3, y=2e-3, z=0, r=0, t=0)
    metadata = _fm_metadata(
        acquisition_datetime="2026-10-23T03:40:00+11:00",
        stage_position=position,
        experiment=REF,
        system_info={"manufacturer": "Demo", "model": "FM", "serial_number": "7"},
    )
    provenance = metadata.provenance
    assert provenance.instrument == InstrumentRef(
        manufacturer="Demo", model="FM", serial_number="7"
    )
    assert (provenance.item, provenance.task) == ("Lamella-01", "Rough Milling")
    assert provenance.acquired_at.isoformat() == "2026-10-23T03:40:00+11:00"
    assert provenance.stage_position == position
    assert provenance.user is None


def test_an_older_fm_file_with_nothing_recorded_reads_as_none():
    provenance = _fm_metadata().provenance
    assert provenance == ImageProvenance()
    assert (provenance.item, provenance.task) == (None, None)


def test_both_answer_blocks_carry_the_same_provenance_keys(beam_metadata):
    class _Image:
        data = np.zeros((64, 96), np.uint8)
        metadata = beam_metadata

    beam = image_metadata(_Image())
    fm = _facts(_fm_metadata(experiment=REF), count=1, planes=1, sharpness=None)
    for key in ("acquired_at", "item", "task"):
        assert key in beam and key in fm
    assert (fm["item"], fm["task"]) == (beam["item"], beam["task"])

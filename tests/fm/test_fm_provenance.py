"""A fluorescence image should say what produced it, like a beam image does.

FM had the geometry (FIB-481) but not the experiment, so a forwarded OME-TIFF could
say which instrument took it and under what arrangement, but not which run, lamella or
task. Same record, same reason: the file has to stand alone. See FIB-466.
"""

import os
import tempfile

import numpy as np
import pytest

from fibsem.fm.structures import (
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
)
from fibsem.structures import FibsemExperimentRef


def _experiment() -> FibsemExperimentRef:
    ref = FibsemExperimentRef(id="exp-1", name="an-experiment")
    ref.set_workflow_metadata(
        item_id="lam-1",
        item_name="grand-dodo",
        task_id="task-9",
        task_name="FM Acquisition",
    )
    return ref


def _channel() -> FluorescenceChannelMetadata:
    return FluorescenceChannelMetadata(
        name="GFP",
        color="#00FF00",
        excitation_wavelength=488,
        emission_wavelength=509,
        power=1.0,
        exposure_time=0.1,
        gain=1.0,
        offset=0.0,
    )


def _image(experiment=None, z: int = 1) -> FluorescenceImage:
    metadata = FluorescenceImageMetadata(
        acquisition_date="2026-08-05T00:00:00",
        pixel_size_x=1e-7,
        pixel_size_y=1e-7,
        resolution=(16, 16),
        channels=[_channel()],
        experiment=experiment,
    )
    return FluorescenceImage(
        data=np.zeros((1, z, 16, 16), dtype=np.uint16), metadata=metadata
    )


def test_an_fm_image_can_carry_the_experiment() -> None:
    image = _image(_experiment())

    assert image.metadata.experiment.item_name == "grand-dodo"
    assert image.metadata.experiment.task_name == "FM Acquisition"


def test_it_survives_an_ome_tiff_round_trip() -> None:
    """The whole point: the record has to reach whoever receives the file.

    It rides in the `FluorescenceImageMetadata` map annotation, which `load` reads in
    preference to reconstructing from OME's own fields.
    """
    experiment = _experiment()

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "fm.ome.tiff")
        _image(experiment).save(path)
        reloaded = FluorescenceImage.load(path)

    assert reloaded.metadata.experiment == experiment


def test_an_image_without_one_round_trips_as_none() -> None:
    """Acquired outside a workflow, or written by other software."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "fm.ome.tiff")
        _image(None).save(path)
        reloaded = FluorescenceImage.load(path)

    assert reloaded.metadata.experiment is None


def test_a_projection_keeps_what_produced_it() -> None:
    """A projection of a lamella's image is still that lamella's."""
    projected = _image(_experiment(), z=3).max_intensity_projection()

    assert projected.metadata.experiment.item_name == "grand-dodo"
    assert projected.metadata.experiment.id == "exp-1"


def test_a_projection_keeps_the_geometry_it_was_captured_under() -> None:
    """Separate bug, found alongside: the derived paths dropped `geometry`.

    They carried `stage_position` and `system_info` forward but not this, so a
    projected image could not be reprojected -- `fm/reprojection.py` raises on a
    missing geometry rather than guessing, since a marker drawn in the wrong place
    looks exactly like one drawn in the right place.
    """
    from fibsem.structures import FibsemHardwareGeometry

    image = _image(_experiment(), z=3)
    image.metadata.geometry = FibsemHardwareGeometry(
        shuttle_pre_tilt=35.0, is_compustage=True
    )

    projected = image.max_intensity_projection()

    assert projected.metadata.geometry is not None
    assert projected.metadata.geometry.shuttle_pre_tilt == 35.0
    assert projected.metadata.geometry.is_compustage is True


def test_a_focus_stack_keeps_both() -> None:
    """The other derived path, which had the same gap."""
    from fibsem.structures import FibsemHardwareGeometry

    image = _image(_experiment(), z=3)
    image.metadata.geometry = FibsemHardwareGeometry(shuttle_pre_tilt=35.0)

    stacked = image.focus_stack()

    assert stacked.metadata.experiment.item_name == "grand-dodo"
    assert stacked.metadata.geometry.shuttle_pre_tilt == 35.0


@pytest.mark.parametrize("missing", [{}, {"experiment": None}])
def test_metadata_written_before_this_existed_still_loads(missing: dict) -> None:
    """`from_dict` is how a saved FM image is read back."""
    base = _image(_experiment()).metadata.to_dict()
    base.pop("experiment", None)
    base.update(missing)

    metadata = FluorescenceImageMetadata.from_dict(base)

    assert metadata.experiment is None
    assert metadata.pixel_size_x == 1e-7

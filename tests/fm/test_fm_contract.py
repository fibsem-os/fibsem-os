"""Contract tests for the FluorescenceMicroscope base classes (FIB-288).

The base classes double as the simulator and as the contract every driver
implements (and, later, the wire contract for remote control). These tests
pin the members promoted into the ABCs and the per-instance acquisition
state, using the simulated implementations directly.

TFS-driver conformance can only run on machines with the autoscript SDK;
odemis-driver conformance is covered in test_odemis_driver.py.
"""

import numpy as np
import pytest

from fibsem.fm.microscope import (
    BINNING_VALUES,
    SIM_CAMERA_EXPOSURE_LIMITS,
    FluorescenceMicroscope,
)


@pytest.fixture()
def fm():
    return FluorescenceMicroscope()


class TestPromotedContract:
    """power_limits / exposure_time_limits / available_binnings / state."""

    def test_camera_available_binnings(self, fm):
        assert fm.camera.available_binnings == tuple(BINNING_VALUES)

    def test_camera_exposure_time_limits(self, fm):
        assert fm.camera.exposure_time_limits == SIM_CAMERA_EXPOSURE_LIMITS

    def test_light_source_power_limits_fraction(self, fm):
        assert fm.light_source.power_limits == (0.0, 1.0)

    def test_binning_setter_rejects_unsupported(self, fm):
        with pytest.raises(ValueError):
            fm.camera.binning = 3

    def test_binning_setter_accepts_supported(self, fm):
        fm.camera.binning = 2
        assert fm.camera.binning == 2

    def test_objective_state_is_a_string_property(self, fm):
        assert isinstance(fm.objective.state, str)
        assert fm.objective.state in ("Inserted", "Retracted", "Busy", "Error", "Other")


class TestPerInstanceAcquisitionState:
    """The stop event and thread handle must not be shared between instances."""

    def test_stop_events_are_independent(self):
        fm_a = FluorescenceMicroscope()
        fm_b = FluorescenceMicroscope()
        assert fm_a._stop_acquisition_event is not fm_b._stop_acquisition_event
        fm_a._stop_acquisition_event.set()
        assert not fm_b._stop_acquisition_event.is_set()

    def test_thread_handles_are_independent(self):
        fm_a = FluorescenceMicroscope()
        fm_b = FluorescenceMicroscope()
        fm_a._acquisition_thread = object()
        assert fm_b._acquisition_thread is None


class TestConstructImageFrameMetadata:
    """Optional per-frame metadata overrides the state snapshot (F20)."""

    def test_no_frame_metadata_uses_state_snapshot(self, fm):
        image = fm._construct_image(np.zeros((8, 8), dtype=np.uint16))
        assert image.metadata.pixel_size_x == pytest.approx(fm.camera.pixel_size[0])

    def test_frame_metadata_overrides_geometry_and_timing(self, fm):
        frame_metadata = {
            "pixel_size": (5e-8, 6e-8),
            "acquisition_date": "2026-07-22T10:00:00",
            "exposure_time": 0.42,
        }
        image = fm._construct_image(
            np.zeros((8, 8), dtype=np.uint16), frame_metadata=frame_metadata
        )
        assert image.metadata.pixel_size_x == pytest.approx(5e-8)
        assert image.metadata.pixel_size_y == pytest.approx(6e-8)
        assert image.metadata.acquisition_date == "2026-07-22T10:00:00"
        assert image.metadata.channels[0].exposure_time == pytest.approx(0.42)

    def test_partial_frame_metadata_only_overrides_present_keys(self, fm):
        image = fm._construct_image(
            np.zeros((8, 8), dtype=np.uint16),
            frame_metadata={"exposure_time": 0.42},
        )
        # exposure overridden, geometry still from the state snapshot
        assert image.metadata.channels[0].exposure_time == pytest.approx(0.42)
        assert image.metadata.pixel_size_x == pytest.approx(fm.camera.pixel_size[0])

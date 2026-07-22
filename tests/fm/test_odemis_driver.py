"""Unit tests for the Odemis fluorescence microscope driver (fibsem.fm.odemis).

Runs everywhere: odemis is replaced by the stub modules in _odemis_stubs.py,
which mimic the verified odemis behaviour (SI units, binning-coupled camera
geometry, favourite positions). See docs/design/odemis-fm-driver.md.

Each test class covers a finding from Phase 1 (FIB-285) or Phase 2 (FIB-286):
- F1  wavelength setters select the requested band (metres-vs-nm regression)
- F2  objective.state is a property with the base Literal values
- F3  camera geometry reads live, no binning double-count
- F4  power is a fraction (0-1) of maximum, normalised to watts
- F5/F6 objective limits from the focuser axis range + move clipping
- F7  init tolerates missing metadata keys (baseline, FAV positions, pixel size)
- F7b add_odemis_path is safe without /etc/odemis.conf
- F8  acquire_image raises on failure instead of returning None
- F9  'Fluorescence' emission maps to the band matching the excitation
- F10 live acquisition via dataflow subscription (stream active once,
      frames pushed, light off on stop in all paths)
- F11-F13 gain wiring via hasVA, binning/exposure limits from the VAs
- F20 per-frame DataArray metadata overrides the state snapshot
"""

import sys
import time
import types
from datetime import datetime

import numpy as np
import pytest

from fibsem.fm.structures import ChannelSettings, FluorescenceImage
from tests.fm import _odemis_stubs as stubs


def wait_until(condition, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """Poll condition until true or timeout; returns whether it became true."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


@pytest.fixture(scope="module")
def odemis_env():
    """Install stub odemis modules and import the driver against them."""
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    stubs.install_odemis_stubs()
    import fibsem.fm.odemis as fm_odemis  # noqa: F401 (imported against stubs)

    yield types.SimpleNamespace(module=fm_odemis, stubs=stubs)

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


@pytest.fixture()
def fm(odemis_env):
    """Fresh stub components + OdemisFluorescenceMicroscope per test."""
    components = stubs.use_components(stubs.default_components())
    microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
    microscope.components = components  # convenience handle for tests
    return microscope


class TestAddOdemisPath:
    """F7b: add_odemis_path must be safe without an odemis config."""

    def test_missing_config_is_noop(self, odemis_env, tmp_path):
        from fibsem.microscopes.odemis_microscope import add_odemis_path

        before = list(sys.path)
        add_odemis_path(config_path=str(tmp_path / "missing.conf"))
        assert sys.path == before

    def test_valid_config_appends_devpath(self, odemis_env, tmp_path):
        from fibsem.microscopes.odemis_microscope import add_odemis_path

        conf = tmp_path / "odemis.conf"
        conf.write_text('DEVPATH="/home/dev"\nMODEL="/usr/share/odemis/meteor.yaml"\n')
        before = list(sys.path)
        try:
            add_odemis_path(config_path=str(conf))
            assert "/home/dev/odemis/src" in sys.path
        finally:
            sys.path[:] = before


class TestFilterSetWavelengths:
    """F1: setters must select the requested band, not the lowest one."""

    def test_available_excitation_wavelengths_in_nm(self, fm):
        available = sorted(fm.filter_set.available_excitation_wavelengths)
        assert available == pytest.approx([365, 450, 550, 635])

    def test_available_emission_wavelengths_in_nm(self, fm):
        available = fm.filter_set.available_emission_wavelengths
        assert None in available
        numeric = sorted(wl for wl in available if wl is not None)
        assert numeric == pytest.approx([420, 500, 590, 680])

    def test_excitation_setter_selects_requested_band(self, fm):
        # regression: the old implementation always picked the lowest band
        for requested in (635, 550, 450, 365):
            fm.filter_set.excitation_wavelength = requested
            assert fm.filter_set.excitation_wavelength == pytest.approx(requested)
            band = fm.filter_set._stream.excitation.value
            assert band[2] == pytest.approx(requested * 1e-9)

    def test_excitation_setter_closest_match(self, fm):
        fm.filter_set.excitation_wavelength = 640  # closest available: 635
        assert fm.filter_set.excitation_wavelength == pytest.approx(635)

    def test_excitation_setter_rejects_non_numeric(self, fm):
        with pytest.raises(TypeError):
            fm.filter_set.excitation_wavelength = "red"

    def test_emission_setter_selects_requested_band(self, fm):
        for requested in (680, 500, 590, 420):
            fm.filter_set.emission_wavelength = requested
            assert fm.filter_set.emission_wavelength == pytest.approx(requested)
            band = fm.filter_set._stream.emission.value
            assert band[0] == pytest.approx(requested * 1e-9)

    def test_emission_none_sets_pass_through(self, fm):
        fm.filter_set.emission_wavelength = None
        assert fm.filter_set._stream.emission.value == stubs.BAND_PASS_THROUGH
        assert fm.filter_set.emission_wavelength is None

    def test_emission_setter_rejects_non_numeric_non_str(self, fm):
        with pytest.raises(TypeError):
            fm.filter_set.emission_wavelength = [500]

    def test_emission_setter_skips_when_unchanged(self, fm):
        fm.filter_set.emission_wavelength = 590
        emission_va = fm.filter_set._stream.emission
        sets_before = emission_va.set_count
        fm.filter_set.emission_wavelength = 590  # unchanged: filter wheel is slow
        assert emission_va.set_count == sets_before


class TestObjectiveState:
    """F2: state is a property returning the base contract's Literal values."""

    def test_state_is_a_property_not_a_method(self, fm):
        assert isinstance(fm.objective.state, str)

    def test_state_retracted_at_deactive_position(self, fm):
        assert fm.objective.state == "Retracted"  # stub starts at deactive

    def test_insert_updates_state(self, fm):
        fm.objective.insert()
        assert fm.objective.state == "Inserted"

    def test_retract_updates_state(self, fm):
        fm.objective.insert()
        fm.objective.retract()
        assert fm.objective.state == "Retracted"

    def test_state_other_between_positions(self, fm):
        focuser = fm.components["focus"]
        midpoint = (
            focuser.getMetadata()[stubs.MD_FAV_POS_ACTIVE]["z"]
            + focuser.getMetadata()[stubs.MD_FAV_POS_DEACTIVE]["z"]
        ) / 2
        fm.objective.move_absolute(midpoint)
        assert fm.objective.state == "Other"

    def test_state_error_when_unreadable(self, fm):
        fm.components["focus"].position = None
        assert fm.objective.state == "Error"

    def test_workflow_precondition_comparison(self, fm):
        # the exact comparison used by workflow tasks / movement guards
        fm.objective.insert()
        assert (fm.objective.state == "Inserted") is True


class TestCameraGeometry:
    """F3: pixel_size/resolution read live, no binning double-count."""

    def test_pixel_size_reads_live_metadata(self, fm):
        assert fm.camera.pixel_size == pytest.approx((1e-7, 1e-7))

    def test_no_double_count_after_binning_change(self, fm):
        fm.set_binning(2)
        # stub backend couples binning -> resolution + MD_PIXEL_SIZE like odemis
        assert fm.camera.pixel_size == pytest.approx((2e-7, 2e-7))
        assert fm.camera.resolution == (1024, 1024)

    def test_init_at_non_unit_binning(self, odemis_env):
        # a leftover binning from a previous session must not corrupt geometry
        components = stubs.default_components()
        components["ccd"].binning.value = (2, 2)  # couples res + pixel size
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        assert microscope.camera.pixel_size == pytest.approx((2e-7, 2e-7))
        assert microscope.camera.resolution == (1024, 1024)

    def test_field_of_view_invariant_under_binning(self, fm):
        fov_before = fm.camera.field_of_view
        fm.set_binning(4)
        assert fm.camera.field_of_view == pytest.approx(fov_before)


class TestLightSourcePower:
    """F4: power is a fraction of maximum, mapped to watts on the stream."""

    def test_set_power_fraction_maps_to_watts(self, fm):
        fm.set_power(0.5)  # stream range is (0, 0.4) W
        assert fm.light_source._stream.power.value == pytest.approx(0.2)

    def test_get_power_returns_fraction(self, fm):
        fm.light_source._stream.power.value = 0.1
        assert fm.light_source.power == pytest.approx(0.25)

    def test_power_round_trip(self, fm):
        fm.set_power(0.25)
        assert fm.light_source.power == pytest.approx(0.25)

    def test_power_clips_out_of_range_fraction(self, fm):
        fm.set_power(1.5)
        assert fm.light_source.power == pytest.approx(1.0)
        assert fm.light_source._stream.power.value == pytest.approx(0.4)

    def test_power_rejects_non_numeric(self, fm):
        with pytest.raises(TypeError):
            fm.set_power("high")


class TestChannelAndMetadata:
    """set_channel end-to-end + metadata records driver-normalised values."""

    def test_set_channel_configures_stream(self, fm):
        channel = ChannelSettings(
            name="ch-635",
            excitation_wavelength=635,
            emission_wavelength=680,
            power=0.25,
            exposure_time=0.05,
            color="red",
            gain=0.0,
        )
        fm.set_channel(channel)
        stream = fm.filter_set._stream
        assert stream.excitation.value[2] == pytest.approx(635e-9)
        assert stream.emission.value[0] == pytest.approx(680e-9)
        assert stream.power.value == pytest.approx(0.1)  # 0.25 * 0.4 W
        assert fm.camera.exposure_time == pytest.approx(0.05)
        assert fm.channel_name == "ch-635"

    def test_set_channel_reflection_mode(self, fm):
        channel = ChannelSettings(
            name="reflection",
            excitation_wavelength=550,
            emission_wavelength=None,
            power=0.1,
            exposure_time=0.01,
        )
        fm.set_channel(channel)
        assert fm.filter_set._stream.emission.value == stubs.BAND_PASS_THROUGH

    def test_metadata_records_power_fraction(self, fm):
        fm.set_power(0.25)
        metadata = fm.get_metadata()
        assert metadata.channels[0].power == pytest.approx(0.25)

    def test_metadata_geometry_tracks_binning(self, fm):
        fm.set_binning(2)
        metadata = fm.get_metadata()
        assert metadata.pixel_size_x == pytest.approx(2e-7)
        assert metadata.resolution == (1024, 1024)

    def test_acquire_image_returns_fluorescence_image(self, fm):
        image = fm.acquire_image()
        assert isinstance(image, FluorescenceImage)
        assert image.data.shape == fm.camera.resolution[::-1]
        assert image.metadata is not None


class TestObjectiveLimitsAndClipping:
    """F5/F6: limits from the focuser axis range; move_absolute clips/validates."""

    def test_limits_from_focuser_axis_range(self, fm):
        assert fm.objective.limits == pytest.approx((-2.0e-3, 10.0e-3))

    def test_default_user_limit_is_hardware_max(self, fm):
        assert fm.objective.limit_position == pytest.approx(10.0e-3)

    def test_move_absolute_clips_to_user_limit(self, fm):
        fm.objective.limit_position = 5.0e-3
        fm.objective.move_absolute(8.0e-3)
        assert fm.objective.position == pytest.approx(5.0e-3)

    def test_move_absolute_within_limit_is_unclipped(self, fm):
        fm.objective.limit_position = 5.0e-3
        fm.objective.move_absolute(3.0e-3)
        assert fm.objective.position == pytest.approx(3.0e-3)

    def test_move_absolute_raises_outside_axis_range(self, fm):
        with pytest.raises(ValueError):
            fm.objective.move_absolute(-5.0e-3)  # below focuser range minimum

    def test_insert_uses_calibrated_position_despite_user_limit(self, fm):
        # the favourite active position is calibrated, so insert() is not
        # subject to the user focusing limit
        fm.objective.limit_position = 5.0e-3
        fm.objective.insert()
        assert fm.objective.position == pytest.approx(8.0e-3)


class TestInitFallbacks:
    """F7: missing metadata keys must not disable the FM subsystem."""

    def test_missing_baseline_defaults_to_zero(self, odemis_env):
        components = stubs.default_components()
        del components["ccd"]._metadata[stubs.MD_BASELINE]
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        assert microscope.camera.offset == 0

    def test_missing_pixel_size_falls_back_to_sensor(self, odemis_env):
        components = stubs.default_components()
        del components["ccd"]._metadata[stubs.MD_PIXEL_SIZE]
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        assert microscope.camera.pixel_size == pytest.approx((6.5e-6, 6.5e-6))

    def test_missing_fav_active_uses_current_position(self, odemis_env):
        components = stubs.default_components()
        del components["focus"]._metadata[stubs.MD_FAV_POS_ACTIVE]
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        # falls back to the current (deactive) position instead of crashing
        assert microscope.objective.focus_position == pytest.approx(-1.0e-3)
        assert microscope.objective.state == "Other"

    def test_insert_without_fav_active_warns_and_stays(self, odemis_env):
        components = stubs.default_components()
        del components["focus"]._metadata[stubs.MD_FAV_POS_ACTIVE]
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        position_before = microscope.objective.position
        microscope.objective.insert()  # must not raise
        assert microscope.objective.position == pytest.approx(position_before)


class TestAcquireImageErrors:
    """F8: acquisition failures raise instead of returning None."""

    def test_acquisition_error_raises(self, fm, odemis_env, monkeypatch):
        monkeypatch.setattr(
            odemis_env.module,
            "acquire",
            lambda streams: stubs.FakeFuture(([], Exception("camera timeout"))),
        )
        with pytest.raises(RuntimeError, match="camera timeout"):
            fm.camera.acquire_image()

    def test_empty_acquisition_raises(self, fm, odemis_env, monkeypatch):
        monkeypatch.setattr(
            odemis_env.module,
            "acquire",
            lambda streams: stubs.FakeFuture(([], None)),
        )
        with pytest.raises(RuntimeError, match="no data"):
            fm.camera.acquire_image()


class TestEmissionFluorescenceMapping:
    """F9: TFS-style 'Fluorescence' maps to the band matching the excitation."""

    @pytest.mark.parametrize(
        "excitation_nm,expected_emission_nm",
        [(365, 420), (450, 500), (550, 590), (635, 680)],
    )
    def test_fluorescence_follows_excitation(
        self, fm, excitation_nm, expected_emission_nm
    ):
        fm.filter_set.excitation_wavelength = excitation_nm
        fm.filter_set.emission_wavelength = "Fluorescence"
        assert fm.filter_set.emission_wavelength == pytest.approx(
            expected_emission_nm
        )

    def test_set_channel_with_tfs_style_settings(self, fm):
        channel = ChannelSettings(
            name="tfs-style",
            excitation_wavelength=635,
            emission_wavelength="Fluorescence",
            power=0.1,
            exposure_time=0.01,
        )
        fm.set_channel(channel)
        assert fm.filter_set.emission_wavelength == pytest.approx(680)

    def test_fluorescence_skips_write_when_unchanged(self, fm):
        fm.filter_set.excitation_wavelength = 550
        fm.filter_set.emission_wavelength = "Fluorescence"
        emission_va = fm.filter_set._stream.emission
        sets_before = emission_va.set_count
        fm.filter_set.emission_wavelength = "Fluorescence"
        assert emission_va.set_count == sets_before


class TestGainWiring:
    """F11: gain maps to the camera gain VA when present; honest otherwise."""

    def test_gain_none_without_hardware_va(self, fm):
        assert fm.camera.gain is None

    def test_set_gain_without_va_is_warn_once_noop(self, fm, caplog):
        fm.set_gain(0.5)
        fm.set_gain(0.7)
        warnings = [r for r in caplog.records if "no gain control" in r.message]
        assert len(warnings) == 1
        assert fm.camera.gain is None  # never echoes back a software value

    def test_metadata_gain_none_without_va(self, fm):
        metadata = fm.get_metadata()
        assert metadata.channels[0].gain is None

    def test_gain_wired_when_va_present(self, odemis_env):
        components = stubs.default_components()
        components["ccd"].gain = stubs.FakeVA(1.0, range=(1.0, 30.0))
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        microscope.set_gain(2.5)
        assert components["ccd"].gain.value == pytest.approx(2.5)
        assert microscope.camera.gain == pytest.approx(2.5)
        assert microscope.get_metadata().channels[0].gain == pytest.approx(2.5)


class TestCameraLimitsFromHardware:
    """F12/F13: binning and exposure limits come from the camera VAs."""

    def test_available_binnings_from_range(self, fm):
        assert fm.camera.available_binnings == (1, 2, 4, 8, 16)

    def test_binning_setter_accepts_hardware_supported_value(self, fm):
        fm.set_binning(16)
        assert fm.camera.binning == 16

    def test_binning_setter_rejects_unsupported_value(self, fm):
        with pytest.raises(ValueError):
            fm.set_binning(3)

    def test_available_binnings_from_choices(self, odemis_env):
        components = stubs.default_components()
        components["ccd"].binning.choices = {(1, 1), (2, 2), (4, 4)}
        stubs.use_components(components)
        microscope = odemis_env.module.OdemisFluorescenceMicroscope(parent=None)
        assert microscope.camera.available_binnings == (1, 2, 4)

    def test_exposure_time_limits_from_va(self, fm):
        assert fm.camera.exposure_time_limits == (1e-3, 10.0)

    def test_power_limits_is_a_fraction_property(self, fm):
        assert fm.light_source.power_limits == (0.0, 1.0)


class TestFrameMetadata:
    """F20: per-frame DataArray metadata overrides the state snapshot."""

    def test_single_shot_uses_frame_metadata(self, fm):
        image = fm.acquire_image()
        # fake_acquire stamps a fixed acquisition date + the VA exposure time
        expected_date = datetime.fromtimestamp(1_780_000_000.0).isoformat()
        assert image.metadata.acquisition_date == expected_date
        assert image.metadata.channels[0].exposure_time == pytest.approx(
            fm.components["ccd"].exposureTime.value
        )

    def test_live_frame_metadata_overrides_state(self, fm):
        received = []
        fm.acquisition_signal.connect(received.append)
        try:
            fm.set_rate_limit(0)
            fm.start_acquisition()
            stream = fm.camera._stream
            dataflow = fm.components["ccd"].data
            assert wait_until(lambda: stream.is_active.value)
            assert wait_until(lambda: dataflow.listeners)
            frame = stubs.FakeDataArray(
                np.zeros((16, 16), dtype=np.uint16),
                metadata={stubs.MD_PIXEL_SIZE: (9e-8, 9e-8)},
            )
            dataflow.push(frame)
            assert wait_until(lambda: len(received) >= 1)
            # the frame's own pixel size wins over the current camera state
            assert received[0].metadata.pixel_size_x == pytest.approx(9e-8)
        finally:
            fm.stop_acquisition()
            fm.acquisition_signal.disconnect(received.append)

    def test_plain_ndarray_frames_fall_back_to_state(self, fm):
        image = fm._construct_image(np.zeros((8, 8), dtype=np.uint16))
        assert image.metadata.pixel_size_x == pytest.approx(1e-7)


class TestFastAcquisition:
    """F10: live acquisition activates the stream once and subscribes to
    the dataflow; the stream is deactivated (light off) in all stop paths."""

    def _start_live(self, fm):
        """Start live acquisition and wait until the stream owns the hardware."""
        fm.set_rate_limit(0)  # emit every frame in tests
        fm.start_acquisition()
        stream = fm.camera._stream
        dataflow = fm.components["ccd"].data
        assert wait_until(lambda: stream.is_active.value), "stream never activated"
        assert wait_until(lambda: dataflow.listeners), "listener never subscribed"
        return stream, dataflow

    def test_frames_are_pushed_and_emitted(self, fm):
        received = []
        fm.acquisition_signal.connect(received.append)
        try:
            stream, dataflow = self._start_live(fm)
            frame = np.full((2048, 2048), 42, dtype=np.uint16)
            dataflow.push(frame)
            assert wait_until(lambda: len(received) >= 1), "no frame emitted"
            assert received[0].data[0, 0] == 42
        finally:
            fm.stop_acquisition()
            fm.acquisition_signal.disconnect(received.append)

    def test_stop_deactivates_stream_and_unsubscribes(self, fm):
        try:
            stream, dataflow = self._start_live(fm)
        finally:
            fm.stop_acquisition()
        assert wait_until(lambda: not fm.is_acquiring), "worker did not stop"
        assert stream.is_active.value is False
        assert dataflow.listeners == []

    def test_frame_handler_error_does_not_kill_live_view(self, fm, monkeypatch):
        try:
            stream, dataflow = self._start_live(fm)
            monkeypatch.setattr(
                fm, "_construct_image", lambda data: (_ for _ in ()).throw(RuntimeError("boom"))
            )
            dataflow.push(np.zeros((8, 8), dtype=np.uint16))  # handled, not raised
            assert stream.is_active.value is True
            assert dataflow.listeners  # still subscribed
        finally:
            fm.stop_acquisition()
        assert stream.is_active.value is False

    def test_light_off_even_if_unsubscribe_fails(self, fm, monkeypatch):
        try:
            stream, dataflow = self._start_live(fm)
            monkeypatch.setattr(
                dataflow,
                "unsubscribe",
                lambda listener: (_ for _ in ()).throw(RuntimeError("ipc lost")),
            )
        finally:
            fm.stop_acquisition()
        assert wait_until(lambda: not fm.is_acquiring)
        assert stream.is_active.value is False  # deactivation must still happen

    def test_acquire_image_during_live_uses_dataflow(self, fm, odemis_env, monkeypatch):
        # snapshot during live must not run a full acqmng acquisition
        def _fail(streams, settings_obs=None):
            raise AssertionError("acqmng.acquire must not be called during live view")

        try:
            self._start_live(fm)
            monkeypatch.setattr(odemis_env.module, "acquire", _fail)
            image = fm.acquire_image()
            assert isinstance(image, FluorescenceImage)
            assert image.data.shape == fm.camera.resolution[::-1]
        finally:
            fm.stop_acquisition()

    def test_acquire_image_when_idle_uses_acqmng(self, fm, monkeypatch):
        # the dataflow snapshot path must only be used while the stream is active
        dataflow = fm.components["ccd"].data
        monkeypatch.setattr(
            dataflow,
            "get",
            lambda asap=True: (_ for _ in ()).throw(AssertionError("dataflow.get in idle mode")),
        )
        image = fm.acquire_image()
        assert isinstance(image, FluorescenceImage)

"""The JEOL driver over the fake JEOL library (``tests/fixtures/jeol_sdk.py``).

No JEOL library or instrument: the fake knows only the calls the old-API driver made,
so these pin the driver's conversions (units, signs, pixel placement, doses) and the
calls it makes, not JEOL's behaviour, which the hardware session checks.
"""

import math

import pytest

from fibsem.drivers import registry
from fibsem.drivers.jeol import devices as jd
from fibsem.drivers.jeol import services as js
from fibsem.drivers.jeol.connection import load_sdk
from fibsem.drivers.jeol.settings import JeolMillingSettings
from fibsem.imaging.spot import SpotBurnSettings, SpotBurnStatus
from fibsem.structures.settings import (
    BeamType,
    CrossSectionPattern,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from tests.fixtures import jeol_sdk

E, I = BeamType.ELECTRON, BeamType.ION


@pytest.fixture
def connected(monkeypatch):
    return jeol_sdk.connect(monkeypatch)


# -- the driver record and connect ----------------------------------------------------


def test_registry_names_jeol():
    entry = registry.get_driver("JEOL")
    assert entry.microscope_class == "fibsem.drivers.jeol.microscope:JeolMicroscope"
    assert entry.port == 22218
    assert registry.get_driver("jeol") is entry


def test_load_sdk_needs_a_path(monkeypatch):
    monkeypatch.delenv("FIBSEM_JEOL_API_PATH", raising=False)
    with pytest.raises(ImportError, match="FIBSEM_JEOL_API_PATH"):
        load_sdk(None)


def test_connect_builds_the_devices_and_services(monkeypatch):
    fake = jeol_sdk.FakeJeol()
    microscope, _ = jeol_sdk.connect(monkeypatch, fake=fake)
    assert set(microscope.beams) == {E, I}
    assert isinstance(microscope.beams[E], jd.JeolElectronBeam)
    assert isinstance(microscope.beams[I], jd.JeolIonBeam)
    assert isinstance(microscope.stage, jd.JeolStage)
    assert isinstance(microscope.milling, js.JeolMilling)
    assert isinstance(microscope.spot_burn, js.JeolSpotBurn)
    assert microscope.manufacturer == "JEOL"
    assert microscope.system.info.manufacturer == "JEOL"
    # both clients on the driver's ports; SEM on the left, FIB on the right
    assert fake.fib.address == ("localhost", 22218)
    assert fake.sem.address == ("localhost", 22300)
    assert (
        fake.fib.view_signals[jeol_sdk.EViewType.LEFT] is jeol_sdk.EViewSignal.SEM_LED
    )
    assert fake.fib.view_signals[jeol_sdk.EViewType.RIGHT] is jeol_sdk.EViewSignal.FIB
    assert microscope.get_detector_type(E) == "SEM_LED"
    assert microscope.get_detector_type(I) == "FIB"


def test_a_disabled_column_is_never_built(monkeypatch):
    microscope, fake = jeol_sdk.connect(
        monkeypatch, jeol_sdk.jeol_system(electron=False)
    )
    assert set(microscope.beams) == {I}
    assert microscope.milling is not None


def test_disconnect_closes_both_clients_once(connected):
    microscope, fake = connected
    microscope.disconnect()
    microscope.disconnect()
    assert fake.calls() == ["close", "close"]


def test_a_failed_call_warns_and_carries_on(connected, caplog):
    # as the old driver did: a failed call is logged, never raised
    microscope, fake = connected
    fake.fib.setFibBeamMode = lambda mode: 1
    with caplog.at_level("WARNING"):
        microscope.beams[I].current.set_value(1e-9)
    assert "setFibBeamMode" in caplog.text


# -- beams ----------------------------------------------------------------------------


@pytest.mark.parametrize("beam_type", [E, I])
def test_hfw_is_128_mm_over_the_magnification(connected, beam_type):
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    beam.hfw.set_value(64e-6)
    client = fake.fib if beam_type is I else fake.sem
    assert client.magnification == 2000
    assert beam.hfw.get_value() == pytest.approx(64e-6)
    # clipped to the column's limits
    beam.hfw.write_through(1.0)
    assert client.magnification == round(128e-3 / jd.HFW_LIMITS[beam_type].max)


def test_the_nearest_magnification_is_not_an_error(connected):
    microscope, fake = connected
    fake.fib.setFibMagnification = lambda magnification: jd.MAGNIFICATION_NEAREST
    microscope.beams[I].hfw.set_value(100e-6)


@pytest.mark.parametrize("beam_type", [E, I])
def test_current_is_the_nearest_beam_mode(connected, beam_type):
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    assert beam.current.choices == jd.BEAM_CURRENTS[beam_type]
    beam.current.write_through(2.5e-9)  # between 1 nA and 3 nA (ion) / 1 nA and 4 nA
    expected = 3e-9 if beam_type is I else 1e-9 * 4
    mode = jd.BEAM_CURRENTS[beam_type].index(expected) + 1
    assert (fake.fib.beam_mode if beam_type is I else fake.sem.probe) == mode
    assert beam.current.get_value() == expected


def test_nearest_current_is_by_ratio():
    # 25 pA is nearer 30 pA than 10 pA; by ratio, 5 pA is nearer 3 pA than 10 pA
    assert jd.nearest_beam_current(25e-12, I)[1] == 30e-12
    assert jd.nearest_beam_current(5e-12, I)[1] == 3e-12


def test_sem_voltage_is_in_kilovolts(connected):
    microscope, fake = connected
    microscope.beams[E].voltage.set_value(10000)
    assert fake.sem.kv == 10
    assert microscope.beams[E].voltage.get_value() == 10000


@pytest.mark.parametrize("beam_type", [E, I])
def test_shift_is_in_fibsem_convention(connected, beam_type):
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    beam.shift.set_value(Point(1.86e-6, -3.72e-6))
    if beam_type is I:
        assert fake.fib.fine_shift == pytest.approx([-1.86e-3, 3.72e-3])  # mm, signed
    else:
        assert fake.sem.image_shift == [-100, 200]  # DAC steps, signed
    shift = beam.shift.get_value()
    assert (shift.x, shift.y) == pytest.approx((1.86e-6, -3.72e-6))


def test_sem_shift_is_clipped_to_the_dac(connected):
    microscope, fake = connected
    microscope.beams[E].shift.write_through(Point(-1e-3, 0))
    assert fake.sem.image_shift == [2047, 0]


def test_relative_beam_shift_adds(connected):
    microscope, fake = connected
    microscope.beam_shift(1e-6, 0, I)
    microscope.beam_shift(1e-6, 0, I)
    assert microscope.get_beam_shift(I).x == pytest.approx(2e-6)


@pytest.mark.parametrize("beam_type", [E, I])
def test_working_distance_calibration_round_trips(connected, beam_type):
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    beam.working_distance.set_value(10e-3 if beam_type is E else 19e-3)
    assert beam.working_distance.get_value() == pytest.approx(
        10e-3 if beam_type is E else 19e-3, abs=1e-5
    )
    if beam_type is I:
        assert fake.fib.ol_voltage == round(jd.FIB_WORKING_DISTANCE_VOLTAGE_REF)


def test_sem_scan_rotation_switches_on_and_off(connected):
    microscope, fake = connected
    beam = microscope.beams[E]
    beam.scan_rotation.set_value(math.pi)
    assert (fake.sem.scan_rotation, fake.sem.scan_rotation_switch) == (180, 1)
    beam.scan_rotation.set_value(0.0)
    assert (fake.sem.scan_rotation, fake.sem.scan_rotation_switch) == (0, 0)


def test_scan_rotation_is_rounded_into_a_turn(connected):
    microscope, fake = connected
    microscope.beams[I].scan_rotation.write_through(math.radians(-90.4))
    assert fake.fib.scan_rotation == 270


def test_fib_resolution_and_dwell_pick_the_scan_type(connected):
    microscope, fake = connected
    beam = microscope.beams[I]
    assert (1280, 960) in beam.resolution.choices
    beam.resolution.set_value((1280, 960))
    # the dwell (0.49 us at 640x480) kept as near as 1280x960 allows: scan type 4
    assert beam.resolution.get_value() == (1280, 960)
    assert fake.fib.scan_types[0] == 4
    # the dwell times listed are the scan types at 1280x960
    assert beam.dwell_time.choices == [
        0.41e-6,
        0.81e-6,
        8.1e-6,
        16.3e-6,
        32.6e-6,
        65.1e-6,
        130e-6,
    ]
    beam.dwell_time.write_through(16e-6)
    assert fake.fib.scan_types[2] == 10  # 16.3 us, scan memory 2
    assert microscope._connection.fib_scan_mode == 2
    assert beam.dwell_time.get_value() == pytest.approx(16.3e-6)
    # a frame the FIB doesn't have is snapped
    beam.resolution.write_through((1536, 1024))
    assert beam.resolution.get_value() == (1280, 960)


def test_sem_resolution_is_the_photo_size(connected):
    microscope, fake = connected
    beam = microscope.beams[E]
    beam.resolution.set_value((2560, 1920))
    assert fake.sem.photo_size == 3
    beam.resolution.write_through((1536, 1024))
    assert fake.sem.photo_size == 2 and beam.resolution.get_value() == (1280, 960)


def test_sem_dwell_time_is_the_scan_speed(connected):
    microscope, fake = connected
    microscope.beams[E].dwell_time.write_through(1.3e-6)
    assert fake.sem.rate_no == 6
    assert microscope.beams[E].dwell_time.get_value() == pytest.approx(1.333e-6)


@pytest.mark.parametrize("beam_type", [E, I])
def test_contrast_and_brightness_round_trip(connected, beam_type):
    microscope, _ = connected
    beam = microscope.beams[beam_type]
    beam.detector_contrast.set_value(0.25)
    beam.detector_brightness.set_value(0.75)
    assert beam.detector_contrast.get_value() == pytest.approx(0.25, abs=1e-3)
    assert beam.detector_brightness.get_value() == pytest.approx(0.75, abs=1e-3)


def test_the_ion_detector_is_written_to_the_fib_view(connected):
    # the old driver wrote it to the left (SEM) view
    microscope, fake = connected
    microscope.set_detector_type("FIB", I)
    _, _, args, _ = [c for c in fake.log if c[1] == "setViewSignal"][-1]
    assert args[0] == jeol_sdk.EViewType.RIGHT


@pytest.mark.parametrize("beam_type", [E, I])
def test_the_view_layout_is_set_before_each_detector_read_and_write(
    connected, beam_type
):
    # as the old driver's force_gui_configuration: dual view, SEM_LED left, FIB right
    microscope, fake = connected
    beam = microscope.beams[beam_type]
    for touch in (
        beam.detector_type.get_value,
        lambda: beam.detector_type.write_through(beam.detector_type.choices[0]),
    ):
        fake.log.clear()
        touch()
        assert fake.calls()[:3] == ["setViewMode", "setViewSignal", "setViewSignal"]


def test_the_layout_puts_the_sem_back_on_sem_led(connected):
    # what the old layout does: another SEM detector doesn't survive a read
    microscope, _ = connected
    microscope.set_detector_type("SEM_UED", E)
    assert microscope.get_detector_type(E) == "SEM_LED"


def test_what_the_ion_column_lacks_is_absent(connected):
    microscope, _ = connected
    ion = microscope.beams[I]
    assert "blanked" not in ion.parameters and "stigmation" not in ion.parameters
    assert not ion.commands["auto_focus"].available
    assert microscope.beams[E].commands["auto_focus"].available


# -- imaging --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "beam_type, requested, frame",
    [
        (I, (640, 480), (640, 480)),
        (I, (1536, 1024), (1280, 960)),
        (E, (1536, 1024), (1280, 960)),
    ],
)
def test_acquire(connected, beam_type, requested, frame):
    microscope, fake = connected
    image = microscope.acquire_image(
        ImageSettings(beam_type=beam_type, hfw=128e-6, resolution=requested)
    )
    assert image.data.shape == (frame[1], frame[0])
    settings = image.metadata.image_settings
    assert settings.resolution == frame
    assert settings.hfw == pytest.approx(128e-6)
    assert image.metadata.pixel_size.x == pytest.approx(128e-6 / frame[0])
    # the dwell time the column read back, not a fixed 1 us
    assert settings.dwell_time == microscope.beams[beam_type].dwell_time.get_value()


def test_images_are_stamped_through_the_session(connected):
    microscope, _ = connected
    beam = microscope.beams[I]
    stamped = []
    beam.session.stamp = stamped.append
    image = microscope.acquire_image(ImageSettings(beam_type=I, hfw=128e-6))
    assert stamped == [image]


def test_sem_live_reads_the_buffer(connected):
    microscope, _ = connected
    image = microscope.beams[E]._live_frame()
    assert image.data.shape == (960, 1280)


def test_autocontrast_selects_the_view(connected):
    microscope, fake = connected
    microscope.beams[I].autocontrast()
    assert fake.calls() == [
        "setSelectedView",
        "setFibFreezeMode",
        "execACB",
        "setFibFreezeMode",
    ]


# -- stage ----------------------------------------------------------------------------


def test_stage_reads_signed_axes(connected):
    microscope, fake = connected
    fake.fib.stage = {"x": 1e-3, "y": 2e-3, "fz": 0.5e-3, "rz": 0.1, "rx": 0.2}
    position = microscope.get_stage_position()
    assert (
        position.x,
        position.y,
        position.z,
        position.r,
        position.t,
    ) == pytest.approx((-1e-3, -2e-3, -0.5e-3, 0.1, 0.2))


def test_stage_moves_in_millimetres_and_degrees(connected):
    microscope, fake = connected
    microscope.move_stage_absolute(
        FibsemStagePosition(x=1e-3, y=None, z=0.0, r=0.0, t=math.radians(10))
    )
    _, _, args, kwargs = [
        c for c in fake.log if c[1] == "setMultiAxisStagePositionAbs"
    ][0]
    assert kwargs == pytest.approx(
        {"x": -1.0, "y": 0.0, "t": 10.0, "r": 0.0, "fz": 0.0}
    )
    assert microscope.get_stage_position().x == pytest.approx(1e-3)


def test_stage_moves_relative(connected):
    microscope, fake = connected
    microscope.move_stage_relative(FibsemStagePosition(x=1e-3, y=0, z=0, r=0, t=0))
    microscope.move_stage_relative(FibsemStagePosition(x=1e-3, y=0, z=0, r=0, t=0))
    assert microscope.get_stage_position().x == pytest.approx(2e-3)


def test_fib_pose_is_a_tilt_at_the_sem_rotation(connected):
    microscope, fake = connected
    sem, fib = microscope.get_orientation("SEM"), microscope.get_orientation("FIB")
    assert fib.r == pytest.approx(sem.r)
    assert fib.t == pytest.approx(math.radians(53))
    fake.fib.stage = {"x": 0, "y": 0, "fz": 0, "rz": 0.0, "rx": math.radians(53)}
    assert microscope.get_stage_orientation() == "FIB"


def test_stage_limits(connected):
    microscope, _ = connected
    limits = microscope.stage.position.limits
    assert limits["t"].max == pytest.approx(math.radians(55))
    assert limits["x"].min == -55e-3


# -- milling --------------------------------------------------------------------------


def _setup(microscope, current=1e-9, milling_rate=1e9, hfw=128e-6):
    microscope.milling.setup(
        FibsemMillingSettings(
            milling_channel=I,
            milling_current=current,
            hfw=hfw,
            driver={"JEOL": JeolMillingSettings(milling_rate=milling_rate)},
        )
    )


def _draw_rectangle(microscope, depth):
    microscope.milling.draw(
        [
            FibsemRectangleSettings(
                width=10e-6, height=10e-6, depth=depth, centre_x=0, centre_y=0
            )
        ]
    )


def test_dose_from_milling_rate():
    # 1 um deep at 1 nC/um3 (1e9 C/m3) is 1 nC/um2
    assert js.dose_for_depth(1e-6, 1e9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        js.dose_for_depth(1e-6, 0)


def test_the_milling_rate_is_read_from_the_recipes_jeol_block(connected):
    microscope, fake = connected
    _setup(microscope, milling_rate=2e9)
    _draw_rectangle(microscope, depth=1e-6)
    assert fake.fib.items[-1].fields["dose"] == pytest.approx(2.0)


def test_a_recipe_without_a_jeol_block_mills_at_the_default_rate(connected):
    microscope, fake = connected
    microscope.milling.setup(
        FibsemMillingSettings(milling_channel=I, milling_current=1e-9, hfw=128e-6)
    )
    _draw_rectangle(microscope, depth=1e-6)
    default = JeolMillingSettings().milling_rate
    assert fake.fib.items[-1].fields["dose"] == pytest.approx(default * 1e-9)


def test_the_jeol_block_round_trips_through_a_saved_recipe():
    recipe = FibsemMillingSettings(
        driver={"JEOL": JeolMillingSettings(milling_rate=3e9)}
    )
    loaded = FibsemMillingSettings.from_dict(recipe.to_dict())
    assert loaded.driver["JEOL"].milling_rate == pytest.approx(3e9)
    assert loaded.setting("driver.JEOL.milling_rate") == pytest.approx(3e9)


def test_rectangle_is_an_area_item(connected):
    microscope, fake = connected
    microscope.beams[I].hfw.set_value(128e-6)  # 640x480 frame: vfw 96 um
    _setup(microscope)
    microscope.milling.draw(
        [
            FibsemRectangleSettings(
                width=32e-6,
                height=24e-6,
                depth=2e-6,
                centre_x=16e-6,
                centre_y=0,
                scan_direction="TopToBottom",
            )
        ]
    )
    (item,) = fake.fib.items
    assert item.kind == "area"
    fields = item.fields
    # top-left corner (0, 12 um) -> (640, 360) px of 1280x960
    assert (fields["x"], fields["y"]) == (640, 360)
    assert fields["width"] == pytest.approx(32) and fields["height"] == pytest.approx(
        24
    )
    assert fields["beam"] == jd.BEAM_CURRENTS[I].index(1e-9) + 1
    assert fields["dose"] == pytest.approx(2.0)
    assert fields["finish_face"] is jeol_sdk.EProcessEndside.Bottom
    assert fields["mode"] is jeol_sdk.EProcessMode.MILL


def test_line_is_a_line_item(connected):
    microscope, fake = connected
    microscope.beams[I].hfw.set_value(128e-6)
    _setup(microscope)
    microscope.milling.draw(
        [FibsemLineSettings(start_x=-64e-6, end_x=0, start_y=0, end_y=0, depth=1e-6)]
    )
    (item,) = fake.fib.items
    assert item.kind == "line"
    assert item.fields["line_type"] is jeol_sdk.EProcessLineType.HORIZONTAL
    assert (item.fields["x"], item.fields["y"]) == (0, 1920)
    assert item.fields["width"] == pytest.approx(64)


@pytest.mark.parametrize(
    "pattern, error",
    [
        (
            FibsemRectangleSettings(
                width=1e-6,
                height=1e-6,
                depth=1e-6,
                centre_x=0,
                centre_y=0,
                rotation=0.1,
            ),
            ValueError,
        ),
        (
            FibsemCircleSettings(radius=1e-6, depth=1e-6, centre_x=0, centre_y=0),
            NotImplementedError,
        ),
    ],
)
def test_what_jeol_cannot_mill_raises(connected, pattern, error):
    microscope, fake = connected
    _setup(microscope)
    with pytest.raises(error):
        microscope.milling.draw([pattern])
    assert fake.fib.items == []


def test_a_diagonal_line_is_left_out(connected, caplog):
    # as the old driver did: warned about, not drawn
    microscope, fake = connected
    _setup(microscope)
    with caplog.at_level("WARNING"):
        microscope.milling.draw(
            [
                FibsemLineSettings(
                    start_x=0, end_x=1e-6, start_y=0, end_y=1e-6, depth=1e-6
                )
            ]
        )
    assert fake.fib.items == []
    assert "diagonal" in caplog.text


@pytest.mark.parametrize(
    "direction, finish",
    [
        ("DynamicTopToBottom", "Bottom"),
        ("DynamicLeftToRight", "Right"),
        ("InnerToOuter", "Bottom"),  # the default, TopToBottom
    ],
)
def test_a_scan_direction_jeol_lacks_is_converted(connected, direction, finish):
    microscope, fake = connected
    _setup(microscope)
    microscope.milling.draw(
        [
            FibsemRectangleSettings(
                width=1e-6,
                height=1e-6,
                depth=1e-6,
                centre_x=0,
                centre_y=0,
                scan_direction=direction,
            )
        ]
    )
    (item,) = fake.fib.items
    assert item.fields["finish_face"] is getattr(jeol_sdk.EProcessEndside, finish)


def test_cross_sections(connected):
    microscope, fake = connected
    _setup(microscope)
    for cross_section in (
        CrossSectionPattern.RegularCrossSection,
        CrossSectionPattern.CleaningCrossSection,
    ):
        microscope.milling.draw(
            [
                FibsemRectangleSettings(
                    width=1e-6,
                    height=1e-6,
                    depth=1e-6,
                    centre_x=0,
                    centre_y=0,
                    cross_section=cross_section,
                )
            ]
        )
    assert [item.fields["mode"] for item in fake.fib.items] == [
        jeol_sdk.EProcessMode.RAPID
    ] * 2


def test_an_item_jeol_cannot_run_clears_the_group(connected, caplog):
    # as the old driver did: warned about, and the process group deleted
    microscope, fake = connected
    _setup(microscope)
    fake.fib.executable = False
    with caplog.at_level("WARNING"):
        microscope.milling.draw(
            [
                FibsemRectangleSettings(
                    width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0
                )
            ]
        )
    assert "can't run" in caplog.text
    assert fake.calls()[-1] == "deleteProcessGroup"


def test_milling_runs_to_the_end(connected):
    microscope, fake = connected
    microscope.milling.poll_interval = 0
    _setup(microscope)
    rectangle = FibsemRectangleSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0
    )
    microscope.milling.draw([rectangle, rectangle, rectangle])
    assert microscope.milling.estimate_time()["seconds"] == 3.0
    microscope.milling.run()
    assert fake.fib.process_state == "Exit"
    assert fake.fib.items == []  # cleared at the end
    # milling never sets the beam's current: each item carries its beam mode
    assert "setFibBeamMode" not in fake.calls()


def test_pause_resume_stop(connected):
    microscope, fake = connected
    _setup(microscope)
    microscope.milling.draw(
        [
            FibsemRectangleSettings(
                width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0
            )
        ]
    )
    microscope.milling.start()
    microscope.milling.pause()
    assert microscope.milling.state.get_value().name == "PAUSED"
    microscope.milling.resume()
    assert fake.fib.process_state == "Start"
    microscope.milling.stop()
    assert microscope.milling.state.get_value().name == "IDLE"


def test_milling_offers_ion_only(connected):
    microscope, _ = connected
    assert microscope.milling.supported_settings(E) == {}
    supported = microscope.milling.supported_settings(I)
    assert set(supported) == {
        "milling_channel",
        "hfw",
        "milling_current",
        "driver.JEOL.milling_rate",
    }
    assert supported["milling_current"].choices == jd.BEAM_CURRENTS[I]


# -- spot burn ------------------------------------------------------------------------


def test_spot_burn(connected):
    microscope, fake = connected
    microscope.milling.poll_interval = 0
    added = []
    real_add = fake.fib.addProcessItem
    fake.fib.addProcessItem = lambda item: (added.append(item), real_add(item))[1]
    status = microscope.spot_burn.run(
        SpotBurnSettings(
            coordinates=[Point(0.5, 0.5), Point(0.25, 0.75), Point(1.5, 0.5)],
            exposure_time=5,
            milling_current=100e-12,
        )
    )
    assert status is SpotBurnStatus.FINISHED
    assert [item.kind for item in added] == [
        "spot",
        "spot",
    ]  # the point outside dropped
    assert [(i.fields["x"], i.fields["y"]) for i in added] == [(640, 480), (320, 720)]
    # 5 s at 100 pA is the measured 333 nC/um2
    assert added[0].fields["dose"] == pytest.approx(333, rel=1e-3)
    assert added[0].fields["beam"] == 9


def test_spot_burn_refuses_no_exposure(connected):
    microscope, _ = connected
    with pytest.raises(ValueError):
        microscope.spot_burn.run(
            SpotBurnSettings(
                coordinates=[Point(0.5, 0.5)], exposure_time=0, milling_current=1e-10
            )
        )


def test_milling_places_patterns_in_the_recipe_field(connected):
    microscope, fake = connected
    _setup(microscope, hfw=256e-6)
    assert fake.fib.magnification == 500
    microscope.milling.restore()
    assert fake.fib.magnification == 1000  # the imaging field back


def test_acquire_leaves_the_dwell_time(connected):
    # setting the dwell time sets the scan mode, which can undo the resolution
    microscope, _ = connected
    microscope.acquire_image(
        ImageSettings(beam_type=I, hfw=128e-6, resolution=(1280, 960), dwell_time=16e-6)
    )
    beam = microscope.beams[I]
    assert beam.resolution.get_value() == (1280, 960)
    assert beam.dwell_time.get_value() != pytest.approx(16.3e-6)


def test_an_absolute_move_goes_straight_there(connected):
    # no tilt-flat and rotate first, as the old driver had it
    microscope, fake = connected
    microscope.safe_absolute_stage_movement(
        FibsemStagePosition(x=0, y=0, z=0, r=math.radians(180), t=math.radians(30))
    )
    assert fake.calls().count("setMultiAxisStagePositionAbs") == 1


def test_jeol_draws_rectangles_and_lines():
    assert set(js.JeolMilling.supported_pattern_types()) == {
        FibsemRectangleSettings,
        FibsemLineSettings,
    }


@pytest.mark.parametrize(
    "beam_type, field, frame",
    [
        (E, "setMagnification", "setPhotoImageSize"),
        (I, "setFibMagnification", "setFibScanType"),
    ],
)
def test_acquire_selects_the_view_first(connected, beam_type, field, frame):
    # the old driver's order: view, field of view, (state), resolution, frame
    microscope, fake = connected
    microscope.acquire_image(
        ImageSettings(beam_type=beam_type, hfw=128e-6, resolution=(1280, 960))
    )
    calls = fake.calls()
    assert calls[0] == "setSelectedView"
    assert calls.index(field) < calls.index(frame)

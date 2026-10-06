"""OdemisThermoMicroscope against an odemis client that has only the real methods.

The client is a hand-written stand-in for ``odemis.driver.autoscript_client.SEM``:
every method on it exists on the real client with the same name and arguments, and
nothing else does, so a call to a method odemis does not have fails here as it would
on a METEOR. A ``MagicMock`` answers every call, which is how ``run_auto_focus`` and
``connection.vacuum`` went unnoticed (FIB-1091).

No odemis installation or hardware required: odemis is replaced by the stub modules
in tests/fm/_odemis_stubs.py, and the microscope is created without __init__.
"""

import os
import sys

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.milling import FibsemMillingStage
from fibsem.milling.patterning.patterns2 import PolygonPattern
from fibsem.milling.progress import MillingProgressStatus
from fibsem.milling.tasks import FibsemMillingTask, FibsemMillingTaskConfig
from fibsem.structures import (
    BeamType,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemExperimentRef,
    FibsemImage,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    FibsemUser,
    ImageSettings,
    MillingState,
)
from tests.fm import _odemis_stubs as stubs

ODEMIS_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "odemis-configuration.yaml")

FRAME_SHAPE = (1024, 1536)  # rows, columns


class FakeOdemisClient:
    """The subset of ``autoscript_client.SEM`` the microscope uses, by its real names.

    Return values follow the AutoScript adapter (xtadapter 1.16.0): ion currents as
    choices and electron currents as a range, `get_last_image` as a bare array.
    """

    def __init__(self):
        self.calls = []
        self.settings = {
            channel: {
                "resolution": (FRAME_SHAPE[1], FRAME_SHAPE[0]),
                "hfw": 150e-6,
                "dwell_time": 1e-6,
                "working_distance": 4e-3,
            }
            for channel in ("electron", "ion")
        }
        self.chamber_state = "Pumped"
        self.last_image = np.zeros(FRAME_SHAPE, dtype=np.uint8)

    def _record(self, name, *args):
        self.calls.append((name, *args))

    # imaging
    def acquire_image(self, channel, frame_settings=None):
        self._record("acquire_image", channel, frame_settings)
        width, height = self.settings[channel]["resolution"]
        return np.zeros((height, width), dtype=np.uint8), {}

    def get_last_image(self, channel, wait_for_frame=True):
        self._record("get_last_image", channel)
        return self.last_image

    def set_full_frame_scan_mode(self, channel):
        self._record("set_full_frame_scan_mode", channel)

    def set_reduced_area_scan_mode(self, channel, left, top, width, height):
        self._record("set_reduced_area_scan_mode", channel)

    def run_auto_contrast_brightness(self, channel, parameters={}):
        self._record("run_auto_contrast_brightness", channel)

    # beam
    def get_resolution(self, channel):
        return self.settings[channel]["resolution"]

    def set_resolution(self, resolution, channel):
        self._record("set_resolution", resolution, channel)
        self.settings[channel]["resolution"] = tuple(resolution)

    def get_field_of_view(self, channel):
        return self.settings[channel]["hfw"]

    def set_field_of_view(self, hfw, channel):
        self._record("set_field_of_view", hfw, channel)
        self.settings[channel]["hfw"] = hfw

    def get_dwell_time(self, channel):
        return self.settings[channel]["dwell_time"]

    def set_dwell_time(self, dwell_time, channel):
        self._record("set_dwell_time", dwell_time, channel)
        self.settings[channel]["dwell_time"] = dwell_time

    def get_working_distance(self, channel):
        return self.settings[channel]["working_distance"]

    def set_working_distance(self, working_distance, channel):
        self._record("set_working_distance", working_distance, channel)
        self.settings[channel]["working_distance"] = working_distance

    def get_beam_is_on(self, channel):
        return True

    def beam_is_blanked(self, channel):
        return False

    def get_beam_current(self, channel):
        return 1e-9

    def get_high_voltage(self, channel):
        return 30e3

    def get_scan_rotation(self, channel):
        return 0.0

    def get_beam_shift(self, channel):
        return (0.0, 0.0)

    def get_stigmator(self, channel):
        return (0.0, 0.0)

    def beam_current_info(self, channel):
        if channel == "ion":
            return {"unit": "A", "choices": [1e-12, 1e-11, 1e-10]}
        return {"unit": "A", "range": (1e-12, 1e-11)}

    def high_voltage_info(self, channel):
        return {"unit": "V", "range": (1000, 20000)}

    # detector
    def get_detector_type(self, channel):
        return "ETD"

    def get_detector_mode(self, channel):
        return "SecondaryElectrons"

    def get_brightness(self, channel):
        return 0.5

    def get_contrast(self, channel):
        return 0.5

    def detector_type_info(self, channel):
        return {"choices": {"ETD", "TLD"}}

    def detector_mode_info(self, channel):
        return {"choices": {"SecondaryElectrons", "BackscatterElectrons"}}

    # stage
    def is_homed(self):
        return True

    def is_linked(self):
        return False

    # chamber
    def get_chamber_state(self):
        return self.chamber_state

    def get_pressure(self):
        return 1e-4

    # beam control, as milling uses it
    def set_high_voltage(self, voltage, channel):
        self._record("set_high_voltage", voltage, channel)

    def set_beam_current(self, current, channel):
        self._record("set_beam_current", current, channel)

    def set_beam_shift(self, x, y, channel):
        self._record("set_beam_shift", x, y, channel)

    def set_active_view(self, view):
        self._record("set_active_view", view)

    def set_active_device(self, device):
        self._record("set_active_device", device)

    # patterning
    def set_patterning_mode(self, mode):
        self._record("set_patterning_mode", mode)

    def set_default_application_file(self, application_file):
        self._record("set_default_application_file", application_file)

    def set_default_patterning_beam_type(self, channel):
        self._record("set_default_patterning_beam_type", channel)

    def clear_patterns(self):
        self._record("clear_patterns")

    def get_patterning_state(self):
        self._record("get_patterning_state")
        return "Idle"

    def create_rectangle(self, parameters):
        self._record("create_rectangle", parameters)
        return {"id": 1, "time": 1.0}

    def create_circle(self, parameters):
        self._record("create_circle", parameters)
        return {"id": 2, "time": 1.0}

    def create_line(self, parameters):
        self._record("create_line", parameters)
        return {"id": 3, "time": 1.0}


class FakeStage:
    def __init__(self):
        self.position = stubs.FakeVA({"x": 0, "y": 0, "z": 0, "rz": 0, "rx": 0})


@pytest.fixture(scope="module")
def odemis_microscope_cls():
    """Import OdemisThermoMicroscope against stub odemis modules."""
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)

    stubs.install_odemis_stubs()
    from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope

    yield OdemisThermoMicroscope

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


@pytest.fixture
def microscope(odemis_microscope_cls):
    microscope = object.__new__(odemis_microscope_cls)  # skip __init__ (hardware)
    microscope.system = utils.load_microscope_configuration(ODEMIS_CONFIG_PATH).system
    microscope.connection = FakeOdemisClient()
    microscope._vendor_stage = FakeStage()
    microscope.stage_is_compustage = False
    microscope.fm = None
    microscope.user = FibsemUser()
    microscope.experiment = FibsemExperimentRef()
    microscope._last_imaging_settings = ImageSettings()
    microscope.milling_channel = BeamType.ION
    microscope._default_application_file = "Si"
    microscope._build_devices()
    microscope.connection.calls.clear()
    return microscope


def test_acquire_image_with_settings_stamps_the_shared_metadata(microscope):
    image_settings = ImageSettings(
        beam_type=BeamType.ELECTRON, resolution=[1536, 1024], hfw=100e-6
    )

    image = microscope.acquire_image(image_settings)

    assert isinstance(image, FibsemImage)
    assert image.data.shape == FRAME_SHAPE
    assert image.metadata.pixel_size.x == pytest.approx(100e-6 / 1536)
    # what _set_additional_metadata adds, as on every other backend
    assert image.metadata.system_info is not None
    assert image.metadata.hardware_geometry is not None
    assert image.metadata.experiment is not microscope.experiment


def test_acquire_image_with_beam_type_uses_the_current_settings(microscope):
    microscope.connection.settings["ion"]["hfw"] = 80e-6

    image = microscope.acquire_image(beam_type=BeamType.ION)

    assert image.metadata.image_settings.beam_type is BeamType.ION
    assert image.metadata.image_settings.hfw == pytest.approx(80e-6)
    assert image.metadata.image_settings.resolution == [1536, 1024]
    # nothing about the beam was changed to take it
    names = [call[0] for call in microscope.connection.calls]
    assert names == ["acquire_image"]
    assert microscope.connection.calls[0] == ("acquire_image", "ion", None)


def test_acquire_image_needs_settings_or_a_beam(microscope):
    with pytest.raises(ValueError):
        microscope.acquire_image()


@pytest.mark.parametrize("returns_tuple", [False, True])
def test_last_image(microscope, returns_tuple):
    frame = np.full((512, 768), 7, dtype=np.uint8)
    microscope.connection.last_image = (frame, {}) if returns_tuple else frame

    image = microscope.last_image(BeamType.ELECTRON)

    assert np.array_equal(image.data, frame)
    assert image.metadata.image_settings.resolution == [768, 512]


def test_auto_focus_runs_the_working_distance_sweep(microscope):
    microscope.auto_focus(BeamType.ELECTRON)

    names = [call[0] for call in microscope.connection.calls]
    assert "set_working_distance" in names
    assert "acquire_image" in names


@pytest.mark.parametrize(
    "reported, expected",
    [
        ("Pumped", "Pumped"),
        ("vacuum", "Pumped"),
        ("vented", "Vented"),
        ("prevac", "Unknown"),  # a state the chamber device does not list
    ],
)
def test_chamber_state(microscope, reported, expected):
    microscope.connection.chamber_state = reported
    assert microscope.get("chamber_state") == expected


def test_chamber_pressure(microscope):
    assert microscope.get("chamber_pressure") == pytest.approx(1e-4)


def test_current_choices(microscope):
    assert microscope.get_available_values("current", BeamType.ION) == [
        1e-12,
        1e-11,
        1e-10,
    ]
    # the electron range, stepped by doubling as ThermoMicroscope does
    electron = microscope.get_available_values("current", BeamType.ELECTRON)
    assert electron[0] == pytest.approx(1e-12)
    assert electron[-1] <= 1e-11
    assert all(b == pytest.approx(2 * a) for a, b in zip(electron, electron[1:]))


def test_voltage_choices(microscope):
    assert microscope.get_available_values("voltage", BeamType.ELECTRON) == [
        1000,
        2000,
        3000,
        5000,
        10000,
        20000,
    ]
    assert microscope.get_available_values("voltage", BeamType.ION) == [
        1000,
        2000,
        8000,
        16000,
    ]


# -- milling (FIB-1092) ------------------------------------------------------------


def test_finish_milling_restores_the_beam_and_the_patterning_mode(microscope):
    microscope.finish_milling(imaging_current=20e-12, imaging_voltage=30e3)

    calls = microscope.connection.calls
    assert ("clear_patterns",) in calls
    assert ("set_high_voltage", 30e3, "ion") in calls
    assert ("set_beam_current", 20e-12, "ion") in calls
    assert ("set_patterning_mode", "Serial") in calls


def test_set_patterning_mode_refuses_unknown_modes(microscope):
    with pytest.raises(ValueError):
        microscope.set_patterning_mode("Sideways")


def test_milling_state_is_read_on_the_milling_channel(microscope):
    assert microscope.get_milling_state() is MillingState.IDLE

    names = [call[0] for call in microscope.connection.calls]
    assert names.index("set_active_view") < names.index("get_patterning_state")


@pytest.mark.parametrize(
    "settings",
    [
        FibsemBitmapSettings(
            width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0
        ),
        FibsemPolygonSettings(vertices=[(0, 0), (1e-6, 0), (0, 1e-6)], depth=1e-6),
    ],
)
def test_unsupported_patterns_raise_naming_the_adapter(microscope, settings):
    with pytest.raises(NotImplementedError, match="Delmic AutoScript adapter"):
        microscope.draw_pattern(settings)


def test_a_task_with_an_unsupported_pattern_returns_and_cleans_up(microscope):
    """The raise replaces a silent no-op that left an empty stage to wait on.

    It lands in FibsemMillingTask._mill_stage, which logs a stage's error and moves
    on, so the stage never reports finished and the task's cleanup still restores the
    beam. The task as a whole still reports finished: that is the task's handling of
    any failed stage, on every backend, and not this backend's to change.
    """
    progress = []
    microscope.milling_progress_signal.connect(progress.append)
    stage = FibsemMillingStage(
        name="polygon",
        pattern=PolygonPattern(
            vertices=np.array([[0, 0], [1e-6, 0], [0, 1e-6]]), depth=1e-6
        ),
    )
    config = FibsemMillingTaskConfig.from_stages([stage], name="polygon")
    config.alignment.enabled = False
    config.acquisition.acquire_final_image = False

    FibsemMillingTask(microscope, config).run()

    statuses = [p.status for p in progress]
    assert MillingProgressStatus.STAGE_STARTED in statuses
    assert MillingProgressStatus.STAGE_FINISHED not in statuses
    assert ("set_patterning_mode", "Serial") in microscope.connection.calls
    assert "start_milling" not in [call[0] for call in microscope.connection.calls]


def test_dropped_pattern_settings_are_warned_about(microscope, caplog):
    settings = FibsemRectangleSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0, passes=3
    )
    with caplog.at_level("WARNING"):
        microscope.draw_rectangle(settings)
    assert "passes" in caplog.text

    caplog.clear()
    plain = FibsemRectangleSettings(
        width=1e-6, height=1e-6, depth=1e-6, centre_x=0, centre_y=0
    )
    with caplog.at_level("WARNING"):
        microscope.draw_rectangle(plain)
    assert "ignores them" not in caplog.text


def test_an_annulus_is_drawn_with_its_inner_diameter(microscope):
    microscope.draw_circle(
        FibsemCircleSettings(
            radius=5e-6, depth=1e-6, centre_x=0, centre_y=0, thickness=1e-6
        )
    )
    (parameters,) = [
        call[1] for call in microscope.connection.calls if call[0] == "create_circle"
    ]
    assert parameters["outer_diameter"] == pytest.approx(10e-6)
    assert parameters["inner_diameter"] == pytest.approx(8e-6)


def test_gis_is_not_fitted_and_raises(microscope):
    assert microscope.DEFAULT_FITTED["gis"] is False
    assert microscope.DEFAULT_FITTED["gis_multichem"] is False
    with pytest.raises(NotImplementedError):
        microscope.cryo_deposition_v2(gis_settings=None)
    with pytest.raises(NotImplementedError):
        microscope.setup_sputter({})

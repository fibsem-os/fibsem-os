"""The device prototype on the Demo backend: metadata, validation, signals,
dependencies, resources and commands.
"""

import logging
import math
import os
import tempfile
import threading

import pytest
import yaml

from fibsem import config as cfg
from fibsem import utils
from fibsem.devices import (
    Beam,
    Device,
    Parameter,
    ParameterMetadata,
    ParameterReadOnly,
    ParameterUnavailable,
    Resources,
    command,
)
from fibsem.devices.beam import DEFAULT_WORKING_DISTANCE_LIMITS
from fibsem.structures import BeamType, FibsemImage, RangeLimit

BEAMS = (BeamType.ELECTRON, BeamType.ION)


def _demo(plasma: bool = False):
    """A Demo session; on a plasma FIB (Xenon) with *plasma*."""
    config_path = None
    if plasma:
        with open(cfg.DEFAULT_CONFIGURATION_PATH) as f:
            configuration = yaml.safe_load(f)
        configuration["sim"] = {
            **(configuration.get("sim") or {}),
            "plasma_gas": "Xenon",
        }
        config_path = os.path.join(tempfile.mkdtemp(), "plasma-configuration.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(configuration, f)
    microscope, _ = utils.setup_session(
        config_path=config_path, manufacturer="Demo", setup_logging=False
    )
    return microscope


@pytest.fixture
def microscope():
    return _demo()


@pytest.fixture
def beams(microscope):
    return microscope.beams


# -- the new API --------------------------------------------------------------------


def test_parameters_describe_themselves(beams):
    sem = beams[BeamType.ELECTRON]
    assert sem.voltage.unit == "V"
    assert sem.current.choices == [
        c for c in sem.parent.get_available_values("current", BeamType.ELECTRON)
    ]
    assert sem.scan_rotation.limits == RangeLimit(
        min=0.0, max=2 * math.pi
    )  # static, from the class
    assert sem.hfw.limits == RangeLimit(min=100e-9, max=3e-3)  # from the driver
    # static, from the class: the range the beam panel has always allowed
    assert sem.working_distance.limits == DEFAULT_WORKING_DISTANCE_LIMITS
    assert sem.working_distance.settable
    assert sem.describe()["voltage"] == {
        "type": "float",
        "unit": "V",
        "limits": None,
        "choices": [2000, 5000, 10000, 20000, 30000],
        "settable": True,
    }


def test_absent_read_only_and_settable_are_distinct(beams):
    sem = beams[BeamType.ELECTRON]
    # absent: declared on Beam, not bound by Demo
    assert "preset" in Beam.declared_parameters()
    assert "preset" not in sem.parameters
    assert not hasattr(sem, "preset")
    with pytest.raises(ParameterUnavailable):
        sem.preset.set_value("x")
    # read-only: bound without a write
    sem.bind("preset", read=lambda: "default")
    assert sem.preset.get_value() == "default"
    assert not sem.preset.settable
    with pytest.raises(ParameterReadOnly):
        sem.preset.set_value("x")
    # settable
    assert sem.voltage.settable


def test_numeric_values_clip_to_limits_with_a_warning(beams, caplog):
    sem = beams[BeamType.ELECTRON]
    with caplog.at_level(logging.WARNING):
        written = sem.scan_rotation.set_value(7.0)
    assert written == pytest.approx(2 * math.pi)
    assert sem.scan_rotation.get_value() == pytest.approx(2 * math.pi)
    assert "outside" in caplog.text


def test_invalid_choice_and_wrong_type_raise(beams):
    sem = beams[BeamType.ELECTRON]
    before = sem.current.get_value()
    with pytest.raises(ValueError):
        sem.current.set_value(1.234e-9)
    with pytest.raises(TypeError):
        sem.current.set_value("1 nA")
    with pytest.raises(TypeError):
        sem.hfw.set_value(True)
    assert sem.current.get_value() == before


def test_a_choice_matches_within_float_noise(beams):
    fib = beams[BeamType.ION]
    choice = fib.current.choices[3]
    assert fib.current.set_value(choice * (1 + 1e-9)) == choice


def test_changed_carries_the_value_from_both_apis(microscope, beams):
    sem = beams[BeamType.ELECTRON]
    seen, on_device = [], []
    sem.hfw.changed.connect(seen.append)
    sem.changed.connect(lambda name, value: on_device.append((name, value)))

    sem.hfw.set_value(100e-6)
    microscope.set_field_of_view(50e-6, BeamType.ELECTRON)

    assert seen == [100e-6, 50e-6]
    assert on_device == [("hfw", 100e-6), ("hfw", 50e-6)]


def test_a_live_read_that_finds_a_new_value_emits_changed(microscope, beams):
    sem = beams[BeamType.ELECTRON]
    sem.hfw.get_value()  # prime the cache
    seen = []
    sem.hfw.changed.connect(seen.append)

    sem.sim_beam.hfw = 42e-6  # changed behind our back
    assert sem.hfw.cached == 150e-6  # the cache does not read the instrument
    assert sem.hfw.get_value() == 42e-6
    assert seen == [42e-6]
    assert sem.hfw.get_value() == 42e-6
    assert seen == [42e-6]  # no change, no event


def test_cached_reads_do_not_touch_the_instrument():
    reads = []

    class Thing(Device):
        level = Parameter(float)

    thing = Thing("thing")
    thing.bind("level", read=lambda: reads.append(1) or 3.0, write=lambda v: None)
    thing.level.set_value(5.0)
    for _ in range(3):
        assert thing.level.cached == 5.0
    assert reads == []


def test_dependent_metadata_is_refreshed_and_announced():
    microscope = _demo(plasma=True)
    fib = microscope.beams[BeamType.ION]
    xenon = list(fib.current.choices)
    metadatas = []
    fib.current.metadata_changed.connect(metadatas.append)

    fib.plasma_gas.set_value("Argon")

    assert fib.current.choices == microscope.get_available_values(
        "current", BeamType.ION
    )
    assert fib.current.choices != xenon
    assert metadatas == [fib.current.metadata]
    with pytest.raises(ValueError):
        fib.plasma_gas.set_value("Helium")


def test_a_dependency_changed_outside_refreshes_the_dependent_metadata():
    """A read (or report) that finds a dependency changed behind the device, in the
    vendor UI, reads the dependent's metadata again, as a write does. A read that
    finds it unchanged reads nothing more."""
    kind = {"value": "ETD"}
    modes = {"ETD": ["SE", "BSE"], "ICE": ["SE"]}
    metadata_reads = []

    class Detector(Device):
        kind = Parameter(str)
        mode = Parameter(str, depends_on=("kind",))

    def mode_metadata():
        metadata_reads.append(kind["value"])
        return ParameterMetadata(choices=modes[kind["value"]])

    det = Detector("detector")
    det.bind("kind", read=lambda: kind["value"])
    det.bind("mode", read=lambda: "SE", metadata=mode_metadata)
    det.kind.get_value()
    assert det.mode.choices == ["SE", "BSE"]
    metadata_reads.clear()

    det.kind.get_value()
    assert metadata_reads == []  # unchanged: nothing read again

    kind["value"] = "ICE"  # changed behind the device
    det.kind.get_value()
    assert det.mode.choices == ["SE"]
    assert metadata_reads == ["ICE"]

    kind["value"] = "ETD"
    det.kind.report("ETD")
    assert det.mode.choices == ["SE", "BSE"]


def test_needs_channel_claims_the_resource_and_selects_the_channel():
    resources = Resources()
    events = []

    class Detector(Device):
        contrast = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))

    det = Detector("sem-detector", resources=resources)
    det.bind_channel(lambda: events.append("select"))

    def read():
        lock = resources.lock("imaging_channel")
        # held by this thread while the read runs
        assert lock._is_owned()  # type: ignore[attr-defined]
        events.append("read")
        return 0.5

    det.bind(
        "contrast",
        read=read,
        write=lambda v: events.append(("write", v)),
        needs_channel=True,
    )
    det.contrast.get_value()
    det.contrast.set_value(0.7)
    assert events == ["select", "read", "select", ("write", 0.7)]


def test_a_selected_channel_is_restored_before_the_resource_is_released():
    resources = Resources()
    lock = resources.lock("imaging_channel")
    events = []

    class Detector(Device):
        contrast = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))

    def select():
        events.append("select")
        return lambda: events.append(("restore", lock._is_owned()))  # type: ignore[attr-defined]

    def failing_write(value):
        raise RuntimeError("instrument refused")

    det = Detector("fm-camera", resources=resources)
    det.bind_channel(select)
    det.bind(
        "contrast",
        read=lambda: events.append("read") or 0.5,
        write=failing_write,
        needs_channel=True,
    )

    det.contrast.get_value()
    with pytest.raises(RuntimeError):
        det.contrast.set_value(0.7)
    # claim, select, act, restore, release -- restored even when the act failed
    assert events == ["select", "read", ("restore", True), "select", ("restore", True)]
    assert not lock._is_owned()  # type: ignore[attr-defined]


def test_resources_can_use_a_lock_that_already_exists():
    existing = threading.RLock()
    resources = Resources({"imaging_channel": "view"}, locks={"view": existing})
    assert resources.lock("imaging_channel") is existing
    assert resources.lock("stage") is not existing


def test_resources_share_one_lock_unless_a_backend_separates_them():
    default = Resources()
    assert default.lock("imaging_channel") is default.lock("connection")

    tescan_like = Resources(
        {"connection": "connection", "imaging_channel": "connection"}
    )
    assert tescan_like.lock("imaging_channel") is tescan_like.lock("connection")

    separate = Resources({"imaging_channel": "view", "manipulator": "manipulator"})
    assert separate.lock("imaging_channel") is not separate.lock("manipulator")

    # A remote device gets its own registry and shares nothing with the beams.
    assert Resources().lock("imaging_channel") is not default.lock("imaging_channel")


def test_a_claimed_resource_blocks_another_thread():
    resources = Resources()
    acquired = []
    with resources.claim("imaging_channel"):
        worker = threading.Thread(
            target=lambda: acquired.append(
                resources.lock("connection").acquire(timeout=0.05)
            )
        )
        worker.start()
        worker.join()
    assert acquired == [False]


def test_actions_are_plain_methods_that_describe_themselves(beams):
    sem = beams[BeamType.ELECTRON]
    commands = sem.commands
    assert set(commands) == {
        "acquire",
        "blank",
        "unblank",
        "spot",
        "reduced_area",
        "full_frame",
        "last_image",
        "autocontrast",
        "auto_focus",
        "start_live",
        "stop_live",
    }
    assert commands["acquire"].signature.startswith("(image_settings")
    assert commands["blank"].available
    # the Demo has every imaging hook; a driver without one lacks that command
    assert all(info.available for info in commands.values())

    class Bare(Beam):
        pass

    bare = Bare(BeamType.ELECTRON).commands
    assert not bare["last_image"].available
    assert not bare["auto_focus"].available
    assert not bare["start_live"].available

    sem.blank()
    assert sem.blanked.get_value() is True
    assert isinstance(sem.acquire(), FibsemImage)


def test_a_beam_without_an_acquire_hook_raises_instead_of_recursing():
    class Bare(Beam):
        pass

    with pytest.raises(NotImplementedError, match="no _acquire"):
        Bare(BeamType.ELECTRON).acquire()


def test_an_action_can_be_unavailable():
    class Gun(Device):
        @command(available=lambda gun: False)
        def fire(self) -> None:
            pass

    assert Gun("gun").commands["fire"].available is False


def test_value_property_is_shorthand_for_get_value_and_set_value(beams, caplog):
    sem = beams[BeamType.ELECTRON]
    seen = []
    sem.current.changed.connect(seen.append)

    sem.current.value = 1e-9
    assert sem.current.value == sem.current.get_value() == 1e-9
    assert seen == [1e-9]
    with caplog.at_level(logging.WARNING):
        sem.scan_rotation.value = 7.0
    assert sem.scan_rotation.value == pytest.approx(2 * math.pi)
    assert "outside" in caplog.text
    with pytest.raises(ValueError):
        sem.current.value = 1.234e-9


def test_a_misnamed_implementation_is_an_error_not_an_absent_parameter():
    with pytest.raises(TypeError, match="curent"):

        class Typo(Beam):
            def read_curent(self):
                return 1.0


def test_a_backend_cannot_change_a_parameters_type_or_unit():
    with pytest.raises(TypeError, match="current"):

        class CurrentAsLabel(Beam):
            current = Parameter(str)

    with pytest.raises(TypeError, match="hfw"):

        class HfwInMicrons(Beam):
            hfw = Parameter(float, unit="um")

    class NarrowerRotation(Beam):  # same type and unit, new static limits: allowed
        scan_rotation = Parameter(
            float, unit="rad", limits=RangeLimit(min=0.0, max=1.0)
        )

    assert NarrowerRotation.declared_parameters()["scan_rotation"].limits == RangeLimit(
        min=0.0, max=1.0
    )


def test_needs_channel_is_declared_on_the_backend_class():
    events = []

    class Detector(Device):
        contrast = Parameter(float)
        needs_channel = frozenset({"contrast"})

        def select_channel(self):
            events.append("select")

        def read_contrast(self):
            events.append("read")
            return 0.5

    det = Detector("det").connect()
    det.contrast.get_value()
    assert events == ["select", "read"]
    assert not det.contrast.settable  # no write_contrast: read-only


# FibsemMicroscope.set routes a moved key to its device parameter.


def test_routed_set_emits_the_parameter_change():
    microscope = _demo()
    seen = []
    microscope.beams[BeamType.ELECTRON].hfw.changed.connect(seen.append)
    microscope.set("hfw", 150e-6, BeamType.ELECTRON)
    assert seen == [150e-6]

"""The device prototype on the Demo backend.

Two kinds of test. Parity: every call through the compatibility front gives what the
untouched old call gives, including its no-ops and missing checks. And the new API:
metadata, validation, signals, dependencies, resources and actions.
"""

import logging
import math
import threading

import pytest

from fibsem import utils
from fibsem.devices import (
    BEAM_ROUTES,
    Beam,
    CompatibilityFront,
    Device,
    Parameter,
    ParameterReadOnly,
    ParameterUnavailable,
    ParamMeta,
    Resources,
    action,
    bind_demo_beams,
)
from fibsem.structures import BeamType, FibsemImage

BEAMS = (BeamType.ELECTRON, BeamType.ION)


def _demo(plasma: bool = False):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    if plasma:
        microscope.system.ion.plasma_gas = "Xenon"
    return microscope


@pytest.fixture
def microscope():
    return _demo()


@pytest.fixture
def beams(microscope):
    return bind_demo_beams(microscope)


# -- parity: the old API through the front behaves exactly as before ---------------


@pytest.mark.parametrize("plasma", [False, True])
def test_front_get_matches_old_get_for_every_routed_key(plasma):
    microscope = _demo(plasma)
    front = CompatibilityFront(microscope, bind_demo_beams(microscope))
    for beam_type in BEAMS:
        for key in list(BEAM_ROUTES) + ["shift", "on", "no_such_key"]:
            assert front.get(key, beam_type) == microscope.get(key, beam_type), key
    assert front.get("stage_position") == microscope.get("stage_position")


def test_front_available_values_match_old(microscope, beams):
    front = CompatibilityFront(microscope, beams)
    for beam_type in BEAMS:
        for key in (
            "voltage",
            "current",
            "detector_type",
            "detector_mode",
            "scan_direction",
        ):
            assert front.get_available_values(key, beam_type) == (
                microscope.get_available_values(key, beam_type)
            ), key


# Old calls, including ones the new API would clip or refuse. The old path must not.
OLD_SETS = [
    ("current", 1.234e-9),  # not one of the available currents
    ("voltage", 12345),  # not one of the available voltages
    ("scan_rotation", 7.0),  # above 2*pi: the old path does not clip on Demo
    ("hfw", 1.0),  # Demo has no hfw limit
    ("working_distance", 5e-3),
    ("detector_type", "NotADetector"),
    ("detector_mode", "BackscatteredElectrons"),
    ("blanked", True),
    ("preset", "anything"),  # no-op on Demo, before and after
    ("no_such_key", 1),  # warning and nothing else
]


@pytest.mark.parametrize("beam_type", BEAMS)
@pytest.mark.parametrize("key, value", OLD_SETS)
def test_front_set_leaves_the_same_state_as_old_set(beam_type, key, value):
    old, new = _demo(), _demo()
    front = CompatibilityFront(new, bind_demo_beams(new))

    old.set(key, value, beam_type)
    front.set(key, value, beam_type)

    for read_key in list(BEAM_ROUTES) + ["shift", "stigmation", "resolution"]:
        assert new.get(read_key, beam_type) == old.get(read_key, beam_type), read_key


def test_front_keeps_the_spot_burn_side_effect_of_unblanking(monkeypatch):
    burns = []
    microscope = _demo()
    monkeypatch.setattr(microscope, "_burn_into_sample_scene", burns.append)
    front = CompatibilityFront(microscope, bind_demo_beams(microscope))
    microscope.ion_system.scanning_mode = "spot"

    front.set("blanked", False, BeamType.ION)

    assert burns == [BeamType.ION]


def test_front_keeps_ignoring_an_unavailable_plasma_gas():
    old, new = _demo(plasma=True), _demo(plasma=True)
    front = CompatibilityFront(new, bind_demo_beams(new))

    old.set("plasma_gas", "Helium", BeamType.ION)
    front.set("plasma_gas", "Helium", BeamType.ION)

    assert new.get("plasma_gas", BeamType.ION) == old.get("plasma_gas", BeamType.ION)
    assert new.get("plasma_gas", BeamType.ION) == "Xenon"


def test_the_old_api_is_untouched_by_binding(microscope):
    before = {k: microscope.get(k, BeamType.ELECTRON) for k in BEAM_ROUTES}
    bind_demo_beams(microscope)
    assert {k: microscope.get(k, BeamType.ELECTRON) for k in BEAM_ROUTES} == before


# -- the new API --------------------------------------------------------------------


def test_parameters_describe_themselves(beams):
    sem = beams[BeamType.ELECTRON]
    assert sem.voltage.unit == "V"
    assert sem.current.choices == [
        c for c in sem.parent.get_available_values("current", BeamType.ELECTRON)
    ]
    assert sem.scan_rotation.limits == (0.0, 2 * math.pi)  # static, from the class
    assert sem.hfw.limits is None and sem.hfw.settable
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
    front = CompatibilityFront(microscope, beams)
    seen, on_device = [], []
    sem.hfw.changed.connect(seen.append)
    sem.changed.connect(lambda name, value: on_device.append((name, value)))

    sem.hfw.set_value(100e-6)
    front.set("hfw", 50e-6, BeamType.ELECTRON)

    assert seen == [100e-6, 50e-6]
    assert on_device == [("hfw", 100e-6), ("hfw", 50e-6)]


def test_a_live_read_that_finds_a_new_value_emits_changed(microscope, beams):
    sem = beams[BeamType.ELECTRON]
    sem.hfw.get_value()  # prime the cache
    seen = []
    sem.hfw.changed.connect(seen.append)

    microscope.electron_system.beam.hfw = 42e-6  # changed behind our back
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
    fib = bind_demo_beams(microscope)[BeamType.ION]
    xenon = list(fib.current.choices)
    metas = []
    fib.current.meta_changed.connect(metas.append)

    fib.plasma_gas.set_value("Argon")

    assert fib.current.choices == microscope.get_available_values(
        "current", BeamType.ION
    )
    assert fib.current.choices != xenon
    assert metas == [fib.current.meta]
    with pytest.raises(ValueError):
        fib.plasma_gas.set_value("Helium")


def test_needs_channel_claims_the_resource_and_selects_the_channel():
    resources = Resources()
    events = []

    class Detector(Device):
        contrast = Parameter(float, limits=(0.0, 1.0))

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


def test_resources_share_one_lock_unless_a_backend_separates_them():
    default = Resources()
    assert default.lock("imaging_channel") is default.lock("connection")

    tescan_like = Resources(
        {"connection": "connection", "imaging_channel": "connection"}
    )
    assert tescan_like.lock("imaging_channel") is tescan_like.lock("connection")

    separate = Resources({"imaging_channel": "view", "gis": "gis"})
    assert separate.lock("imaging_channel") is not separate.lock("gis")

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
    actions = sem.actions
    assert set(actions) == {"acquire", "blank", "unblank"}
    assert actions["acquire"].signature.startswith("(image_settings")
    assert actions["blank"].available

    sem.blank()
    assert sem.parent.get("blanked", BeamType.ELECTRON) is True
    assert isinstance(sem.acquire(), FibsemImage)


def test_an_action_can_be_unavailable():
    class Gun(Device):
        @action(available=lambda gun: False)
        def fire(self) -> None:
            pass

    assert Gun("gun").actions["fire"].available is False

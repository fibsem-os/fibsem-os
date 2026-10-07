"""The Odemis FM drivers make the odemis calls the old Odemis FM class made.

``fibsem.drivers.odemis.devices`` is the old ``OdemisFluorescenceMicroscope``'s
parts moved onto the FM devices. Each case runs a device call over a stub odemis
(``_odemis_stubs``) whose components and stream record every read, write and call, and
compares it with the pin of the matching old call: its result and its odemis calls,
recorded from the old class over the same stubs before it was removed
(``tests/fixtures/odemis/old_fm_pins.json``). Cases: every camera, light source, filter
set and objective parameter and command, a channel acquisition (each emission kind,
with and without a gain control) and live view, with the objective in and out and the
stream idle and running.

What has to match:

- the result, or the exception, of every case;
- for the parts, every odemis call in the same order, with three differences, each
  reads only: a device reads its limits and choices once, at connect; the excitation
  write reads back the band odemis took (it snaps to the nearest band); and an objective
  move doesn't read position and state back for a display, which the FM API does after
  it, as for every driver;
- for an acquisition and live view, every change (each write and call) in the same
  order, and the same settings read for the metadata, in another order. Live view pulls
  each frame (``data.get``) where the old one subscribed to the camera.

Nothing here has run on a METEOR.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, List

import numpy as np
import pytest

from fibsem.fm.structures import REFLECTION, ChannelSettings
from fibsem.util.timestamps import iso_from_posix
from tests.fm import _odemis_stubs as stubs

# What the old Odemis FM class did in each case, keyed as the cases are.
PINS = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "odemis" / "old_fm_pins.json").read_text()
)

# -- recording ------------------------------------------------------------------------


class _World:
    """One FM's stub odemis: its components, and what was done to them."""

    def __init__(self, components):
        self.components = components
        self.log: List[tuple] = []


_CURRENT: List[_World] = []


def _plain(value: Any) -> Any:
    if isinstance(value, _Recorded):
        return f"<{object.__getattribute__(value, '_path')}>"
    if isinstance(value, np.ndarray):
        return ("array", value.shape)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_plain(v) for v in value)
    if isinstance(value, float):
        return round(value, 15)
    if callable(value):
        return getattr(value, "__name__", "callable")
    return value


def _pin(value: Any) -> Any:
    """*value* as the pins store it: plain JSON, a set as its sorted items."""

    def default(v: Any) -> Any:
        if isinstance(v, (set, frozenset)):
            return sorted(repr(x) for x in v)
        return repr(v)

    return json.loads(json.dumps(_plain(value), default=default))


def _unwrap(value: Any) -> Any:
    if isinstance(value, _Recorded):
        return object.__getattribute__(value, "_target")
    return value


class _Recorded:
    """A component, VA or dataflow whose every attribute read, write and call is logged
    with its path (``ccd.exposureTime.value``)."""

    def __init__(self, target: Any, path: str, log: List[tuple]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_log", log)

    def __getattr__(self, name: str) -> Any:
        target = object.__getattribute__(self, "_target")
        path = f"{object.__getattribute__(self, '_path')}.{name}"
        log = object.__getattribute__(self, "_log")
        value = getattr(target, name)
        if (
            isinstance(value, (stubs.FakeVA, stubs.FakeDataflow, stubs.FakeCamera))
            or type(value).__module__ == stubs.__name__
            and not callable(value)
        ):
            return _Recorded(value, path, log)
        if callable(value):

            def call(*args, **kwargs):
                log.append(("call", path, _plain(args), _plain(kwargs)))
                return value(*[_unwrap(a) for a in args], **kwargs)

            return call
        log.append(("get", path, _plain(value)))
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        path = f"{object.__getattribute__(self, '_path')}.{name}"
        object.__getattribute__(self, "_log").append(("set", path, _plain(value)))
        setattr(object.__getattribute__(self, "_target"), name, value)


def _get_component(role: str) -> Any:
    world = _CURRENT[-1]
    try:
        return _Recorded(world.components[role], role, world.log)
    except KeyError:
        raise LookupError(f"No stub component with role '{role}'")


def _fluo_stream(*args, **kwargs):
    world = _CURRENT[-1]
    world.log.append(("call", "FluoStream", (), _plain(kwargs)))
    stream = stubs.FakeFluoStream(*args, **kwargs)
    return _Recorded(stream, "stream", world.log)


def _acquire(streams, settings_obs=None):
    world = _CURRENT[-1]
    world.log.append(("call", "acquire", _plain(tuple(streams)), {}))
    stream = _unwrap(streams[0])
    stream._detector = _unwrap(stream._detector)
    return stubs.fake_acquire([stream])


def _has_va(component, name):
    return isinstance(getattr(_unwrap(component), name, None), stubs.FakeVA)


@pytest.fixture(scope="module")
def odemis():
    saved = {}
    for name in stubs.ODEMIS_MODULE_NAMES + stubs.FIBSEM_ODEMIS_MODULE_NAMES:
        if name in sys.modules:
            saved[name] = sys.modules.pop(name)
    stubs.install_odemis_stubs()
    sys.modules["odemis.model"].getComponent = _get_component
    sys.modules["odemis.model"].hasVA = _has_va
    sys.modules["odemis.acq.stream"].FluoStream = _fluo_stream
    sys.modules["odemis.acq.acqmng"].acquire = _acquire
    import fibsem.drivers.odemis.devices as drivers
    import fibsem.fm.odemis as fm_odemis

    yield fm_odemis, drivers

    stubs.remove_odemis_stubs()
    sys.modules.update(saved)


# -- the FM -----------------------------------------------------------------------------


def _components(state: str):
    components = stubs.default_components()
    focus = components["focus"]
    if "inserted" in state:
        focus.position._value = {"z": 8.0e-3}
    if "between" in state:
        focus.position._value = {"z": 3.0e-3}
    if "gain" in state:
        components["ccd"].gain = stubs.FakeVA(1.0)
    if "no-favourites" in state:
        focus._metadata = {}
    return components


def _build(state: str, build: Callable[[], Any]):
    world = _World(_components(state))
    _CURRENT.append(world)
    try:
        built = build()
    finally:
        _CURRENT.pop()
    world.log.clear()
    return world, built


def _prepare(world, stream, state):
    """The stream's state for the case, set without recording."""
    target = _unwrap(stream)
    if "reflection" in state:
        target.emission._value = stubs.BAND_PASS_THROUGH
    if "live" in state:
        target.is_active._value = True


def _run(world, action):
    _CURRENT.append(world)
    try:
        result = action()
    except Exception as e:
        result = f"EXC {type(e).__name__}: {e}"
    finally:
        _CURRENT.pop()
    return result


def _devices(odemis, state):
    _, drivers = odemis
    world, devices = _build(state, lambda: drivers.bind_odemis_fm())
    _prepare(world, devices["fm"]._stream, state)
    return world, devices


def _emission_value(f):
    """A device's emission filter as the old class names it."""
    if f == REFLECTION:
        return None
    return f.low


def _undated(value):
    """*value* with each ``acquisition_date`` reduced to whether there is one.

    The pins hold the old naive string, the stub's POSIX time read in the zone that
    recorded them; a frame now carries it with its offset (FIB-1190), which
    `test_a_frame_carries_odemis_time_with_its_offset` checks exactly. Compared as
    strings, the pins held only in the recording's zone.
    """
    if isinstance(value, dict):
        return {
            k: bool(v) if k == "acquisition_date" else _undated(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_undated(v) for v in value]
    return value


def _same(a, b):
    a, b = _undated(a), _undated(b)
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


# -- the cases ------------------------------------------------------------------------

STATES = ("retracted", "inserted", "between", "reflection", "live", "gain")


def _moved(objective):
    """What the old move's ``_notify_moved`` reads; the device leaves it to the API."""
    objective.position.get_value()
    objective.state.get_value()


def _with(action, then):
    def run():
        result = action()
        then()
        return result

    return run


def _moved_if_it_moved(objective):
    """As `_moved`, but only after a move: an insert or retract with no favourite
    position returns without moving, and the old one announces nothing then."""
    log = _CURRENT[-1].log
    if any(e[1] in ("focus.moveAbs", "focus.moveRel") for e in log):
        _moved(objective)


def _part_cases():
    """(name, device action, how to compare the result with the old one's,
    skip-in-states). ``old`` is the old class's call the pin recorded, kept as the
    record of what each pin is."""
    cases = []

    def add(name, old, new, to_old=lambda r: r, states=STATES):
        cases.append((name, new, to_old, states))

    # camera
    add(
        "camera get exposure_time",
        lambda fm: fm.camera.exposure_time,
        lambda d: d["camera"].exposure_time.get_value(),
    )
    add(
        "camera set exposure_time 0.2",
        lambda fm: setattr(fm.camera, "exposure_time", 0.2),
        lambda d: d["camera"].exposure_time.write_through(0.2),
    )
    add(
        "camera get binning",
        lambda fm: fm.camera.binning,
        lambda d: d["camera"].binning.get_value(),
    )
    for b in (2, 3):
        add(
            f"camera set binning {b}",
            lambda fm, b=b: setattr(fm.camera, "binning", b),
            lambda d, b=b: d["camera"].binning.write_through(b),
        )
    add(
        "camera get gain",
        lambda fm: fm.camera.gain,
        lambda d: d["camera"].gain.get_value(),
        states=("gain",),
    )
    add(
        "camera set gain 2.0",
        lambda fm: setattr(fm.camera, "gain", 2.0),
        lambda d: d["camera"].gain.write_through(2.0),
        states=("gain",),
    )
    add(
        "camera get offset",
        lambda fm: fm.camera.offset,
        lambda d: d["camera"].offset.get_value(),
    )
    add(
        "camera get pixel_size",
        lambda fm: fm.camera.pixel_size,
        lambda d: d["camera"].pixel_size.get_value(),
    )
    add(
        "camera get resolution",
        lambda fm: fm.camera.resolution,
        lambda d: d["camera"].resolution.get_value(),
    )
    add(
        "camera acquire",
        lambda fm: _plain(fm.camera.acquire_image()),
        lambda d: _plain(d["camera"].acquire()),
    )
    # light source
    add(
        "light get power",
        lambda fm: fm.light_source.power,
        lambda d: d["light_source"].power.get_value(),
    )
    for p in (0.5, 1.5):
        add(
            f"light set power {p}",
            lambda fm, p=p: setattr(fm.light_source, "power", p),
            lambda d, p=p: d["light_source"].power.write_through(p),
        )
    # filter set
    add(
        "filter get excitation",
        lambda fm: fm.filter_set.excitation_wavelength,
        lambda d: d["filter_set"].excitation_wavelength.get_value(),
    )
    for nm in (450.0, 365.0):
        add(
            f"filter set excitation {nm}",
            lambda fm, nm=nm: setattr(fm.filter_set, "excitation_wavelength", nm),
            lambda d, nm=nm: d["filter_set"].excitation_wavelength.write_through(nm),
        )
    add(
        "filter get emission",
        lambda fm: fm.filter_set.emission_wavelength,
        lambda d: d["filter_set"].emission_filter.get_value(),
        _emission_value,
    )
    for nm in (None, 420.0, 500.0, 680.0):

        def new(d, nm=nm):
            from fibsem.fm.microscope import emission_filter_named

            param = d["filter_set"].emission_filter
            param.write_through(emission_filter_named(nm, param.choices))

        add(
            f"filter set emission {nm}",
            lambda fm, nm=nm: setattr(fm.filter_set, "emission_wavelength", nm),
            new,
        )
    add(
        "filter set emission Fluorescence",
        lambda fm: setattr(fm.filter_set, "emission_wavelength", "Fluorescence"),
        lambda d: d["filter_set"].select_fluorescence(),
    )
    # objective
    for name in ("position", "magnification", "numerical_aperture", "limit_position"):
        add(
            f"objective get {name}",
            lambda fm, n=name: getattr(fm.objective, n),
            lambda d, n=name: d["objective"].parameters[n].get_value(),
        )
    add(
        "objective get state",
        lambda fm: fm.objective.state,
        lambda d: d["objective"].state.get_value(),
        lambda s: {
            "INSERTED": "Inserted",
            "RETRACTED": "Retracted",
            "UNKNOWN": "Other",
            "ERROR": "Error",
        }[s.name],
    )
    for name, value in (
        ("magnification", 50.0),
        ("numerical_aperture", 0.9),
        ("limit_position", 5.0e-3),
    ):
        add(
            f"objective set {name} {value}",
            lambda fm, n=name, v=value: setattr(fm.objective, n, v),
            lambda d, n=name, v=value: d["objective"].parameters[n].write_through(v),
        )
    for position in (4.0e-3, 9.5e-3, 11.0e-3):
        add(
            f"objective move_absolute {position}",
            lambda fm, p=position: fm.objective.move_absolute(p),
            lambda d, p=position: _with(
                lambda: d["objective"].move_absolute(p), lambda: _moved(d["objective"])
            )(),
        )
    add(
        "objective move_relative 1e-4",
        lambda fm: fm.objective.move_relative(1e-4),
        lambda d: _with(
            lambda: d["objective"].move_relative(1e-4), lambda: _moved(d["objective"])
        )(),
    )
    for command in ("insert", "retract"):
        add(
            f"objective {command}",
            lambda fm, c=command: getattr(fm.objective, c)(),
            lambda d, c=command: _with(
                lambda: getattr(d["objective"], c)(),
                lambda: _moved_if_it_moved(d["objective"]),
            )(),
            # The device says whether it moved; the old method returns nothing.
            lambda moved: None,
        )
    return cases


PART_CASES = _part_cases()


def _cases(states=STATES + ("no-favourites",)):
    for state in states:
        for name, new, to_old, only in PART_CASES:
            if state in only or state == "no-favourites" and "objective" in name:
                yield pytest.param(state, name, new, to_old, id=f"{state}: {name}")


CACHED_AT_CONNECT = {
    "ccd.exposureTime.range",
    "ccd.binning.choices",
    "ccd.binning.range",
    "stream.excitation.choices",
    "stream.emission.choices",
    "focus.axes",
}


# Gain is a fraction of the gain VA's maximum on the devices, so they read its range
# (or choices) to convert; the old class passes gain through in camera units. The
# "gain" state's VA has neither, so the values match and only these reads differ.
GAIN_RANGE_READS = {"ccd.gain.range", "ccd.gain.choices"}


# Writes that read back what odemis took, so the change is cached and signalled.
READBACKS = {
    "set excitation": "stream.excitation.value",
    "emission Fluorescence": "stream.emission.value",
}

# Calls that only read: counted as reads.
READ_CALLS = {"ccd.getMetadata", "focus.getMetadata"}


def _reads(log):
    return {e[1] for e in log if e[0] == "get" or e[1] in READ_CALLS}


def _changes(log, drop=()):
    return [
        e
        for e in log
        if e[0] in ("set", "call") and e[1] not in drop and e[1] not in READ_CALLS
    ]


def _without_reads_of(log, paths):
    return [e for e in log if not (e[0] == "get" and e[1] in paths)]


@pytest.mark.parametrize("state, name, new, to_old", list(_cases()))
def test_each_part_makes_the_old_calls(odemis, state, name, new, to_old):
    pinned = PINS["parts"][f"{state}: {name}"]
    world, devices = _devices(odemis, state)

    new_result = _run(world, lambda: new(devices))

    if not (isinstance(new_result, str) and new_result.startswith("EXC")):
        new_result = to_old(new_result)
    assert _same(pinned["result"], _pin(new_result))
    expected = pinned["log"]
    found = _pin(world.log)
    readback = READBACKS.get(next((k for k in READBACKS if k in name), None))
    if readback:
        assert found[-1][:2] == ["get", readback]
        found = found[:-1]
    assert _without_reads_of(
        found, CACHED_AT_CONNECT | GAIN_RANGE_READS
    ) == _without_reads_of(expected, CACHED_AT_CONNECT)


def test_every_part_case_ran(odemis):
    assert len(list(_cases())) > 200
    assert {p.id for p in _cases()} == set(PINS["parts"])


def test_the_pins_hold_the_calls():
    """Guard against comparing with empty logs."""
    pinned = PINS["parts"]["inserted: objective move_absolute 0.004"]["log"]
    assert [e[:2] for e in pinned] == [
        ["get", "focus.axes"],
        ["call", "focus.moveAbs"],
        ["get", "focus.position.value"],
        ["get", "focus.position.value"],
    ]


def test_connecting_makes_the_old_calls_and_reads_the_limits(odemis):
    _, drivers = odemis
    old_log = PINS["connect"]["log"]
    world = _World(_components("retracted"))
    _CURRENT.append(world)
    drivers.bind_odemis_fm()
    _CURRENT.pop()
    new_log = _pin(world.log)

    assert _changes(new_log) == _changes(old_log)
    assert _reads(old_log) <= _reads(new_log)
    assert _reads(new_log) - _reads(old_log) <= CACHED_AT_CONNECT | {
        "stream.excitation.value",
        "stream.emission.value",
        "stream.power.range",
        "stream.power.unit",  # the watts the power fraction is of, for display
        "stream.power.value",
        "ccd.exposureTime.value",
        "ccd.binning.value",
    }


# -- acquisitions and live view -------------------------------------------------------

CHANNELS = {
    "fluorescence": ChannelSettings(
        excitation_wavelength=450,
        emission_wavelength=500.0,
        power=0.3,
        exposure_time=0.2,
        gain=2.0,
    ),
    "reflection": ChannelSettings(
        excitation_wavelength=550,
        emission_wavelength=None,
        power=0.1,
        exposure_time=0.05,
    ),
    "label": ChannelSettings(
        excitation_wavelength=635,
        emission_wavelength="Fluorescence",
        power=1.0,
        exposure_time=0.5,
    ),
    "current": None,
}

METADATA_KEYS = (
    "exposure_time",
    "binning",
    "power",
    "excitation_wavelength",
    "objective_position",
    "objective_magnification",
    "objective_numerical_aperture",
)


def _old_frame(image):
    md = image.metadata
    ch = md.channels[0]
    found = {key: getattr(ch, key) for key in METADATA_KEYS}
    found["pixel_size"] = (md.pixel_size_x, md.pixel_size_y)
    found["resolution"] = tuple(md.resolution)
    found["acquisition_date"] = md.acquisition_date
    return image.data.shape, found


def _new_frame(frame):
    md = frame.metadata
    found = {key: md.get(key) for key in METADATA_KEYS}
    found["pixel_size"] = tuple(md["pixel_size"])
    found["resolution"] = tuple(md["resolution"])
    found["acquisition_date"] = md["acquisition_date"]
    return frame.data.shape, found


@pytest.mark.parametrize("state", ("retracted", "inserted", "reflection", "gain"))
@pytest.mark.parametrize("channel", list(CHANNELS))
def test_an_acquisition_makes_the_same_changes(odemis, state, channel):
    pinned = PINS["acquisitions"][f"{state}: {channel}"]
    old_log = pinned["log"]
    world, devices = _devices(odemis, state)
    settings = CHANNELS[channel]
    as_dict = settings.to_dict() if settings is not None else None

    new_result = _run(world, lambda: _new_frame(devices["fm"].acquire_frame(as_dict)))
    new_log = _pin(world.log)

    assert _same(pinned["result"], _pin(new_result))
    assert [e[1] for e in _changes(old_log)].count("acquire") == 1
    readback = {"stream.excitation.value"}
    assert _changes(new_log) == _changes(old_log)
    assert _reads(new_log) - readback - GAIN_RANGE_READS <= _reads(old_log)
    assert _reads(old_log) <= _reads(new_log)


def test_a_frame_carries_odemis_time_with_its_offset(odemis):
    """odemis stamps MD_ACQ_DATE (POSIX) at exposure; the frame keeps it, written
    with this machine's offset, over the time `acquire_frame` took before it."""
    world, devices = _devices(odemis, "inserted")
    settings = CHANNELS["fluorescence"]
    frame = _run(world, lambda: devices["fm"].acquire_frame(settings.to_dict()))
    assert frame.metadata["acquisition_date"] == iso_from_posix(1_780_000_000.0)


def stubs_wait(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.mark.parametrize("channel", ["fluorescence", "current"])
def test_live_view_runs_the_stream_as_the_old_live_view(odemis, channel):
    old_log = PINS["live"][channel]["log"]
    new_world, devices = _devices(odemis, "inserted")
    settings = CHANNELS[channel]
    as_dict = settings.to_dict() if settings is not None else None

    def new():
        group = devices["fm"]
        group.start_live(as_dict)
        frames = [group.acquire_frame() for _ in range(3)]
        group.stop_live()
        return len(frames) == 3

    assert _run(new_world, new) is True

    pushed = {"ccd.data.subscribe", "ccd.data.unsubscribe"}
    pulled = {"ccd.data.get"}
    assert _changes(_pin(new_world.log), drop=pulled) == _changes(old_log, drop=pushed)
    gets = [e for e in new_world.log if e[1] == "ccd.data.get"]
    assert len(gets) == 3 and all(e[3] == {"asap": False} for e in gets)
    actives = [
        e[2]
        for e in new_world.log
        if e[1] == "stream.is_active.value" and e[0] == "set"
    ]
    assert actives == [True, False]


def test_live_view_stops_the_stream_when_nobody_pulls(odemis):
    new_world, devices = _devices(odemis, "inserted")
    group = devices["fm"]
    group.live_timeout = 0.2
    _run(new_world, lambda: group.start_live(None))
    assert stubs_wait(lambda: not group.is_live, timeout=3.0)
    assert _unwrap(group._stream).is_active.value is False


def test_the_drivers_are_the_fm_devices(odemis):
    _, drivers = odemis
    world = _World(_components("gain"))
    _CURRENT.append(world)
    try:
        devices = drivers.bind_odemis_fm()
    finally:
        _CURRENT.pop()
    assert {name: type(d).__name__ for name, d in devices.items()} == {
        "fm": "OdemisFM",
        "camera": "OdemisFMCamera",
        "light_source": "OdemisFMLightSource",
        "filter_set": "OdemisFMFilterSet",
        "objective": "OdemisFMObjective",
    }
    assert sorted(devices["camera"].parameters) == [
        "binning",
        "exposure_time",
        "gain",
        "mount_transform",
        "offset",
        "pixel_size",
        "resolution",
    ]
    assert not devices["camera"].offset.settable
    assert devices["camera"].binning.choices == [1, 2, 4, 8, 16]
    assert devices["filter_set"].emission_filter.choices[0] is not None
    assert REFLECTION in devices["filter_set"].emission_filter.choices


def test_a_camera_without_gain_has_no_gain_parameter(odemis):
    _, drivers = odemis
    world = _World(_components("retracted"))
    _CURRENT.append(world)
    try:
        devices = drivers.bind_odemis_fm()
    finally:
        _CURRENT.pop()
    assert "gain" not in devices["camera"].parameters


def _camera_with_gain(drivers, va):
    components = _components("retracted")
    components["ccd"].gain = va
    world = _World(components)
    _CURRENT.append(world)
    try:
        return drivers.bind_odemis_fm()["camera"]
    finally:
        _CURRENT.pop()


def test_gain_is_a_fraction_of_the_cameras_range(odemis):
    _, drivers = odemis
    va = stubs.FakeVA(4.0, range=(0.0, 16.0))
    camera = _camera_with_gain(drivers, va)

    assert camera.gain.get_value() == pytest.approx(0.25)
    camera.gain.write_through(0.5)
    assert va.value == pytest.approx(8.0)
    assert (camera.gain.limits.min, camera.gain.limits.max) == (0.0, 1.0)
    camera.gain.write_through(1.5)  # clipped to the camera's maximum
    assert va.value == pytest.approx(16.0)


def test_gain_with_set_values_takes_the_nearest(odemis):
    _, drivers = odemis
    va = stubs.FakeVA(2.0, choices={1.0, 2.0, 4.0})
    camera = _camera_with_gain(drivers, va)

    assert camera.gain.get_value() == pytest.approx(0.5)
    camera.gain.write_through(0.9)  # 3.6 in camera units
    assert va.value == 4.0


def test_gain_without_a_range_stays_in_camera_units(odemis):
    _, drivers = odemis
    va = stubs.FakeVA(3.0)
    camera = _camera_with_gain(drivers, va)

    assert camera.gain.get_value() == 3.0
    assert camera.gain.limits is None


# -- the FM API over the devices, against the old Odemis class's pins ------------------


def _api_fm(odemis, state):
    fm_odemis, drivers = odemis

    def new_fm():
        devices = drivers.bind_odemis_fm()
        devices["fm"].live_timeout = None
        return fm_odemis.DeviceOdemisFluorescenceMicroscope(devices)

    world, fm = _build(state, new_fm)
    _prepare(world, fm.devices["fm"]._stream, state)
    return world, fm


def _summary(value):
    """An image, or a list of them, as what it shows and what it says it was."""
    from fibsem.fm.structures import FluorescenceImage

    if isinstance(value, FluorescenceImage):
        shape, found = _old_frame(value)
        found["emission_wavelength"] = value.metadata.channels[0].emission_wavelength
        found["gain"] = value.metadata.channels[0].gain
        return shape, found
    if isinstance(value, (list, tuple)):
        return [_summary(v) for v in value]
    return value


def _api_cases():
    from fibsem.fm.acquisition import acquire_channels, acquire_z_stack
    from fibsem.fm.structures import ZParameters, ZStackOrder

    fluorescence, reflection, label = (
        CHANNELS["fluorescence"],
        CHANNELS["reflection"],
        CHANNELS["label"],
    )
    reads = {
        "camera exposure_time": lambda fm: fm.camera.exposure_time,
        "camera binning": lambda fm: fm.camera.binning,
        "camera gain": lambda fm: fm.camera.gain,
        "camera offset": lambda fm: fm.camera.offset,
        "camera pixel_size": lambda fm: fm.camera.pixel_size,
        "camera resolution": lambda fm: fm.camera.resolution,
        "camera available_binnings": lambda fm: fm.camera.available_binnings,
        "camera exposure_time_limits": lambda fm: fm.camera.exposure_time_limits,
        "light power": lambda fm: fm.light_source.power,
        "light power_limits": lambda fm: fm.light_source.power_limits,
        "filter excitation": lambda fm: fm.filter_set.excitation_wavelength,
        "filter emission": lambda fm: fm.filter_set.emission_wavelength,
        "filter excitations": lambda fm: sorted(
            fm.filter_set.available_excitation_wavelengths
        ),
        "filter emissions": lambda fm: sorted(
            fm.filter_set.available_emission_wavelengths, key=lambda v: v or 0
        ),
        "filter emission_bands": lambda fm: sorted(
            fm.filter_set.emission_bands.items()
        ),
        "objective position": lambda fm: fm.objective.position,
        "objective state": lambda fm: fm.objective.state,
        "objective magnification": lambda fm: fm.objective.magnification,
        "objective numerical_aperture": lambda fm: fm.objective.numerical_aperture,
        "objective limits": lambda fm: fm.objective.limits,
        "objective limit_position": lambda fm: fm.objective.limit_position,
        "objective focus_position": lambda fm: fm.objective.focus_position,
    }
    writes = {
        "set exposure_time": lambda fm: setattr(fm.camera, "exposure_time", 0.3),
        "set binning": lambda fm: setattr(fm.camera, "binning", 4),
        "set gain": lambda fm: setattr(fm.camera, "gain", 3.0),
        "set power": lambda fm: setattr(fm.light_source, "power", 0.7),
        "set excitation": lambda fm: setattr(
            fm.filter_set, "excitation_wavelength", 365
        ),
        "set emission None": lambda fm: setattr(
            fm.filter_set, "emission_wavelength", None
        ),
        "set emission 590": lambda fm: setattr(
            fm.filter_set, "emission_wavelength", 590.0
        ),
        "set emission Fluorescence": lambda fm: setattr(
            fm.filter_set, "emission_wavelength", "Fluorescence"
        ),
        "set_channel fluorescence": lambda fm: fm.set_channel(fluorescence),
        "set_channel label": lambda fm: fm.set_channel(label),
        "objective move_absolute": lambda fm: fm.objective.move_absolute(4e-3),
        "objective move_relative": lambda fm: fm.objective.move_relative(1e-4),
        "objective insert": lambda fm: fm.objective.insert(),
        "objective retract": lambda fm: fm.objective.retract(),
    }
    acquisitions = {
        "acquire_image fluorescence": lambda fm: fm.acquire_image(fluorescence),
        "acquire_image reflection": lambda fm: fm.acquire_image(reflection),
        "acquire_image label": lambda fm: fm.acquire_image(label),
        "acquire_image current": lambda fm: fm.acquire_image(None),
        "acquire_channels": lambda fm: acquire_channels(fm, [fluorescence, reflection]),
        "acquire_z_stack by channel": lambda fm: acquire_z_stack(
            fm, [fluorescence, reflection], ZParameters(zmin=-2e-6, zmax=2e-6)
        ),
        "acquire_z_stack by z level": lambda fm: acquire_z_stack(
            fm,
            [fluorescence, reflection],
            ZParameters(zmin=-1e-6, zmax=1e-6, order=ZStackOrder.Z_LEVEL),
        ),
    }
    return {**reads, **writes, **acquisitions}


API_CASES = _api_cases()
API_STATES = ("retracted", "inserted", "reflection", "gain", "no-favourites")


@pytest.mark.parametrize("state", API_STATES)
@pytest.mark.parametrize("name", list(API_CASES))
def test_the_api_makes_the_old_changes(odemis, state, name):
    pinned = PINS["api"][f"{state}: {name}"]
    old_log = pinned["log"]
    world, fm = _api_fm(odemis, state)
    action = API_CASES[name]

    new_result = _pin(_run(world, lambda: _summary(action(fm))))
    new_log = _pin(world.log)

    assert _same(pinned["result"], new_result), (pinned["result"], new_result)
    assert _changes(new_log) == _changes(old_log)
    readbacks = set(READBACKS.values())
    assert _reads(new_log) - readbacks - GAIN_RANGE_READS <= _reads(old_log)
    assert _reads(old_log) - CACHED_AT_CONNECT <= _reads(new_log)


def test_the_api_cases_are_pinned():
    """Guard against comparing with empty logs: an acquisition called odemis."""
    assert {f"{s}: {n}" for s in API_STATES for n in API_CASES} == set(PINS["api"])
    pinned = PINS["api"]["inserted: acquire_image fluorescence"]["log"]
    assert "acquire" in [e[1] for e in _changes(pinned)]


@pytest.mark.parametrize("channel", ["fluorescence", "current"])
def test_api_live_view_runs_the_stream_as_the_old_live_view(odemis, channel):
    old_log = PINS["live"][channel]["log"]
    new_world, new_fm = _api_fm(odemis, "inserted")
    settings = CHANNELS[channel]

    def new():
        new_fm.start_acquisition(settings)
        assert stubs_wait(
            lambda: sum(e[1] == "ccd.data.get" for e in list(new_world.log)) >= 3
        )
        new_fm.stop_acquisition()
        return not new_fm.is_streaming

    assert _run(new_world, new) is True

    pushed = {"ccd.data.subscribe", "ccd.data.unsubscribe"}
    pulled = {"ccd.data.get"}
    assert _changes(_pin(new_world.log), drop=pulled) == _changes(old_log, drop=pushed)
    stream = _unwrap(new_fm.devices["fm"]._stream)
    assert stream.is_active.value is False


def test_an_odemis_microscope_builds_its_fm_from_the_devices(odemis):
    from types import SimpleNamespace

    import fibsem.drivers.odemis.microscope as odemis_microscope
    from fibsem.structures import CameraImageTransform, FluorescenceSystemSettings

    microscope = odemis_microscope.OdemisThermoMicroscope.__new__(
        odemis_microscope.OdemisThermoMicroscope
    )
    microscope.system = SimpleNamespace(
        fm=FluorescenceSystemSettings(mount_transform=CameraImageTransform.FLIP_X)
    )
    world = _World(_components("inserted"))
    _CURRENT.append(world)
    try:
        fm = microscope._connect_fluorescence_devices()
    finally:
        _CURRENT.pop()
    assert type(fm).__name__ == "DeviceOdemisFluorescenceMicroscope"
    assert fm.parent is microscope
    assert sorted(microscope.fm_devices) == [
        "camera",
        "filter_set",
        "fm",
        "light_source",
        "objective",
    ]
    assert fm.devices["fm"].live_timeout is None
    assert all(d.parent is microscope for d in fm.devices.values())
    # As the configuration's fm entry states it.
    assert fm.mount_transform is CameraImageTransform.FLIP_X

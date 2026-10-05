"""Record the AutoScript calls of Thermo's old FM class and of the FM device drivers.

Run as a script, in its own interpreter: it installs a fake
``autoscript_sdb_microscope_client`` in ``sys.modules`` before importing
``fibsem.fm.autoscript``, which must see the SDK at import. It writes JSON to the path
it is given: ``cases``, each holding the old call's result and SDK log and the
driver's, for the test to compare.

Each log entry is ``[kind, path, args, view]``: ``view`` is the connection's active
view when the call was made, so the test can check that every FM call was made with
the FM selected. ``kind`` is ``chan`` for the channel bookkeeping (``get_active_view``,
``set_active_view``, ``set_active_device``), ``get`` for a read and ``call``/``set``
for everything else. Each case also records the view the connection was left on.
"""

import copy
import enum
import json
import logging
import sys
import threading
import types

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

FM_VIEW = 3
BEAM_VIEW = 1
LOG: list = []
STATE = {"view": BEAM_VIEW}


# -- a fake AutoScript SDK, just enough for fibsem.fm.autoscript ----------------------


class CameraEmissionType(enum.Enum):
    BLUE = "Blue"
    GREEN_YELLOW = "GreenYellow"
    RED = "Red"
    VIOLET = "Violet"


class CameraFilterType(enum.Enum):
    REFLECTION = "Reflection"
    FLUORESCENCE = "Fluorescence"


class ImagingDevice(enum.Enum):
    FLUORESCENCE_LIGHT_MICROSCOPE = "FluorescenceLightMicroscope"


class ImagingState(enum.Enum):
    IDLE = "Idle"
    ACQUIRING = "Acquiring"


class GrabFrameSettings:
    def __init__(self, emission_type=None):
        self.emission_type = emission_type


class Limits:
    def __init__(self, min, max):
        self.min, self.max = min, max


class _AnyName:
    def __getattr__(self, name):
        return name


def _install_fake_sdk():
    package = types.ModuleType("autoscript_sdb_microscope_client")
    package.SdbMicroscopeClient = type("SdbMicroscopeClient", (), {})
    build = types.ModuleType("autoscript_sdb_microscope_client.build_information")
    build.INFO_VERSIONSHORT = "4.8.1"
    package.build_information = build
    proxies = types.ModuleType(
        "autoscript_sdb_microscope_client._dynamic_object_proxies"
    )
    enums = types.ModuleType("autoscript_sdb_microscope_client.enumerations")
    structs = types.ModuleType("autoscript_sdb_microscope_client.structures")
    for name, value in {
        "CameraEmissionType": CameraEmissionType,
        "CameraFilterType": CameraFilterType,
        "ImagingDevice": ImagingDevice,
        "ImagingState": ImagingState,
    }.items():
        setattr(enums, name, value)
    structs.GrabFrameSettings = GrabFrameSettings
    structs.Limits = Limits
    # Anything else fibsem.microscopes.autoscript imports, by name.
    for module in (proxies, enums, structs):
        module.__getattr__ = lambda name: type(name, (), {})
    for module in (package, build, proxies, enums, structs):
        sys.modules[module.__name__] = module


_install_fake_sdk()


def _plain(value):
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, GrabFrameSettings):
        return {"emission_type": _plain(value.emission_type)}
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (int, str, bool, type(None))):
        return value
    return repr(value)


def log(kind, path, *args, **kwargs):
    entry_args = _plain(list(args))
    if kwargs:
        entry_args.append(_plain(kwargs))
    LOG.append([kind, path, entry_args, STATE["view"]])


class Setting:
    """A camera or detector setting: a value, its limits or its available values."""

    def __init__(self, path, value, limits=None, available=None):
        self._path, self._value = path, value
        self._limits, self._available = limits, available

    @property
    def value(self):
        log("get", f"{self._path}.value")
        return self._value

    @value.setter
    def value(self, value):
        log("set", f"{self._path}.value", value)
        self._value = value

    @property
    def limits(self):
        log("get", f"{self._path}.limits")
        return Limits(*self._limits)

    @property
    def available_values(self):
        log("get", f"{self._path}.available_values")
        return self._available


class Image:
    def __init__(self, n):
        self.data = np.full((4, 4), n, dtype=np.uint16)


class FakeImaging:
    def __init__(self):
        self._state = ImagingState.IDLE
        self.frames = 0
        self.on_frame = None

    def get_active_view(self):
        log("chan", "imaging.get_active_view")
        return STATE["view"]

    def set_active_view(self, view):
        log("chan", "imaging.set_active_view", view)
        STATE["view"] = view

    def set_active_device(self, device):
        log("chan", "imaging.set_active_device", device)

    @property
    def state(self):
        log("get", "imaging.state")
        return self._state

    def grab_frame(self, settings):
        log("call", "imaging.grab_frame", settings)
        return Image(1)

    def start_acquisition(self):
        log("call", "imaging.start_acquisition")
        self._state = ImagingState.ACQUIRING

    def stop_acquisition(self):
        log("call", "imaging.stop_acquisition")
        self._state = ImagingState.IDLE

    def get_image(self):
        log("call", "imaging.get_image")
        self.frames += 1
        if self.on_frame is not None:
            self.on_frame(self.frames)
        return Image(10 + self.frames)


class FakeEmission:
    def __init__(self):
        self.type = Setting(
            "detector.camera_settings.emission.type", CameraEmissionType.GREEN_YELLOW
        )

    def start(self, emission_type=None):
        log(
            "call",
            "detector.camera_settings.emission.start",
            emission_type=emission_type,
        )

    def stop(self):
        log("call", "detector.camera_settings.emission.stop")


class FakeCameraSettings:
    def __init__(self, filter_mode):
        p = "detector.camera_settings"
        self.exposure_time = Setting(f"{p}.exposure_time", 0.1, limits=(0.001, 10.0))
        self.binning = Setting(f"{p}.binning", 2, available=(1, 2, 4))
        self.focus = Setting(f"{p}.focus", 5e-3, limits=(0.0, 9e-3))
        self.filter = types.SimpleNamespace(
            type=Setting(f"{p}.filter.type", filter_mode)
        )
        self.emission = FakeEmission()


class FakeDetector:
    def __init__(self, objective, filter_mode):
        self._settings = FakeCameraSettings(filter_mode)
        self._state = objective
        self.contrast = Setting("detector.contrast", 0.3)
        self.brightness = Setting("detector.brightness", 0.2, limits=(0.0, 1.0))

    @property
    def camera_settings(self):
        return self._settings

    @property
    def state(self):
        log("get", "detector.state")
        return self._state

    def insert(self):
        log("call", "detector.insert")
        self._state = "Inserted"

    def retract(self):
        log("call", "detector.retract")
        self._state = "Retracted"


def make_connection(objective, filter_mode):
    return types.SimpleNamespace(
        imaging=FakeImaging(), detector=FakeDetector(objective, filter_mode)
    )


# -- the old class and the drivers, each over its own fake connection ----------------

from fibsem.devices.core import IMAGING_CHANNEL, Resources  # noqa: E402
from fibsem.devices.drivers.autoscript_fm import (  # noqa: E402
    MULTI_BAND,
    bind_autoscript_fm,
)
from fibsem.fm.autoscript import ThermoFisherFluorescenceMicroscope  # noqa: E402
from fibsem.fm.structures import REFLECTION, ChannelSettings  # noqa: E402


class Parent:
    """What either side needs of the Thermo microscope: its connection and its lock."""

    def __init__(self, connection):
        self.connection = connection
        self._threading_lock = threading.RLock()
        self.resources = Resources(
            groups={IMAGING_CHANNEL: IMAGING_CHANNEL},
            locks={IMAGING_CHANNEL: self._threading_lock},
        )
        self.experiment = None

    # The image metadata's own reads, which are the coordinator's, not the FM's.
    def get_stage_position(self):
        return None

    def fm_image_geometry(self):
        return None


def old_fm(view, objective, filter_mode):
    connection = make_connection(objective, filter_mode)
    fm = ThermoFisherFluorescenceMicroscope(Parent(connection), connection)
    return fm, connection


def new_fm(view, objective, filter_mode):
    connection = make_connection(objective, filter_mode)
    devices = bind_autoscript_fm(Parent(connection))
    devices["fm"].live_timeout = None
    return devices, connection


def _value(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "sum": int(value.sum())}
    if value is REFLECTION or value == REFLECTION:
        return "REFLECTION"
    if value == MULTI_BAND:
        return "MULTI_BAND"
    if hasattr(value, "data") and hasattr(value, "metadata"):
        return {"data": _value(value.data), "metadata": _metadata(value)}
    if hasattr(value, "min") and hasattr(value, "max"):
        return [value.min, value.max]
    if isinstance(value, enum.Enum):
        return value.name
    return _plain(value)


def _metadata(image):
    """What the frame was taken with, from an old image or a device frame."""
    md = image.metadata
    if isinstance(md, dict):  # a device Frame
        emission = md.get("emission_filter")
        return {
            "exposure_time": md["exposure_time"],
            "gain": md["gain"],
            "offset": md["offset"],
            "binning": md["binning"],
            "pixel_size": list(md["pixel_size"]),
            "resolution": list(md["resolution"]),
            "power": md["power"],
            "excitation_wavelength": md["excitation_wavelength"],
            "emission": "REFLECTION"
            if emission == REFLECTION.to_dict()
            else "MULTI_BAND",
            "objective_position": md["objective_position"],
            "objective_magnification": md["objective_magnification"],
            "objective_numerical_aperture": md["objective_numerical_aperture"],
        }
    ch = md.channels[0]
    return {
        "exposure_time": ch.exposure_time,
        "gain": ch.gain,
        "offset": ch.offset,
        "binning": ch.binning,
        "pixel_size": [md.pixel_size_x, md.pixel_size_y],
        "resolution": list(md.resolution),
        "power": ch.power,
        "excitation_wavelength": ch.excitation_wavelength,
        "emission": "REFLECTION" if ch.emission_wavelength is None else "MULTI_BAND",
        "objective_position": ch.objective_position,
        "objective_magnification": ch.objective_magnification,
        "objective_numerical_aperture": ch.objective_numerical_aperture,
    }


def run(fn, view):
    """What *fn* returns (or raises), every SDK call it made, and the view it left."""
    STATE["view"] = view
    LOG.clear()
    try:
        result = _value(fn())
    except Exception as e:  # recorded, so a raise on one side only is a difference
        result = f"EXC {type(e).__name__}"
    return [result, copy.deepcopy(LOG), STATE["view"]]


def pair(view, objective, filter_mode, old, new):
    """Run *old* on the old class and *new* on the drivers, each over a fresh fake."""
    fm, _ = old_fm(view, objective, filter_mode)
    out = {"old": run(lambda: old(fm), view)}
    devices, _ = new_fm(view, objective, filter_mode)  # binding is not part of a case
    out["new"] = run(lambda: new(devices), view)
    return out


CHANNEL = ChannelSettings(
    name="GFP",
    excitation_wavelength=450,
    emission_wavelength="Fluorescence",
    power=0.4,
    exposure_time=0.25,
    gain=0.5,
)
REFLECTION_CHANNEL = ChannelSettings(
    name="Reflection",
    excitation_wavelength=500,  # between bands: the nearest, 550
    emission_wavelength=None,
    power=0.1,
    exposure_time=0.05,
    gain=0.0,
)


def _scoped(fm, fn):
    with fm.active_channel():
        return fn()


def _old_live(fm, channel, frames):
    """The old live view, run on this thread: the worker's channel, then the fast
    acquisition, stopped after *frames* frames as stopping the acquisition does."""

    def stop_after(n):
        if n >= frames:
            fm._stop_acquisition_event.set()

    fm.connection.imaging.on_frame = stop_after
    fm._stop_acquisition_event.clear()
    if channel is not None:
        fm.set_channel(channel)
    fm.camera._start_fast_acquisition()


def _new_live(devices, channel, frames):
    group = devices["fm"]
    group.start_live(channel.to_dict() if channel is not None else None)
    out = [group.acquire_frame() for _ in range(frames)]
    group.stop_live()
    return out[-1].data


def cases():
    out = []
    for view in (BEAM_VIEW, FM_VIEW):
        for objective in ("Retracted", "Inserted"):
            for mode in (CameraFilterType.FLUORESCENCE, CameraFilterType.REFLECTION):
                tag = f"view={view} objective={objective} filter={mode.name}"

                def add(name, old, new):
                    out.append(
                        {
                            "key": f"{tag} {name}",
                            "group": name.split(" ")[0],
                            **pair(view, objective, mode, old, new),
                        }
                    )

                _cases(add)
    return out


def _moved(d, move):
    """A device move, then what the old objective announces after one: its position
    and state, read in one scope (``ObjectiveLens._notify_moved``). The FM API does
    the same after a device move."""
    move()
    objective = d["objective"]
    with objective._channel.scope():
        objective.position.get_value()
        objective.state.get_value()


def _moved_if(d, move):
    """`_moved`, for insert and retract, which announce only when they moved: the
    command returns False when nothing did."""
    if move() is not False:
        _moved(d, lambda: None)


def _cases(add):
    cam = lambda d: d["camera"]  # noqa: E731
    light = lambda d: d["light_source"]  # noqa: E731
    filters = lambda d: d["filter_set"]  # noqa: E731
    obj = lambda d: d["objective"]  # noqa: E731

    # -- camera
    add(
        "camera get exposure_time",
        lambda fm: fm.camera.exposure_time,
        lambda d: cam(d).exposure_time.get_value(),
    )
    for value in (0.5, 100.0):
        add(
            f"camera set exposure_time {value}",
            lambda fm, v=value: setattr(fm.camera, "exposure_time", v),
            lambda d, v=value: cam(d).exposure_time.write_through(v),
        )
    add(
        "camera exposure_time limits",
        lambda fm: list(fm.camera.exposure_time_limits),
        lambda d: cam(d).metadata_exposure_time().limits,
    )
    add(
        "camera get binning",
        lambda fm: fm.camera.binning,
        lambda d: cam(d).binning.get_value(),
    )
    for value in (4, 3):
        add(
            f"camera set binning {value}",
            lambda fm, v=value: setattr(fm.camera, "binning", v),
            lambda d, v=value: cam(d).binning.write_through(v),
        )
    add(
        "camera binnings",
        lambda fm: list(fm.camera.available_binnings),
        lambda d: cam(d).metadata_binning().choices,
    )
    add("camera get gain", lambda fm: fm.camera.gain, lambda d: cam(d).gain.get_value())
    add(
        "camera set gain",
        lambda fm: setattr(fm.camera, "gain", 0.7),
        lambda d: cam(d).gain.write_through(0.7),
    )
    add(
        "camera get offset",
        lambda fm: fm.camera.offset,
        lambda d: cam(d).offset.get_value(),
    )
    for value in (5.0, -1.0):
        add(
            f"camera set offset {value}",
            lambda fm, v=value: setattr(fm.camera, "offset", v),
            lambda d, v=value: cam(d).offset.write_through(v),
        )
    add(
        "camera pixel_size",
        lambda fm: list(fm.camera.pixel_size),
        lambda d: list(cam(d).pixel_size.get_value()),
    )
    add(
        "camera resolution",
        lambda fm: list(fm.camera.resolution),
        lambda d: list(cam(d).resolution.get_value()),
    )
    # The old camera grabs inside the acquisition's scope; the device grab holds it.
    add(
        "camera acquire",
        lambda fm: _scoped(fm, fm.camera.acquire_image),
        lambda d: cam(d).acquire(),
    )

    # -- light source
    add(
        "light get power",
        lambda fm: fm.light_source.power,
        lambda d: light(d).power.get_value(),
    )
    add(
        "light set power",
        lambda fm: setattr(fm.light_source, "power", 0.4),
        lambda d: light(d).power.write_through(0.4),
    )
    add(
        "light power limits",
        lambda fm: list(fm.light_source.power_limits),
        lambda d: light(d).metadata_power().limits,
    )

    # -- filter set
    add(
        "filter get excitation",
        lambda fm: fm.filter_set.excitation_wavelength,
        lambda d: filters(d).excitation_wavelength.get_value(),
    )
    for value in (450, 500):
        add(
            f"filter set excitation {value}",
            lambda fm, v=value: (
                setattr(fm.filter_set, "excitation_wavelength", v),
                fm.filter_set.excitation_wavelength,
            )[1],
            lambda d, v=value: (
                filters(d).excitation_wavelength.write_through(v),
                filters(d).excitation_wavelength.get_value(),
            )[1],
        )
    add(
        "filter excitations",
        lambda fm: list(fm.filter_set.available_excitation_wavelengths),
        lambda d: filters(d).metadata_excitation_wavelength().choices,
    )
    # The old filter set says fluorescence with the excitation wavelength, as the FM
    # adapter (FMFilterSet) reads it: the multi-band filter.
    add(
        "filter get emission",
        lambda fm: (
            "REFLECTION" if fm.filter_set.emission_wavelength is None else "MULTI_BAND"
        ),
        lambda d: filters(d).emission_filter.get_value(),
    )
    for old_value, new_value in ((None, REFLECTION), ("Fluorescence", MULTI_BAND)):
        add(
            f"filter set emission {old_value}",
            lambda fm, v=old_value: setattr(fm.filter_set, "emission_wavelength", v),
            lambda d, v=new_value: filters(d).emission_filter.write_through(v),
        )

    # -- objective
    add(
        "objective get position",
        lambda fm: fm.objective.position,
        lambda d: obj(d).position.get_value(),
    )
    add(
        "objective limits",
        lambda fm: list(fm.objective.limits),
        lambda d: obj(d).metadata_position().limits,
    )
    add(
        "objective get state",
        lambda fm: fm.objective.state,
        lambda d: {"INSERTED": "Inserted", "RETRACTED": "Retracted"}.get(
            obj(d).state.get_value().name, "Other"
        ),
    )
    add(
        "objective magnification",
        lambda fm: [fm.objective.magnification, fm.objective.numerical_aperture],
        lambda d: [
            obj(d).magnification.get_value(),
            obj(d).numerical_aperture.get_value(),
        ],
    )
    add(
        "objective limit_position",
        lambda fm: (
            setattr(fm.objective, "limit_position", 8.2e-3),
            fm.objective.limit_position,
        )[1],
        lambda d: (
            obj(d).limit_position.write_through(8.2e-3),
            obj(d).limit_position.get_value(),
        )[1],
    )
    for position in (6e-3, 9.5e-3, 8.8e-3):
        add(
            f"objective move_absolute {position}",
            lambda fm, p=position: fm.objective.move_absolute(p),
            lambda d, p=position: _moved(d, lambda: obj(d).move_absolute(p)),
        )
    for delta in (1e-4, -1e-2):
        add(
            f"objective move_relative {delta}",
            lambda fm, x=delta: fm.objective.move_relative(x),
            lambda d, x=delta: _moved(d, lambda: obj(d).move_relative(x)),
        )
    add(
        "objective insert",
        lambda fm: fm.objective.insert(),
        lambda d: _moved_if(d, obj(d).insert),
    )
    add(
        "objective retract",
        lambda fm: fm.objective.retract(),
        lambda d: _moved_if(d, obj(d).retract),
    )

    # -- the group: an acquisition, and live view
    for name, channel in (
        ("fluorescence", CHANNEL),
        ("reflection", REFLECTION_CHANNEL),
    ):
        add(
            f"acquire {name}",
            lambda fm, c=channel: fm.acquire_image(c),
            lambda d, c=channel: d["fm"].acquire_frame(c.to_dict()),
        )
    add(
        "acquire current settings",
        lambda fm: fm.acquire_image(None),
        lambda d: d["fm"].acquire_frame(None),
    )
    for frames in (1, 3):
        add(
            f"live {frames} frames",
            lambda fm, n=frames: (_old_live(fm, CHANNEL, n), None)[1],
            lambda d, n=frames: (_new_live(d, CHANNEL, n), None)[1],
        )


# -- the FM API: today's class, and the FM API over the devices --------------------


def api_pair(view, objective, filter_mode, fn):
    """Run *fn* on today's Thermo FM class and on the FM API over the devices, each
    over a fresh fake: the same call on both, since the API is the same."""
    from fibsem.fm.autoscript import DeviceThermoFisherFluorescenceMicroscope

    fm, _ = old_fm(view, objective, filter_mode)
    out = {"old": run(lambda: _api_value(fn(fm)), view)}
    connection = make_connection(objective, filter_mode)
    parent = Parent(connection)
    devices = bind_autoscript_fm(parent)
    devices["fm"].live_timeout = None  # as ThermoMicroscope builds it
    api = DeviceThermoFisherFluorescenceMicroscope(devices, parent=parent)
    out["new"] = run(lambda: _api_value(fn(api)), view)
    return out


def _api_value(value):
    """An image as its data and metadata, without the time it was taken."""
    if hasattr(value, "data") and hasattr(value, "metadata"):
        md = value.metadata.to_dict()
        md.pop("acquisition_date", None)
        for channel in md.get("channels") or []:
            channel.pop("acquisition_date", None)
        return {"data": _value(value.data), "metadata": _plain(md)}
    return _value(value)


def _live(fm, channel, frames):
    """Live view through the FM API, stopped after *frames* frames, as the UI's stop
    button would: the worker thread runs it, and this waits for it to finish."""

    def stop_after(n):
        if n >= frames:
            fm._stop_acquisition_event.set()

    fm.connection.imaging.on_frame = stop_after
    fm.start_acquisition(channel)
    fm._acquisition_thread.join(10)
    return not fm._acquisition_thread.is_alive()


def _tileset(fm):
    """What a tileset does with the FM: hold the channel for the run, and for each
    tile a z-stack and the channels at the tile's focus."""
    from fibsem.fm.acquisition import acquire_channels, acquire_z_stack
    from fibsem.fm.structures import ZParameters

    zparams = ZParameters(zmin=-2e-6, zmax=2e-6, zstep=2e-6)
    with fm.active_channel():
        for _ in range(2):
            acquire_z_stack(fm, [CHANNEL, REFLECTION_CHANNEL], zparams)
            acquire_channels(fm, [CHANNEL])
    return None


def api_cases():
    from fibsem.fm.acquisition import acquire_channels, acquire_z_stack
    from fibsem.fm.structures import ZParameters, ZStackOrder

    calls = {
        # reads
        "camera reads": lambda fm: [
            fm.camera.exposure_time,
            fm.camera.binning,
            fm.camera.gain,
            fm.camera.offset,
            list(fm.camera.pixel_size),
            list(fm.camera.resolution),
            list(fm.camera.exposure_time_limits),
            list(fm.camera.available_binnings),
        ],
        "light reads": lambda fm: [
            fm.light_source.power,
            list(fm.light_source.power_limits),
        ],
        "filter reads": lambda fm: [
            fm.filter_set.excitation_wavelength,
            fm.filter_set.emission_wavelength,
            list(fm.filter_set.available_excitation_wavelengths),
            list(fm.filter_set.available_emission_wavelengths),
        ],
        "objective reads": lambda fm: [
            fm.objective.position,
            list(fm.objective.limits),
            fm.objective.state,
            fm.objective.magnification,
            fm.objective.numerical_aperture,
            fm.objective.focus_position,
            fm.objective.limit_position,
        ],
        # writes
        "set_channel fluorescence": lambda fm: fm.set_channel(CHANNEL),
        "set_channel reflection": lambda fm: fm.set_channel(REFLECTION_CHANNEL),
        "set_binning": lambda fm: fm.set_binning(4),
        "set_exposure_time out of range": lambda fm: fm.set_exposure_time(100.0),
        "emission then read": lambda fm: (
            setattr(fm.filter_set, "excitation_wavelength", 365),
            setattr(fm.filter_set, "emission_wavelength", 365),
            fm.filter_set.emission_wavelength,
        )[2],
        "objective moves": lambda fm: (
            fm.objective.move_absolute(6e-3),
            fm.objective.move_relative(1e-4),
            fm.objective.move_absolute(8.8e-3),
            fm.objective.position,
        )[3],
        "objective insert retract": lambda fm: (
            fm.objective.insert(),
            fm.objective.retract(),
            fm.objective.state,
        )[2],
        "objective limit_position": lambda fm: (
            setattr(fm.objective, "limit_position", 7e-3),
            fm.objective.move_absolute(8e-3),
            fm.objective.position,
        )[2],
        # acquisitions
        "acquire_image fluorescence": lambda fm: fm.acquire_image(CHANNEL),
        "acquire_image reflection": lambda fm: fm.acquire_image(REFLECTION_CHANNEL),
        "acquire_image current settings": lambda fm: fm.acquire_image(None),
        "acquire_channels": lambda fm: acquire_channels(
            fm, [CHANNEL, REFLECTION_CHANNEL]
        ),
        "acquire_z_stack by channel": lambda fm: acquire_z_stack(
            fm, [CHANNEL, REFLECTION_CHANNEL], ZParameters(zmin=-2e-6, zmax=2e-6)
        ),
        "acquire_z_stack by z level": lambda fm: acquire_z_stack(
            fm,
            [CHANNEL, REFLECTION_CHANNEL],
            ZParameters(zmin=-1e-6, zmax=1e-6, order=ZStackOrder.Z_LEVEL),
        ),
        "tileset": _tileset,
        "live 3 frames": lambda fm: _live(fm, CHANNEL, 3),
        "live current settings": lambda fm: _live(fm, None, 2),
    }
    out = []
    for view in (BEAM_VIEW, FM_VIEW):
        for objective in ("Retracted", "Inserted"):
            for mode in (CameraFilterType.FLUORESCENCE, CameraFilterType.REFLECTION):
                tag = f"view={view} objective={objective} filter={mode.name}"
                for name, fn in calls.items():
                    out.append(
                        {
                            "key": f"{tag} {name}",
                            **api_pair(view, objective, mode, fn),
                        }
                    )
    return out


def _completes_while_held(view, read):
    """Whether *read* finishes while another thread holds the microscope's lock."""
    parent = Parent(make_connection("Retracted", CameraFilterType.FLUORESCENCE))
    devices = bind_autoscript_fm(parent)
    STATE["view"] = view
    held, release, done = threading.Event(), threading.Event(), threading.Event()

    def hold():
        with parent._threading_lock:
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    held.wait(5)
    reader = threading.Thread(target=lambda: (read(devices), done.set()), daemon=True)
    reader.start()
    finished = done.wait(1)
    release.set()
    holder.join(5)
    reader.join(5)
    return finished


def _thermo_microscope_fm():
    """What a Thermo microscope builds its FM from."""
    import fibsem.microscopes.autoscript as A

    microscope = object.__new__(A.ThermoMicroscope)
    microscope.connection = make_connection("Retracted", CameraFilterType.FLUORESCENCE)
    fm = microscope._connect_fluorescence_devices()
    group = fm.devices["fm"]
    return {
        "fm": type(fm).__name__,
        "devices": sorted(fm.devices),
        "live_timeout": group.live_timeout,
        "shares_the_microscope_lock": group._channel.lock is microscope._threading_lock,
        "parent": fm.parent is microscope,
    }


def facts():
    """What the drivers are, beside the parity cases."""
    parent = Parent(make_connection("Retracted", CameraFilterType.FLUORESCENCE))
    devices = bind_autoscript_fm(parent)
    position = lambda d: d["objective"].position.get_value()  # noqa: E731
    return {
        "thermo_microscope": _thermo_microscope_fm(),
        "devices": {name: type(d).__name__ for name, d in devices.items()},
        "parameters": {name: sorted(d.parameters) for name, d in devices.items()},
        "commands": {name: sorted(d.commands) for name, d in devices.items()},
        "shares_the_microscope_lock": all(
            d.resources.lock(IMAGING_CHANNEL) is parent._threading_lock
            for d in devices.values()
        ),
        # The old scope takes no lock when the FM already has the view, so a read
        # isn't starved by live view re-taking the lock every frame.
        "reads_on_the_fm_view_without_the_lock": _completes_while_held(
            FM_VIEW, position
        ),
        "reads_on_the_beam_view_wait_for_the_lock": not _completes_while_held(
            BEAM_VIEW, position
        ),
    }


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(
            {"cases": cases(), "api": api_cases(), "facts": facts()}, f, default=str
        )

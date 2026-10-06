"""Record the AutoScript calls of Thermo's old stage calls and of the stage drivers.

Run as a script, in its own interpreter: it installs a fake
``autoscript_sdb_microscope_client`` in ``sys.modules`` before importing
``fibsem.microscopes.autoscript``, which must see the SDK at import. It writes JSON to
the path it is given: ``cases``, each holding the driver's result and SDK log, and the
old call's on a microscope routed as connect routes it, for the test to compare with
the old code's (``autoscript_old_calls.json``, recorded over this fake before it was
deleted); and ``facts``, what the new API makes of each stage.

The fake stage records every call and attribute write under the vendor path
(``stage.absolute_move``, ``connection.beams.electron_beam.working_distance.value``,
...), and moves: an absolute move writes the axes it was given, a relative move adds
them, so the read-back after a move sees the move.
"""

import copy
import itertools
import json
import logging
import os
import sys
import threading
import types

logging.disable(logging.CRITICAL)

import numpy as np  # noqa: E402

# -- a fake AutoScript SDK, just enough for fibsem.microscopes.autoscript ------------


class _Struct:
    _fields: tuple = ()

    def __init__(self, *args, **kwargs):
        for name in self._fields:
            setattr(self, name, None)
        for name, value in zip(self._fields, args):
            setattr(self, name, value)
        for name, value in kwargs.items():
            setattr(self, name, value)

    def _record(self):
        return dict(vars(self))


class _AnyName:
    def __getattr__(self, name):
        return name


def _install_fake_sdk():
    package = types.ModuleType("autoscript_sdb_microscope_client")
    package.SdbMicroscopeClient = type("SdbMicroscopeClient", (), {})
    build = types.ModuleType("autoscript_sdb_microscope_client.build_information")
    build.INFO_VERSIONSHORT = "4.9.0"
    package.build_information = build

    proxies = types.ModuleType(
        "autoscript_sdb_microscope_client._dynamic_object_proxies"
    )
    for name in (
        "CirclePattern",
        "CleaningCrossSectionPattern",
        "LinePattern",
        "RectanglePattern",
        "RegularCrossSectionPattern",
    ):
        setattr(proxies, name, type(name, (), {}))

    enums = types.ModuleType("autoscript_sdb_microscope_client.enumerations")
    enums.CoordinateSystem = type(
        "CoordinateSystem", (), {"RAW": "Raw", "SPECIMEN": "Specimen"}
    )
    for name in (
        "ImagingState",
        "ManipulatorCoordinateSystem",
        "ManipulatorSavedPosition",
        "ManipulatorState",
        "PatterningState",
        "RegularCrossSectionScanMethod",
    ):
        setattr(enums, name, _AnyName())

    structs = types.ModuleType("autoscript_sdb_microscope_client.structures")
    fields = {
        "StagePosition": ("x", "y", "z", "r", "t", "coordinate_system"),
        "CompustagePosition": ("x", "y", "z", "a", "coordinate_system"),
        "MoveSettings": ("rotate_compucentric",),
        "Limits": ("min", "max"),
        "Point": ("x", "y"),
    }
    for name in (
        "AdornedImage",
        "BitmapPatternDefinition",
        "CompustagePosition",
        "GetImageSettings",
        "GrabFrameSettings",
        "Limits",
        "Limits2d",
        "ManipulatorPosition",
        "MoveSettings",
        "Point",
        "Rectangle",
        "StagePosition",
    ):
        setattr(
            structs, name, type(name, (_Struct,), {"_fields": fields.get(name, ())})
        )

    for module in (package, build, proxies, enums, structs):
        sys.modules[module.__name__] = module
    return structs


STRUCTS = _install_fake_sdk()
LOG: list = []


def _plain(value):
    if hasattr(value, "_record"):
        return {
            "type": type(value).__name__,
            **{k: _plain(v) for k, v in value._record().items()},
        }
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (int, str, bool, type(None))):
        return value
    return repr(value)


class Node:
    """Any vendor path: records calls and writes, returns a child for any attribute."""

    def __init__(self, path):
        object.__setattr__(self, "_path", path)
        object.__setattr__(self, "_kids", {})

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        kids = object.__getattribute__(self, "_kids")
        if name not in kids:
            kids[name] = Node(f"{self._path}.{name}")
        return kids[name]

    def __setattr__(self, name, value):
        LOG.append(["set", f"{self._path}.{name}", _plain(value)])
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        LOG.append(["call", self._path, _plain(list(args)), _plain(kwargs)])


class FakeStage(Node):
    """``specimen.stage`` or ``specimen.compustage``: moves, and logs reads."""

    def __init__(self, path, position, limits):
        super().__init__(path)
        object.__setattr__(self, "_position", position)
        object.__setattr__(self, "_limits", limits)

    def _log(self, kind, name, *args):
        LOG.append([kind, f"{self._path}.{name}", _plain(list(args)), {}])

    @property
    def current_position(self):
        self._log("get", "current_position")
        return copy.deepcopy(self._position)

    @property
    def is_homed(self):
        self._log("get", "is_homed")
        return True

    @property
    def is_linked(self):
        self._log("get", "is_linked")
        return False

    def set_default_coordinate_system(self, system):
        self._log("call", "set_default_coordinate_system", system)

    def get_axis_limits(self, axis):
        self._log("call", "get_axis_limits", axis)
        low, high = self._limits[axis]
        return STRUCTS.Limits(min=low, max=high)

    def absolute_move(self, position, *args):
        self._log("call", "absolute_move", position, *args)
        for name in position._fields:
            value = getattr(position, name, None)
            if name != "coordinate_system" and value is not None:
                setattr(self._position, name, value)

    def relative_move(self, position, *args):
        self._log("call", "relative_move", position, *args)
        for name in position._fields:
            value = getattr(position, name, None)
            if name != "coordinate_system" and value is not None:
                setattr(self._position, name, getattr(self._position, name) + value)


import fibsem.config as cfg  # noqa: E402
import fibsem.microscopes.autoscript as A  # noqa: E402
from fibsem import utils  # noqa: E402
from fibsem.devices.drivers.autoscript import (  # noqa: E402
    AutoscriptCompustage,
    bind_autoscript_stage,
)
from fibsem.structures import BeamType, FibsemStagePosition  # noqa: E402

assert A.THERMO_API_AVAILABLE, A.THERMO_API_IMPORT_ERROR

OFFSET_LIMITS = {
    "x": (-55e-3, 55e-3),
    "y": (-55e-3, 55e-3),
    "z": (0.0, 10e-3),
    "t": (np.radians(-15), np.radians(70)),
}
SYSTEM = utils.load_microscope_configuration(
    os.path.join(cfg.CONFIG_PATH, "microscope-configuration.yaml")
).system


class _Objective:
    state = "Inserted"
    blocked_axes = ("z", "t")


class _FM:
    objective = _Objective()


def make(compustage, fm_inserted=False):
    """A ThermoMicroscope as connect leaves it, over the fake SDK."""
    microscope = object.__new__(A.ThermoMicroscope)
    microscope._connection_lock = threading.RLock()
    microscope.system = copy.deepcopy(SYSTEM)
    microscope.stage_is_compustage = compustage
    microscope.fm = _FM() if fm_inserted else None
    microscope._stage_position = None
    connection = Node("connection")
    object.__setattr__(connection.beams.electron_beam.working_distance, "value", 4e-3)
    if compustage:
        position = STRUCTS.CompustagePosition(
            x=1e-4, y=2e-4, z=3e-5, a=np.radians(-10), coordinate_system="Specimen"
        )
        microscope._default_stage_coordinate_system = "Specimen"
        stage = FakeStage("specimen.compustage", position, OFFSET_LIMITS)
    else:
        position = STRUCTS.StagePosition(
            x=1e-4,
            y=2e-4,
            z=3e-3,
            r=np.radians(49),
            t=np.radians(18),
            coordinate_system="Raw",
        )
        microscope._default_stage_coordinate_system = "Raw"
        stage = FakeStage("specimen.stage", position, OFFSET_LIMITS)
    microscope.connection = connection
    microscope._vendor_stage = stage
    return microscope


def _preset(node, path, value):
    """Give a vendor attribute a value without recording it as a write."""
    *parents, name = path.split(".")
    for part in parents:
        node = getattr(node, part)
    object.__setattr__(node, name, value)


def _fake_beam(beam, beam_type):
    electron = beam_type is BeamType.ELECTRON
    values = {
        "is_on": True,
        "is_blanked": False,
        "working_distance.value": 4e-3 if electron else 16.5e-3,
        "beam_current.value": 1e-10 if electron else 2e-11,
        "beam_current.limits": STRUCTS.Limits(min=1e-12, max=1e-8),
        "beam_current.available_values": [1e-12, 2e-11, 1e-10, 1e-9],
        "high_voltage.value": 2000 if electron else 30000,
        "high_voltage.limits": STRUCTS.Limits(min=200, max=30000),
        "horizontal_field_width.value": 150e-6,
        "horizontal_field_width.limits": STRUCTS.Limits(min=1e-7, max=2e-3),
        "scanning.dwell_time.value": 1e-6,
        "scanning.rotation.value": 0.0,
        "scanning.resolution.value": "1536x1024",
        "scanning.mode.value": "FullFrame",
        "beam_shift.value": STRUCTS.Point(x=1e-7, y=-2e-7),
        "stigmator.value": STRUCTS.Point(x=0.01, y=-0.02),
        "source.plasma_gas.value": "Xenon",
        "source.plasma_gas.available_values": ["Argon", "Oxygen", "Xenon"],
    }
    if electron:
        values["angular_correction.angle.value"] = 0.05
        values["angular_correction.tilt_correction.is_on"] = False
    for path, value in values.items():
        _preset(beam, path, value)


def fake_beams(connection):
    """Give both vendor beams and the detector the values the beam drivers read."""
    _fake_beam(connection.beams.electron_beam, BeamType.ELECTRON)
    _fake_beam(connection.beams.ion_beam, BeamType.ION)
    _preset(connection, "detector.type.value", "ETD")
    _preset(connection, "detector.type.available_values", ["ETD", "TLD", "ICE"])
    _preset(connection, "detector.mode.value", "SecondaryElectrons")
    _preset(
        connection,
        "detector.mode.available_values",
        ["SecondaryElectrons", "BackscatterElectrons"],
    )
    _preset(connection, "detector.brightness.value", 0.5)
    _preset(connection, "detector.contrast.value", 0.6)


def connect(microscope):
    """Build the beams and the stage as connect does: an absolute move on an offset
    stage restores the working distance through the electron beam."""
    fake_beams(microscope.connection)
    microscope._build_beams()
    microscope._build_stage()


def _value(value):
    if isinstance(value, FibsemStagePosition):
        return [value.x, value.y, value.z, value.r, value.t, value.coordinate_system]
    if isinstance(value, dict):
        return {k: [v.min, v.max] for k, v in value.items()}
    return _plain(value)


def run(fn):
    """What *fn* returns (or raises), and every SDK call and write it made."""
    LOG.clear()
    try:
        result = _value(fn())
    except Exception as e:  # recorded, so a raise on one side only is a difference
        result = f"EXC {type(e).__name__}: {e}"
    return [result, copy.deepcopy(LOG)]


def pair(compustage, fm_inserted, old, new):
    """Run *new* on a fresh microscope's driver, and *old* on another whose stage keys
    and moves are routed as connect routes them."""
    m_new = make(compustage, fm_inserted)
    m_routed = make(compustage, fm_inserted)
    for microscope in (m_new, m_routed):
        connect(microscope)
    stage = m_new.stage
    LOG.clear()  # connect's calls are not part of the case
    return {
        "new": run(lambda: new(stage, m_new)),
        "routed": run(lambda: old(m_routed)),
    }


def positions(compustage):
    """Absolute targets: full, partial, and poses in each orientation."""
    tilts = (-180, -38, 0, 18, 52)
    rotations = (0,) if compustage else (0, 49, 229)
    for t, r in itertools.product(tilts, rotations):
        yield FibsemStagePosition(
            x=2e-4, y=-1e-4, z=4e-5, r=np.radians(r), t=np.radians(t)
        )
    yield FibsemStagePosition(x=5e-5, y=6e-5)
    yield FibsemStagePosition(t=np.radians(30))
    yield FibsemStagePosition(r=np.radians(180), t=np.radians(18))
    yield FibsemStagePosition(z=1e-4)


def deltas():
    yield FibsemStagePosition(x=1e-5, y=-2e-5)
    yield FibsemStagePosition(x=1e-5, y=-2e-5, z=3e-6, r=0.1, t=0.05)
    yield FibsemStagePosition(t=np.radians(-5))
    yield FibsemStagePosition(z=-1e-6)


def cases():
    out = []
    for compustage, fm_inserted in ((False, False), (True, False), (True, True)):
        tag = f"compustage={compustage} fm={fm_inserted}"

        def add(name, old, new):
            out.append(
                {"key": f"{tag} {name}", **pair(compustage, fm_inserted, old, new)}
            )

        # connect: the old limits (converted) and the calls reading them
        def old_limits(m):
            limits = m._get_axis_limits()
            return {
                k: A.RangeLimit(
                    min=float(np.radians(v.min)) if k in "rt" else v.min,
                    max=float(np.radians(v.max)) if k in "rt" else v.max,
                )
                for k, v in limits.items()
            }

        def new_limits(stage, m):
            LOG.clear()
            return bind_autoscript_stage(m).position.limits

        add("limits", old_limits, new_limits)
        add(
            "class",
            lambda m: "compustage" if m.stage_is_compustage else "stage",
            lambda s, m: (
                "compustage" if isinstance(s, AutoscriptCompustage) else "stage"
            ),
        )

        add(
            "get position",
            lambda m: m.get("stage_position"),
            lambda s, m: s.position.get_value(),
        )
        add(
            "get_stage_position",
            lambda m: m.get_stage_position(),
            lambda s, m: s.position.get_value(),
        )
        add(
            "get homed",
            lambda m: m.get("stage_homed"),
            lambda s, m: s.homed.get_value(),
        )
        add("home", lambda m: m.set("stage_home", True), lambda s, m: s._home())
        if not compustage:
            add(
                "get linked",
                lambda m: m.get("stage_linked"),
                lambda s, m: s.linked.get_value(),
            )
            add("link", lambda m: m.set("stage_link", True), lambda s, m: s._link())
            add(
                "link_stage",
                lambda m: m.link_stage(),
                lambda s, m: (s._link(), s.linked.get_value())[1],
            )
        add("home()", lambda m: m.home(), lambda s, m: (s._home(), True)[1])
        add(
            "unlink",
            lambda m: m.set("stage_link", False),
            lambda s, m: m.set("stage_link", False),  # no device unlink: the old path
        )

        for i, position in enumerate(positions(compustage)):
            add(
                f"absolute {i} {position}",
                lambda m, p=position: m.move_stage_absolute(p),
                lambda s, m, p=position: s.move_through(p),
            )
        for i, delta in enumerate(deltas()):
            add(
                f"relative {i} {delta}",
                lambda m, d=delta: m.move_stage_relative(d),
                lambda s, m, d=delta: s.move_through(d, relative=True),
            )
    return out


def facts():
    """What the new API makes of each stage, beside the parity cases."""
    from fibsem.devices import StageLimitError

    out = {}
    for compustage in (False, True):
        microscope = make(compustage)
        connect(microscope)
        stage = microscope.stage
        LOG.clear()
        try:
            stage.move_absolute(FibsemStagePosition(x=1.0))
            refused = None
        except StageLimitError as e:
            refused = str(e)
        out["compustage" if compustage else "stage"] = {
            "class": type(stage).__name__,
            "axes": list(stage.axes),
            "parameters": sorted(stage.parameters),
            "link_available": stage.commands["link"].available,
            "refused": refused,
            "calls_after_refusal": copy.deepcopy(LOG),
            "moved": _value(stage.move_absolute(FibsemStagePosition(x=5e-5))),
        }
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump({"cases": cases(), "facts": facts()}, f, default=str)

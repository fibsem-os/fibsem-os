"""TESCAN and Odemis stage moves return the position they ended at (FIB-1063).

``FibsemMicroscope.move_stage_absolute`` and ``move_stage_relative`` are declared
to return the stage position, and ThermoFisher and Demo always have. TESCAN's
absolute move and Odemis's absolute and relative moves returned nothing, which
failed the HTTP server's move endpoints after the stage had moved, and left the
recorded ``stage_moved`` without the position it ended at.

No SDK or hardware: each microscope is created without ``__init__`` and talks to
a fake of its vendor's stage, below the driver's own code -- so the driver's
conversions and its position read are the ones under test.
"""

import os
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscopes.tescan import TescanMicroscope
from fibsem.structures import FibsemStagePosition
from tests.fm import _odemis_stubs as stubs


class _TescanStage:
    """The SDK's stage: MoveTo sets the axes it is given, in mm and degrees."""

    def __init__(self):
        self.axes = [0.0, 0.0, 0.0, 0.0, 0.0]  # x, y, z, rot, tilt
        self.reads = 0

    def MoveTo(self, x=None, y=None, z=None, rot=None, tiltx=None):
        for i, value in enumerate((x, y, z, rot, tiltx)):
            if value is not None:
                self.axes[i] = value

    def GetPosition(self):
        self.reads += 1
        return tuple(self.axes)


def _tescan():
    microscope = object.__new__(TescanMicroscope)  # skip __init__ (requires the SDK)
    microscope._connection_lock = threading.RLock()
    microscope.system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "tescan-configuration.yaml")
    ).system
    microscope.stage_is_compustage = False
    microscope.fm = None
    stage = _TescanStage()
    microscope.connection = SimpleNamespace(Stage=stage)
    return microscope, stage


def _close(a: FibsemStagePosition, b: FibsemStagePosition) -> bool:
    return all(np.isclose(getattr(a, k), getattr(b, k)) for k in "xyzrt")


TARGET = FibsemStagePosition(
    x=20e-6, y=-10e-6, z=5e-6, r=np.radians(30), t=np.radians(15)
)


def test_a_tescan_absolute_move_returns_where_the_stage_ended():
    microscope, stage = _tescan()

    moved = microscope.move_stage_absolute(TARGET)

    assert isinstance(moved, FibsemStagePosition)
    assert _close(moved, TARGET)
    assert stage.reads == 1  # the one read after the move


def test_a_tescan_relative_move_reads_the_stage_no_more_than_before():
    # It read before and after the move; the read after is now the absolute
    # move's, not a second one of its own.
    microscope, stage = _tescan()
    microscope.move_stage_absolute(TARGET)
    stage.reads = 0

    moved = microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0)
    )

    assert moved.x == pytest.approx(TARGET.x + 1e-6)
    assert stage.reads == 2


def test_a_tescan_safe_move_records_where_it_ended():
    # Every task's move to its lamella: it recorded the position from before it.
    microscope, _ = _tescan()
    microscope.get_stage_position()  # read on connect
    events = []
    microscope.record_signal.connect(
        lambda kind, payload: events.append((kind, payload))
    )

    microscope.safe_absolute_stage_movement(TARGET)

    ((kind, move),) = events
    assert move["move"] == "safe_absolute_stage_movement"
    assert _close(FibsemStagePosition.from_dict(move["end"]), TARGET)


# ── Odemis ────────────────────────────────────────────────────────────────────


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


class _OdemisStage:
    """Odemis's stage actuator: moves are futures, the position a VA of a dict."""

    def __init__(self):
        self.position = SimpleNamespace(
            value={"x": 0.0, "y": 0.0, "z": 0.0, "rx": 0.0, "rz": 0.0}
        )

    def _done(self):
        return SimpleNamespace(result=lambda: None)

    def moveAbs(self, axes):
        self.position.value = {**self.position.value, **axes}
        return self._done()

    def moveRel(self, axes):
        value = dict(self.position.value)
        for k, v in axes.items():
            value[k] += v
        self.position.value = value
        return self._done()


def _odemis(cls):
    microscope = object.__new__(cls)  # skip __init__ (requires odemis)
    microscope.system = utils.load_microscope_configuration(
        os.path.join(cfg.CONFIG_PATH, "odemis-configuration.yaml")
    ).system
    microscope.stage_is_compustage = False
    microscope.stage = _OdemisStage()
    return microscope


def test_an_odemis_absolute_move_returns_where_the_stage_ended(odemis_microscope_cls):
    microscope = _odemis(odemis_microscope_cls)

    moved = microscope.move_stage_absolute(TARGET)

    assert isinstance(moved, FibsemStagePosition)
    assert _close(moved, TARGET)


def test_an_odemis_relative_move_returns_where_the_stage_ended(odemis_microscope_cls):
    microscope = _odemis(odemis_microscope_cls)
    microscope.move_stage_absolute(TARGET)

    moved = microscope.move_stage_relative(
        FibsemStagePosition(x=1e-6, y=0, z=0, r=0, t=0)
    )

    assert moved.x == pytest.approx(TARGET.x + 1e-6)
    assert moved.y == pytest.approx(TARGET.y)


# ── the server ────────────────────────────────────────────────────────────────


def test_the_server_move_endpoint_answers_with_where_a_tescan_stage_went():
    # It called result.to_dict() on None, after the stage had moved.
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from fibsem.server import AuthConfig, build_server

    microscope, _ = _tescan()
    app = build_server(
        microscope, auth=AuthConfig.generate(arm_hardware=True, token="t")
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/move_stage_absolute",
            headers={"Authorization": "Bearer t"},
            json={"position": {**TARGET.to_dict(), "coordinate_system": "RAW"}},
        )

    assert resp.status_code == 200, resp.text
    assert _close(FibsemStagePosition.from_dict(resp.json()["position"]), TARGET)

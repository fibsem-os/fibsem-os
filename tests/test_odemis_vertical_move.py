"""OdemisThermoMicroscope.vertical_move reaches the ThermoFisher bodies intact.

Odemis delegates to ``ThermoMicroscope.vertical_move``, which calls
``self._vertical_move_from_fib/sem(..., relaxation=...)``. On an Odemis
microscope ``self`` resolves to Odemis's own overrides, so every one of them
has to take ``relaxation`` or every vertical move raises ``TypeError``.

The oracle is ThermoMicroscope itself: given the same stage state, an Odemis
vertical move must command exactly the moves a ThermoFisher one does.

No odemis installation or hardware required: odemis is replaced by the stub
modules in tests/fm/_odemis_stubs.py, the microscopes are created without
__init__, and the stage is a recorded in-memory position.
"""

import inspect
import os
import sys

import numpy as np
import pytest

import fibsem.config as cfg
from fibsem import utils
from fibsem.microscope import FibsemMicroscope
from fibsem.microscopes.autoscript import ThermoMicroscope
from fibsem.structures import BeamType, FibsemStagePosition
from tests.fm import _odemis_stubs as stubs

ODEMIS_CONFIG_PATH = os.path.join(cfg.CONFIG_PATH, "odemis-configuration.yaml")


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


def make_microscope(cls, tilt_deg: float = 18.0):
    """Create a microscope without __init__, with a stubbed, recording stage."""
    microscope = object.__new__(cls)  # skip __init__ (requires hardware)
    microscope.system = utils.load_microscope_configuration(ODEMIS_CONFIG_PATH).system
    microscope.stage_is_compustage = False

    microscope._position = FibsemStagePosition(
        x=0, y=0, z=0, r=0, t=np.deg2rad(tilt_deg), coordinate_system="RAW"
    )
    microscope._moves = []

    def move_stage_relative(position):
        microscope._moves.append(position)
        microscope._position = microscope._position + position

    def stable_move(dx, dy, beam_type, static_wd=False):
        # the stable-move math is not under test; any displacement will do
        move_stage_relative(FibsemStagePosition(x=dx, y=dy, z=dy / 2))
        return microscope._position

    microscope.get_stage_position = lambda: microscope._position
    microscope.move_stage_relative = move_stage_relative
    microscope.stable_move = stable_move
    microscope.get_working_distance = lambda beam_type: 4e-3
    microscope.set_working_distance = lambda wd, beam_type: None
    microscope.get_scan_rotation = lambda beam_type: 0.0
    microscope.get_stage_orientation = lambda: "SEM"
    microscope.get_current_milling_angle = lambda: 18.0
    return microscope


def recorded_moves(microscope):
    return [(m.x, m.y, m.z) for m in microscope._moves]


@pytest.mark.parametrize("beam_type", [BeamType.ION, BeamType.ELECTRON])
def test_a_default_vertical_move_does_not_raise(odemis_microscope_cls, beam_type):
    """The reported bug: ThermoMicroscope.vertical_move always passes relaxation on,
    so an override without it failed every call, not only tuned ones."""
    microscope = make_microscope(odemis_microscope_cls)

    microscope.vertical_move(dy=5e-6, beam_type=beam_type)

    assert microscope._moves


@pytest.mark.parametrize("relaxation", [1.0, 0.5])
@pytest.mark.parametrize("beam_type", [BeamType.ION, BeamType.ELECTRON])
def test_odemis_moves_the_stage_as_thermo_does(
    odemis_microscope_cls, beam_type, relaxation
):
    """relaxation is forwarded, not merely accepted: at 0.5 a dropped argument
    would move the full distance and disagree with ThermoMicroscope."""
    odemis = make_microscope(odemis_microscope_cls)
    thermo = make_microscope(ThermoMicroscope)

    odemis.vertical_move(dy=5e-6, beam_type=beam_type, relaxation=relaxation)
    thermo.vertical_move(dy=5e-6, beam_type=beam_type, relaxation=relaxation)

    assert recorded_moves(odemis) == recorded_moves(thermo)


def test_odemis_takes_every_parameter_the_base_declares(odemis_microscope_cls):
    """The same check as test_vertical_move_views.py makes of the other backends."""
    base = inspect.signature(FibsemMicroscope.vertical_move).parameters
    override = inspect.signature(odemis_microscope_cls.vertical_move).parameters
    for name, param in base.items():
        assert name in override, f"OdemisThermoMicroscope.vertical_move has no {name}"
        assert override[name].default == param.default, name

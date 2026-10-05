"""The AutoScript (Thermo Fisher) stage as a device, beside the untouched Thermo backend.

``AutoscriptStage`` implements the ``Stage`` device with what ``ThermoMicroscope`` does
today, moved as-is, so the old call and the device make the same SDK calls in the same
order. ``AutoscriptCompustage`` is the same for a compustage (Arctis, Hydra). Nothing
builds them yet: ``ThermoMicroscope`` still moves its stage itself, and pointing it at
the device is a later step.

The vendor stage is ``microscope.stage``, which the Thermo backend sets at connect to
``specimen.stage`` or ``specimen.compustage``. This module imports the SDK only through
``fibsem.microscopes.autoscript``, which is where the guarded import lives.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, Optional, Type

import numpy as np

from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.stage import Stage, axis_limits_from_degrees
from fibsem.structures import BeamType, FibsemStagePosition, RangeLimit

if TYPE_CHECKING:
    from fibsem.microscopes.autoscript import ThermoMicroscope


class AutoscriptStage(Stage):
    """The AutoScript stage on an offset (non-compustage) mount.

    Each method is what the matching part of ``ThermoMicroscope`` does today:

    - ``read_position``: the ``stage_position`` branch of ``_get``, including setting the
      default coordinate system before every read;
    - ``read_homed`` / ``read_linked``: the ``stage_homed`` / ``stage_linked`` branches;
    - ``metadata_position``: ``_get_axis_limits``, which also says which axes exist;
    - ``_move_absolute``: ``move_stage_absolute``, with its working-distance restore
      and its axis restrictions;
    - ``_move_relative``: ``move_stage_relative``;
    - ``_home`` / ``_link``: the ``stage_home`` / ``stage_link`` branches of ``_set``.

    The old moves end by reading the position back; ``Stage.move_through`` does the
    same read, so a move here is the same calls end to end. ``_get_axis_limits`` gives
    r and t in degrees while positions are in radians; the metadata converts them, so
    limits and values share one unit.
    """

    compustage = False

    def __init__(self, parent: ThermoMicroscope, resources: Optional[Resources] = None):
        super().__init__(parent=parent, resources=resources)

    @property
    def _stage(self):
        """The vendor stage, looked up on each call as the old methods do."""
        return self.parent.stage

    def _to_autoscript(self, position: FibsemStagePosition):
        from fibsem.microscopes.autoscript import stage_position_to_autoscript

        return stage_position_to_autoscript(position, compustage=self.compustage)

    # -- position ----------------------------------------------------------------

    def read_position(self) -> FibsemStagePosition:
        from fibsem.microscopes.autoscript import stage_position_from_autoscript

        self._stage.set_default_coordinate_system(
            self.parent._default_stage_coordinate_system
        )
        return stage_position_from_autoscript(self._stage.current_position)

    def metadata_position(self) -> ParameterMetadata:
        return ParameterMetadata(
            limits=axis_limits_from_degrees(self.parent._get_axis_limits())
        )

    # -- homing and linking -----------------------------------------------------------

    def read_homed(self) -> bool:
        return self._stage.is_homed

    def read_linked(self) -> bool:
        return self._stage.is_linked

    # -- commands ---------------------------------------------------------------------

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.microscopes.autoscript import MoveSettings

        # get current working distance, to be restored later
        wd = self.parent.get_working_distance(BeamType.ELECTRON)

        autoscript_position = self._to_autoscript(position)

        if self.parent._axis_restrictions_apply(
            position
        ):  # ONLY when restrictions are on
            autoscript_position.z = None
            autoscript_position.r = None

        logging.info(f"Moving stage to {position}.")
        self._stage.absolute_move(
            autoscript_position, MoveSettings(rotate_compucentric=True)
        )

        # restore working distance to adjust for microscope compensation
        if not self.compustage:
            self.parent.set_working_distance(wd, BeamType.ELECTRON)

        logging.debug({"msg": "move_stage_absolute", "position": position.to_dict()})

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        logging.info(f"Moving stage by {delta}.")
        self._stage.relative_move(self._to_autoscript(delta))
        logging.debug({"msg": "move_stage_relative", "position": delta.to_dict()})

    def _home(self) -> None:
        logging.info("Homing stage...")
        self._stage.home()
        logging.info("Stage homed.")

    def _link(self) -> None:
        logging.info("Linking stage...")
        self._stage.link()
        logging.info("Stage linked.")


class AutoscriptCompustage(AutoscriptStage):
    """The AutoScript compustage: x, y, z and a tilt ``a`` in specimen coordinates.

    It differs from the offset stage in three places, each as the old code has it:
    positions convert to and from ``CompustagePosition`` (no r), an absolute move does
    not restore the working distance (it still reads it first, as today), and there
    is no linking: the old ``set("stage_link")`` logs and does nothing, so ``linked``
    is absent here and ``link()`` is unavailable. Its limits are the fixed compustage
    table ``_get_axis_limits`` returns, which has no r.
    """

    compustage = True

    def available_linked(self) -> bool:
        return False


def autoscript_stage_class(microscope: ThermoMicroscope) -> Type[AutoscriptStage]:
    """The driver class for the stage the Thermo backend found at connect."""
    return AutoscriptCompustage if microscope.stage_is_compustage else AutoscriptStage


def bind_autoscript_stage(
    microscope: ThermoMicroscope, resources: Optional[Resources] = None
) -> AutoscriptStage:
    """Build ``stage`` for a connected Thermo microscope."""
    return autoscript_stage_class(microscope)(microscope, resources).connect()

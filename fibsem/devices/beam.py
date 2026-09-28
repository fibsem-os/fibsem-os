"""The Beam device, and the key router that keeps today's get/set working.

One ``Beam`` class serves both columns; a parameter one column lacks is simply not
bound on it. ``KeyRouter`` is what ``FibsemMicroscope.get``/``set`` would
become: a key that has moved to a device is routed to its parameter, and every other
key falls through to the backend's untouched if/elif chain.
"""

from __future__ import annotations

import logging
from math import pi
from typing import Any, Dict, List, Mapping, Optional

from fibsem.devices.core import BoundParameter, Device, Parameter, command
from fibsem.devices.stage import Stage
from fibsem.structures import BeamType, FibsemImage, ImageSettings, Point, RangeLimit


class Beam(Device):
    voltage = Parameter(float, unit="V")
    current = Parameter(float, unit="A", depends_on=("plasma_gas",))
    plasma_gas = Parameter(str)
    working_distance = Parameter(float, unit="m")
    hfw = Parameter(float, unit="m")
    scan_rotation = Parameter(float, unit="rad", limits=RangeLimit(min=0.0, max=2 * pi))
    blanked = Parameter(bool)
    preset = Parameter(str)
    detector_type = Parameter(str)
    detector_mode = Parameter(str)
    detector_contrast = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))
    detector_brightness = Parameter(float, limits=RangeLimit(min=0.0, max=1.0))
    resolution = Parameter(tuple, unit="px", doc="(width, height)")
    dwell_time = Parameter(float, unit="s")
    stigmation = Parameter(Point)
    shift = Parameter(Point, unit="m", doc="Beam shift.")
    on = Parameter(bool, doc="The beam is switched on.")
    scanning_mode = Parameter(
        str, doc='"full_frame", "reduced_area" or "spot"; set by the scan commands.'
    )

    def __init__(self, beam_type: BeamType, parent: Any = None, **kwargs: Any):
        super().__init__(name=beam_type.name.lower(), parent=parent, **kwargs)
        self.beam_type = beam_type

    @command(available=lambda beam: "blanked" in beam.parameters)
    def blank(self) -> None:
        """Blank the beam."""
        self.blanked.set_value(True)

    @command(available=lambda beam: "blanked" in beam.parameters)
    def unblank(self) -> None:
        """Unblank the beam."""
        self.blanked.set_value(False)

    @command
    def acquire(self, image_settings: Optional[ImageSettings] = None) -> FibsemImage:
        """Acquire an image with this beam. Imaging is a beam command, not a device."""
        return self.parent.acquire_image(image_settings, beam_type=self.beam_type)


# Old key -> parameter name. Every beam key keeps its old name here, so the table is
# also the list of what has moved. A key missing from it has not moved yet.
BEAM_ROUTES: Dict[str, str] = {
    "voltage": "voltage",
    "current": "current",
    "plasma_gas": "plasma_gas",
    "working_distance": "working_distance",
    "hfw": "hfw",
    "scan_rotation": "scan_rotation",
    "blanked": "blanked",
    "preset": "preset",
    "detector_type": "detector_type",
    "detector_mode": "detector_mode",
    "detector_contrast": "detector_contrast",
    "detector_brightness": "detector_brightness",
    "resolution": "resolution",
    "dwell_time": "dwell_time",
    "stigmation": "stigmation",
    "shift": "shift",
    "on": "on",
    "scanning_mode": "scanning_mode",
}


# Stage keys take no beam type. A get key routes to a parameter; a set key that is a
# verb ("home", "link") routes to a command, which ignores the value as the old
# branches do.
STAGE_ROUTES: Dict[str, str] = {
    "stage_position": "position",
    "stage_homed": "homed",
    "stage_linked": "linked",
}
STAGE_COMMAND_ROUTES: Dict[str, str] = {
    "stage_home": "home",
    "stage_link": "link",
}


class KeyRouter:
    """Today's ``get``/``set``/``get_available_values``, routed where a key has moved.

    Routed calls make the same instrument call the old branch made and skip the new
    API's validation, so a half-migrated backend behaves exactly like an unmigrated one.
    Logging matches ``FibsemMicroscope.get`` and ``set``.
    """

    def __init__(
        self,
        microscope: Any,
        beams: Mapping[BeamType, Beam],
        routes: Optional[Mapping[str, str]] = None,
        stage: Optional[Stage] = None,
    ):
        self.microscope = microscope
        self.beams = dict(beams)
        self.routes = dict(BEAM_ROUTES if routes is None else routes)
        self.stage = stage

    def route(
        self, key: str, beam_type: Optional[BeamType]
    ) -> Optional[BoundParameter]:
        if self.stage is not None and key in STAGE_ROUTES:
            return self.stage.parameters.get(STAGE_ROUTES[key])
        name = self.routes.get(key)
        beam = self.beams.get(beam_type) if beam_type is not None else None
        if name is None or beam is None:
            return None
        return beam.parameters.get(name)

    def get(self, key: str, beam_type: Optional[BeamType] = None) -> Any:
        param = self.route(key, beam_type)
        if param is not None:
            value = param.get_value()
        else:
            value = self.microscope._get(key, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "get", "key": key, "beam_type": beam_name, "value": value}
        )
        return value

    def route_command(self, key: str) -> Optional[Any]:
        """The device command an old set key has become, if it has moved and is available."""
        name = STAGE_COMMAND_ROUTES.get(key)
        if self.stage is None or name is None:
            return None
        info = self.stage.commands.get(name)
        if info is None or not info.available:
            return None
        return getattr(self.stage, name)

    def set(self, key: str, value: Any, beam_type: Optional[BeamType] = None) -> None:
        param = self.route(key, beam_type)
        run = self.route_command(key)
        if run is not None:
            run()
        elif param is not None and param.writable:
            param.write_through(value)
        else:
            # Unmoved keys, and read-only ones (set("stage_position") warns there).
            self.microscope._set(key, value, beam_type)
        beam_name = "None" if beam_type is None else beam_type.name
        logging.debug(
            {"msg": "set", "key": key, "beam_type": beam_name, "value": value}
        )

    def get_available_values(
        self, key: str, beam_type: Optional[BeamType] = None
    ) -> List[Any]:
        param = self.route(key, beam_type)
        if param is not None and param.choices is not None:
            return list(param.choices)
        return self.microscope.get_available_values(key, beam_type)

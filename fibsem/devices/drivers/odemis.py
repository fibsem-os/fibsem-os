"""The beams, the stage and the chamber of a Thermo microscope driven through odemis,
as devices.

``OdemisBeam``, ``OdemisStage`` and ``OdemisChamber`` are ``OdemisThermoMicroscope``'s
beam, stage and chamber keys
moved as they are, so the old call and the device make the same odemis calls in the
same order and log the same messages. ``OdemisThermoMicroscope`` builds them when it
is created and routes its keys and moves to them; its old branches are deleted, and
``tests/fixtures/odemis_device_calls.json`` keeps what they did.

The odemis client (``microscope.connection``, the ``fibsem`` component) takes the
channel on every call, so nothing here selects an imaging channel first. The vendor
stage is ``microscope._vendor_stage`` (the ``stage-bare`` component);
``microscope.stage`` is the stage device. This module imports odemis only through
``fibsem.microscopes.odemis_microscope``, inside the methods, so it loads where odemis
is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import ParameterMetadata, Resources
from fibsem.devices.stage import Stage, axis_limits_from_degrees
from fibsem.structures import (
    BeamType,
    ChamberState,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)

if TYPE_CHECKING:
    from fibsem.microscopes.odemis_microscope import OdemisThermoMicroscope
    from fibsem.microscopes.registry import BuildContext
    from fibsem.structures import DeviceEntry

# The odemis client's name for each column (``beam_type_to_odemis``).
ODEMIS_CHANNELS: Dict[BeamType, str] = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}


class OdemisBeam(Beam):
    """A column of a Thermo microscope driven through odemis.

    Each parameter is the matching branch of ``OdemisThermoMicroscope._get``/``_set``
    moved as it is; the choices are its ``get_available_values``'s. The detector
    writes check as the old branches did, against the same choices.

    The scan commands are the old ``spot_mode`` and ``full_frame`` keys, and
    ``reduced_area`` is the client call ``acquire_image`` and ``autocontrast`` make;
    the old branches had no ``reduced_area`` key, so that set warned and did nothing.
    The client has no read of the scan mode, so ``scanning_mode`` is absent and the
    commands read nothing back.

    Not here, so absent on the new API and answered by ``_get``/``_set``:
    ``plasma_gas`` (the old branch raises on a plasma column) and ``preset`` (there is
    none).
    """

    def __init__(
        self,
        beam_type: BeamType,
        parent: OdemisThermoMicroscope,
        resources: Optional[Resources] = None,
    ):
        super().__init__(beam_type, parent=parent, resources=resources)
        self.channel = ODEMIS_CHANNELS[beam_type]

    @property
    def _client(self) -> Any:
        """The odemis client, looked up on every call, as the old branches did."""
        return self.parent.connection

    def _set_log(self, what: str, value: Any, unit: str) -> None:
        logging.info(f"{self.beam_type.name} {what} set to {value}{unit}.")

    def read_on(self) -> bool:
        return self._client.get_beam_is_on(self.channel)

    def write_on(self, value: bool) -> None:
        self._client.set_beam_power(value, self.channel)
        logging.info(f"{self.beam_type.name} beam turned {'on' if value else 'off'}.")

    def read_blanked(self) -> bool:
        return self._client.beam_is_blanked(self.channel)

    def write_blanked(self, value: bool) -> None:
        client = self._client
        client.blank_beam(self.channel) if value else client.unblank_beam(self.channel)
        logging.info(
            f"{self.beam_type.name} beam {'blanked' if value else 'unblanked'}."
        )

    def read_working_distance(self) -> float:
        return self._client.get_working_distance(self.channel)

    def write_working_distance(self, value: float) -> None:
        self._client.set_working_distance(value, self.channel)
        self._set_log("working distance", value, " m")

    def read_current(self) -> float:
        return self._client.get_beam_current(self.channel)

    def write_current(self, value: float) -> None:
        self._client.set_beam_current(value, self.channel)
        self._set_log("current", value, " A")

    def metadata_current(self) -> ParameterMetadata:
        # The client gives the ion beam's currents as choices and the electron beam's
        # as a range, which doubles from the minimum, as the microscope lists them.
        info = self._client.beam_current_info(self.channel)
        if "choices" in info:
            return ParameterMetadata(choices=list(info["choices"]))
        low, high = info["range"]
        choices, current = [], low
        while current <= high:
            choices.append(current)
            current *= 2.0
        return ParameterMetadata(choices=choices)

    def read_voltage(self) -> float:
        return self._client.get_high_voltage(self.channel)

    def write_voltage(self, value: float) -> None:
        self._client.set_high_voltage(value, self.channel)
        self._set_log("voltage", value, " V")

    def metadata_voltage(self) -> ParameterMetadata:
        from fibsem.microscopes.autoscript import THERMO_VOLTAGE_CHOICES

        low, high = self._client.high_voltage_info(self.channel)["range"]
        return ParameterMetadata(
            choices=[
                v for v in THERMO_VOLTAGE_CHOICES[self.beam_type] if low <= v <= high
            ]
        )

    def read_hfw(self) -> float:
        return self._client.get_field_of_view(self.channel)

    def write_hfw(self, value: float) -> None:
        self._client.set_field_of_view(value, self.channel)
        self._set_log("HFW", value, " m")

    def read_dwell_time(self) -> float:
        return self._client.get_dwell_time(self.channel)

    def write_dwell_time(self, value: float) -> None:
        self._client.set_dwell_time(value, self.channel)
        self._set_log("dwell time", value, " s")

    def read_scan_rotation(self) -> float:
        return self._client.get_scan_rotation(self.channel)

    def write_scan_rotation(self, value: float) -> None:
        self._client.set_scan_rotation(value, self.channel)
        self._set_log("scan rotation", value, " radians")

    def read_shift(self) -> Point:
        shift = self._client.get_beam_shift(self.channel)
        return Point(shift[0], shift[1])

    def write_shift(self, value: Point) -> None:
        self._client.set_beam_shift(value.x, value.y, self.channel)
        self._set_log("shift", value, "")

    def read_stigmation(self) -> Point:
        stigmation = self._client.get_stigmator(self.channel)
        return Point(stigmation[0], stigmation[1])

    def write_stigmation(self, value: Point) -> None:
        self._client.set_stigmator(value.x, value.y, self.channel)
        self._set_log("stigmation", value, "")

    def read_resolution(self) -> List[int]:
        # a list, as the old get returns it
        width, height = self._client.get_resolution(self.channel)
        return [width, height]

    def write_resolution(self, value: Tuple[int, int]) -> None:
        self._client.set_resolution(value, self.channel)

    # The detector. A type or mode not in the choices warns and is not set, and a
    # brightness or contrast outside (0, 1] likewise, as the old branches did.

    def _detector_choices(self, what: str) -> List[str]:
        return getattr(self._client, f"detector_{what}_info")(self.channel)["choices"]

    def read_detector_type(self) -> str:
        return self._client.get_detector_type(self.channel)

    def write_detector_type(self, value: str) -> None:
        if value in self._detector_choices("type"):
            self._client.set_detector_type(value, self.channel)
            logging.info(f"Detector type set to {value}.")
        else:
            logging.warning(f"Detector type {value} not available.")

    def metadata_detector_type(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._detector_choices("type")))

    # No choices for the mode: they are the detector type's, which can change.
    def read_detector_mode(self) -> str:
        return self._client.get_detector_mode(self.channel)

    def write_detector_mode(self, value: str) -> None:
        if value in self._detector_choices("mode"):
            self._client.set_detector_mode(value, self.channel)
            logging.info(f"Detector mode set to {value}.")
        else:
            logging.warning(f"Detector mode {value} not available.")

    def read_detector_brightness(self) -> float:
        return self._client.get_brightness(self.channel)

    def write_detector_brightness(self, value: float) -> None:
        if 0 < value <= 1:
            self._client.set_brightness(value, self.channel)
            logging.info(f"Detector brightness set to {value}.")
        else:
            logging.warning(
                f"Detector brightness {value} not available, must be between 0 and 1."
            )

    def read_detector_contrast(self) -> float:
        return self._client.get_contrast(self.channel)

    def write_detector_contrast(self, value: float) -> None:
        if 0 < value <= 1:
            self._client.set_contrast(value, self.channel)
            logging.info(f"Detector contrast set to {value}.")
        else:
            logging.warning(
                f"Detector contrast {value} not available, mut be between 0 and 1."
            )

    # Imaging and autocontrast, moved from ``OdemisThermoMicroscope`` as they are; the
    # FibsemImage is built by the microscope's ``_construct_image``, as before. No
    # ``_auto_focus``: the client has no focus call, and the working-distance sweep
    # stays ``microscope.auto_focus``'s. No ``_live``: Odemis has no live view.

    def _acquire(self, image_settings: Optional[ImageSettings]) -> FibsemImage:
        microscope, client, channel = self.parent, self._client, self.channel
        if image_settings is None:
            # acquire_image(beam_type=...): the beam's current settings
            image, _md = client.acquire_image(channel=channel, frame_settings=None)
            return microscope._construct_image(
                image, microscope._current_image_settings(self.beam_type, image)
            )

        if image_settings.reduced_area is not None:
            area = image_settings.reduced_area
            client.set_reduced_area_scan_mode(
                channel=channel,
                left=area.left,
                top=area.top,
                width=area.width,
                height=area.height,
            )
        else:
            client.set_full_frame_scan_mode(channel=channel)

        # A square resolution can't be set, but a frame can be grabbed at one: set
        # the current resolution and ask for the square in the frame settings.
        frame_settings = None
        tmp_resolution = None
        resolution = image_settings.resolution
        if resolution[0] == resolution[1]:
            frame_settings = {"resolution": f"{resolution[0]}x{resolution[1]}"}
            tmp_resolution = resolution
            image_settings.resolution = microscope.get_resolution(
                beam_type=self.beam_type
            )
        microscope.set_imaging_settings(image_settings)

        image, _md = client.acquire_image(
            channel=channel, frame_settings=frame_settings
        )

        if image_settings.reduced_area is not None:
            client.set_full_frame_scan_mode(channel=channel)
        if tmp_resolution is not None:
            image_settings.resolution = tmp_resolution
        microscope._last_imaging_settings = image_settings
        return microscope._construct_image(image, image_settings)

    def _last_image(self) -> FibsemImage:
        microscope = self.parent
        image = self._client.get_last_image(channel=self.channel)
        # The client is annotated as returning (image, metadata), but the AutoScript
        # adapter (1.16.0) returns the bare array.
        if isinstance(image, tuple):
            image = image[0]
        return microscope._construct_image(
            image, microscope._current_image_settings(self.beam_type, image)
        )

    def _autocontrast(self, reduced_area: Optional[FibsemRectangle]) -> None:
        client, channel = self._client, self.channel
        if reduced_area is not None:
            client.set_reduced_area_scan_mode(channel, **reduced_area.to_dict())
        client.run_auto_contrast_brightness(channel=channel)
        if reduced_area is not None:
            client.set_full_frame_scan_mode(channel)

    # -- the scan area, with no read of the scan mode -------------------------------

    def _scans(self) -> bool:
        return True

    def _spot(self, point: Point) -> None:
        self._client.set_spot_scan_mode(channel=self.channel, x=point.x, y=point.y)

    def _reduced_area(self, area: FibsemRectangle) -> None:
        self._client.set_reduced_area_scan_mode(
            channel=self.channel,
            left=area.left,
            top=area.top,
            width=area.width,
            height=area.height,
        )

    def _full_frame(self) -> None:
        self._client.set_full_frame_scan_mode(self.channel)


def bind_odemis_beams(
    microscope: OdemisThermoMicroscope, resources: Optional[Resources] = None
) -> Dict[BeamType, OdemisBeam]:
    """Build ``beams[BeamType]`` for an Odemis microscope: one per enabled column, so
    a disabled one is never touched."""
    enabled = {
        BeamType.ELECTRON: microscope.system.electron.enabled,
        BeamType.ION: microscope.system.ion.enabled,
    }
    return {
        beam_type: OdemisBeam(beam_type, microscope, resources).connect()
        for beam_type, on in enabled.items()
        if on
    }


class OdemisStage(Stage):
    """The stage of a Thermo microscope driven through odemis.

    Each method is what the matching part of ``OdemisThermoMicroscope`` does today:

    - ``read_position``: the ``stage_position`` branch of ``_get``, the ``stage-bare``
      component's position;
    - ``read_homed`` / ``read_linked``: the ``stage_homed`` / ``stage_linked``
      branches, which ask the client;
    - ``metadata_position``: ``_get_axis_limits``, the base class's fixed table, in
      degrees for r and t, converted so limits and positions share one unit;
    - ``_move_absolute`` / ``_move_relative``: ``move_stage_absolute`` /
      ``move_stage_relative``, each waiting on the move's future;
    - ``_home`` / ``_link``: the ``stage_home`` / ``stage_link`` branches of ``_set``.

    The old moves end by reading the position back; ``Stage.move_through`` does the
    same read.
    """

    def __init__(
        self, parent: OdemisThermoMicroscope, resources: Optional[Resources] = None
    ):
        super().__init__(parent=parent, resources=resources)

    @property
    def _stage(self) -> Any:
        """The vendor stage, looked up on each call as the old methods do."""
        return self.parent._vendor_stage

    def read_position(self) -> FibsemStagePosition:
        return FibsemStagePosition.from_odemis_dict(self._stage.position.value)

    def metadata_position(self) -> ParameterMetadata:
        return ParameterMetadata(
            limits=axis_limits_from_degrees(self.parent._get_axis_limits())
        )

    def read_homed(self) -> bool:
        return self.parent.connection.is_homed()

    def read_linked(self) -> bool:
        return self.parent.connection.is_linked()

    def _move_absolute(self, position: FibsemStagePosition) -> None:
        from fibsem.microscopes.odemis_microscope import stage_position_to_odemis_dict

        self._stage.moveAbs(stage_position_to_odemis_dict(position)).result()

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        from fibsem.microscopes.odemis_microscope import stage_position_to_odemis_dict

        self._stage.moveRel(stage_position_to_odemis_dict(delta)).result()

    def _home(self) -> None:
        logging.info("Homing stage...")
        self.parent.connection.home_stage()
        logging.info("Stage homed.")

    def _link(self) -> None:
        logging.info("Linking stage...")
        self.parent.connection.link(True)
        logging.info("Stage linked.")


def bind_odemis_stage(
    microscope: OdemisThermoMicroscope, resources: Optional[Resources] = None
) -> OdemisStage:
    """Build ``stage`` for an Odemis microscope."""
    return OdemisStage(microscope, resources).connect()


class OdemisChamber(Chamber):
    """The vacuum of a Thermo microscope driven through odemis.

    ``read_state``/``read_pressure`` are the ``chamber_state``/``chamber_pressure``
    branches of ``OdemisThermoMicroscope._get``, and ``_pump``/``_vent`` the
    ``pump_chamber``/``vent_chamber`` branches of ``_set``. The client names the state
    as AutoScript does ("Pumped") or as xT does ("vacuum"); both read through
    ``ODEMIS_CHAMBER_STATES`` as before, then as a ``ChamberState``.
    """

    def __init__(
        self, parent: OdemisThermoMicroscope, resources: Optional[Resources] = None
    ):
        super().__init__(parent=parent, resources=resources)

    @property
    def _client(self) -> Any:
        return self.parent.connection

    def read_state(self) -> ChamberState:
        from fibsem.microscopes.odemis_microscope import ODEMIS_CHAMBER_STATES

        state = self._client.get_chamber_state()
        return ChamberState.from_name(
            ODEMIS_CHAMBER_STATES.get(str(state).lower(), state)
        )

    def read_pressure(self) -> float:
        return self._client.get_pressure()

    def _pump(self) -> None:
        logging.info("Pumping chamber...")
        self._client.pump()
        logging.info("Chamber pumped.")

    def _vent(self) -> None:
        logging.info("Venting chamber...")
        self._client.vent()
        logging.info("Chamber vented.")


def bind_odemis_chamber(
    microscope: OdemisThermoMicroscope, resources: Optional[Resources] = None
) -> OdemisChamber:
    """Build ``chamber`` for an Odemis microscope."""
    return OdemisChamber(microscope, resources).connect()


# The builders the Odemis driver record names, one per device type: how
# ``OdemisThermoMicroscope`` builds each device from its entry
# (``fibsem.devices.entries``).


def build_odemis_beam(entry: DeviceEntry, context: BuildContext) -> OdemisBeam:
    """The column an ``electron`` or ``ion`` entry names."""
    if entry.name not in ("electron", "ion"):
        raise ValueError("an Odemis beam is named 'electron' or 'ion'")
    beam_type = BeamType.ELECTRON if entry.name == "electron" else BeamType.ION
    return OdemisBeam(beam_type, context.microscope).connect()


def build_odemis_stage(entry: DeviceEntry, context: BuildContext) -> OdemisStage:
    stage = OdemisStage(context.microscope)
    stage.name = entry.name
    return stage.connect()


def build_odemis_chamber(entry: DeviceEntry, context: BuildContext) -> OdemisChamber:
    chamber = OdemisChamber(context.microscope)
    chamber.name = entry.name
    return chamber.connect()

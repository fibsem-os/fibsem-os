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
``fibsem.drivers.odemis.microscope``, inside the methods, so it loads where odemis
is not installed.

The FM.

The METEOR's FM through odemis, as devices.

``OdemisFMCamera``, ``OdemisFMLightSource``, ``OdemisFMFilterSet``,
``OdemisFMObjective`` and the ``OdemisFM`` group are the old
``OdemisFluorescenceMicroscope``'s parts moved onto the FM devices: each read, write
and command makes the odemis calls the old property or method made, in the same order,
on the same components and the same ``FluoStream``
(``tests/fixtures/odemis/old_fm_pins.json`` holds those calls). ``OdemisThermoMicroscope.fm`` is the FM API over
them (``DeviceOdemisFluorescenceMicroscope``), and the device server serves them from the
METEOR PC.

The stream stays. Excitation, emission and power are the stream's settings, which odemis
applies to the light and the filter wheel only while the stream runs, so they are
"settings for the next exposure" (see ``FilterSet``). Driving the components directly
waits until the METEOR works.

Live view keeps what the old live view does, with each frame pulled rather than pushed:
`OdemisFM.start_live` activates the stream (light on, filters set, the camera
streaming), each ``acquire_frame`` while live is the camera's next frame
(``data.get(asap=False)``, as ``acquire_image`` takes it while the stream runs), and
stopping deactivates the stream (light off).

The FM parts need odemis, through ``fibsem.fm.odemis``, which each imports in the
method that uses it, so the module still loads where odemis is not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from fibsem.devices.beam import Beam
from fibsem.devices.chamber import Chamber
from fibsem.devices.core import Device, ParameterMetadata, Resources, resources_of
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.stage import Stage, axis_limits_from_degrees
from fibsem.devices.wire import Frame, to_wire
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    band_name,
    objective_device_state,
)
from fibsem.structures import (
    BeamType,
    ChamberState,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    InsertableDeviceState,
    Point,
    RangeLimit,
    Resolution,
)

# The voltages Odemis offers on a ThermoFisher column, before the column's own range
# narrows them. The same list as the AutoScript driver's: Odemis keeps its own copy so
# the drivers do not import each other.
ODEMIS_VOLTAGE_CHOICES = {
    BeamType.ELECTRON: (1000, 2000, 3000, 5000, 10000, 20000, 30000),
    BeamType.ION: (500, 1000, 2000, 8000, 16000, 30000),
}

if TYPE_CHECKING:
    from fibsem.drivers.odemis.microscope import OdemisThermoMicroscope
    from fibsem.drivers.registry import BuildContext
    from fibsem.structures import DeviceEntry

# The odemis client's name for each column (``beam_type_to_odemis``).
ODEMIS_CHANNELS: Dict[BeamType, str] = {
    BeamType.ELECTRON: "electron",
    BeamType.ION: "ion",
}


class OdemisBeam(Beam):
    """A column of a Thermo microscope driven through odemis.

    Each parameter is the matching branch of the old ``OdemisThermoMicroscope._get``/
    ``_set`` (now removed) moved as it is; the choices are what its old
    ``get_available_values`` answered. The detector writes check as the old branches
    did, against the same choices.

    The scan commands are the old ``spot_mode`` and ``full_frame`` keys, and
    ``reduced_area`` is the client call ``acquire_image`` and ``autocontrast`` make;
    the old branches had no ``reduced_area`` key, so that set warned and did nothing.
    The client has no read of the scan mode, so ``scanning_mode`` is absent and the
    commands read nothing back.

    Not here, so absent (a read is None, a write does nothing): ``plasma_gas`` (the
    old branch raised on a plasma column) and ``preset`` (there is none).
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
        logging.debug(
            f"{self.beam_type.name} {what} set to {value}{unit}.",
            stacklevel=2,  # name the write_* method, not this helper
        )

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
        logging.debug(
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
        low, high = self._client.high_voltage_info(self.channel)["range"]
        return ParameterMetadata(
            choices=[
                v for v in ODEMIS_VOLTAGE_CHOICES[self.beam_type] if low <= v <= high
            ]
        )

    def read_hfw(self) -> float:
        return self._client.get_field_of_view(self.channel)

    def write_hfw(self, value: float) -> None:
        self._client.set_field_of_view(value, self.channel)
        self._set_log("HFW", value, " m")

    def metadata_hfw(self) -> ParameterMetadata:
        return self._range_metadata(self._client.field_of_view_info(self.channel))

    def read_dwell_time(self) -> float:
        return self._client.get_dwell_time(self.channel)

    def write_dwell_time(self, value: float) -> None:
        self._client.set_dwell_time(value, self.channel)
        self._set_log("dwell time", value, " s")

    def metadata_dwell_time(self) -> ParameterMetadata:
        return self._range_metadata(self._client.dwell_time_info(self.channel))

    @staticmethod
    def _range_metadata(info: dict) -> ParameterMetadata:
        """Limits from a client ``*_info`` answer, ``{"unit": ..., "range": (min, max)}``."""
        low, high = info["range"]
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

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
            logging.debug(f"Detector type set to {value}.")
        else:
            logging.warning(f"Detector type {value} not available.")

    def metadata_detector_type(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._detector_choices("type")))

    def read_detector_mode(self) -> str:
        return self._client.get_detector_mode(self.channel)

    def write_detector_mode(self, value: str) -> None:
        if value in self._detector_choices("mode"):
            self._client.set_detector_mode(value, self.channel)
            logging.debug(f"Detector mode set to {value}.")
        else:
            logging.warning(f"Detector mode {value} not available.")

    def metadata_detector_mode(self) -> ParameterMetadata:
        # the detector type's modes, read again when the type changes
        return ParameterMetadata(choices=list(self._detector_choices("mode")))

    def read_detector_brightness(self) -> float:
        return self._client.get_brightness(self.channel)

    def write_detector_brightness(self, value: float) -> None:
        if 0 < value <= 1:
            self._client.set_brightness(value, self.channel)
            logging.debug(f"Detector brightness set to {value}.")
        else:
            logging.warning(
                f"Detector brightness {value} not available, must be between 0 and 1."
            )

    def read_detector_contrast(self) -> float:
        return self._client.get_contrast(self.channel)

    def write_detector_contrast(self, value: float) -> None:
        if 0 < value <= 1:
            self._client.set_contrast(value, self.channel)
            logging.debug(f"Detector contrast set to {value}.")
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
            frame_settings = {"resolution": str(Resolution(*resolution))}
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
        from fibsem.drivers.odemis.microscope import stage_position_to_odemis_dict

        self._stage.moveAbs(stage_position_to_odemis_dict(position)).result()

    def _move_relative(self, delta: FibsemStagePosition) -> None:
        from fibsem.drivers.odemis.microscope import stage_position_to_odemis_dict

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
        from fibsem.drivers.odemis.microscope import ODEMIS_CHAMBER_STATES

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


# The FM.


def emission_filter_of(choice: Any) -> EmissionFilter:
    """The filter an odemis emission choice is: pass-through is reflection; a band, or
    a multi-band filter's bands, in nm."""
    from fibsem.fm.odemis import _odemis_bands_nm, model

    if isinstance(choice, str) and choice == model.BAND_PASS_THROUGH:
        return REFLECTION
    bands = _odemis_bands_nm(choice)
    return EmissionFilter(band_name(bands), bands=bands)


class OdemisFMCamera(Camera):
    """The camera: the ``ccd`` component, with acquisitions through the stream."""

    def __init__(
        self,
        camera: Any,
        stream: Any,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        from fibsem.fm.odemis import model

        super().__init__(name="camera", parent=parent, resources=resources)
        self._camera = camera
        self._stream = stream
        camera_md = self._camera.getMetadata()
        # fallback values only: pixel_size/resolution are read live
        self._resolution = tuple(self._camera.resolution.value)

        pixel_size = camera_md.get(model.MD_PIXEL_SIZE)
        if pixel_size is None:
            pixel_size = self._camera.pixelSize.value  # sensor pixel size
            logging.warning(
                "Camera metadata has no MD_PIXEL_SIZE; falling back to the "
                f"sensor pixel size {pixel_size} - image scale will not "
                "account for magnification."
            )
        self._pixel_size = tuple(pixel_size)
        # not all cameras publish a baseline (only some drivers set MD_BASELINE)
        self._offset = camera_md.get(model.MD_BASELINE, 0)

    def read_exposure_time(self) -> float:
        return self._camera.exposureTime.value

    def write_exposure_time(self, value: float) -> None:
        self._camera.exposureTime.value = value

    def metadata_exposure_time(self) -> ParameterMetadata:
        rng = self._camera.exposureTime.range
        return ParameterMetadata(limits=RangeLimit(min=rng[0], max=rng[1]))

    def read_binning(self) -> int:
        return self._camera.binning.value[0]

    def write_binning(self, value: int) -> None:
        # Checked against the camera's binnings as they are now, as the old setter.
        if value not in self._binnings():
            raise ValueError(f"Binning must be one of {self._binnings()}, got {value}")
        self._camera.binning.value = (value, value)

    def metadata_binning(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._binnings()))

    def _binnings(self) -> Tuple[int, ...]:
        """The VA's choices when enumerated; otherwise powers of two within its range."""
        binning_va = self._camera.binning
        choices = getattr(binning_va, "choices", None)
        if choices:
            return tuple(sorted({int(c[0]) for c in choices}))
        min_binning = binning_va.range[0][0]
        max_binning = binning_va.range[1][0]
        binnings = []
        b = 1
        while b <= max_binning:
            if b >= min_binning:
                binnings.append(b)
            b *= 2
        return tuple(binnings)

    # Gain is a fraction of the camera's maximum, as power is of the light's. A camera
    # without a gain control, or whose gain gives no range or choices to scale by, has
    # no gain parameter; the old class reads it as None and ignores a write.
    def available_gain(self) -> bool:
        from fibsem.fm.odemis import model

        return model.hasVA(self._camera, "gain") and self._max_gain() is not None

    def _max_gain(self) -> Optional[float]:
        """The top of the gain VA's range or choices; None when it gives neither."""
        va = self._camera.gain
        top = va.range[1] if getattr(va, "range", None) else None
        if top is None and getattr(va, "choices", None):
            top = max(va.choices)
        return float(top) if top is not None and top > 0 else None

    def read_gain(self) -> float:
        return self._camera.gain.value / self._max_gain()

    def write_gain(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            logging.warning(f"Gain fraction {value} outside [0, 1], clipping.")
            value = min(max(value, 0.0), 1.0)
        raw = value * self._max_gain()
        choices = getattr(self._camera.gain, "choices", None)
        if choices:
            raw = min(choices, key=lambda c: abs(c - raw))
        self._camera.gain.value = raw

    def metadata_gain(self) -> ParameterMetadata:
        return ParameterMetadata(
            limits=RangeLimit(min=0.0, max=1.0),
            native_max=self._max_gain(),
            native_unit=getattr(self._camera.gain, "unit", None),
        )

    def read_offset(self) -> float:
        """The camera's MD_BASELINE, read at connect. Read-only, as on the old class."""
        return self._offset

    def read_pixel_size(self) -> tuple:
        """Sample-plane, from the camera metadata: odemis republishes it on binning and
        magnification changes, so no binning scaling here."""
        from fibsem.fm.odemis import model

        try:
            pixel_size = self._camera.getMetadata().get(model.MD_PIXEL_SIZE)
        except Exception as e:
            logging.warning(
                f"Failed to read camera metadata, using cached pixel size: {e}"
            )
            return tuple(self._pixel_size)
        if pixel_size is None:
            return tuple(self._pixel_size)
        return tuple(pixel_size)

    def read_resolution(self) -> tuple:
        return tuple(self._camera.resolution.value)

    def _acquire(self) -> np.ndarray:
        from fibsem.fm.odemis import acquire

        if self._stream.is_active.value:
            # asap=False guarantees the frame is acquired after this call
            return self._camera.data.get(asap=False)

        da: List[Any]
        f = acquire([self._stream])
        da, err = f.result()
        if err:
            raise RuntimeError(f"Error acquiring image: {err}")
        if not da:
            raise RuntimeError("Acquisition returned no data.")
        return da[0]


class OdemisFMLightSource(LightSource):
    """The light source: the stream's power, as a fraction of its maximum."""

    def __init__(
        self, stream: Any, parent: Any = None, resources: Optional[Resources] = None
    ):
        super().__init__(name="light_source", parent=parent, resources=resources)
        self._stream = stream

    def read_power(self) -> float:
        max_power = self._stream.power.range[1]
        if max_power <= 0:
            return 0.0
        return self._stream.power.value / max_power

    def write_power(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            logging.warning(f"Power fraction {value} outside [0, 1], clipping.")
            value = min(max(value, 0.0), 1.0)
        max_power = self._stream.power.range[1]
        self._stream.power.value = value * max_power

    def metadata_power(self) -> ParameterMetadata:
        power = self._stream.power
        return ParameterMetadata(
            limits=RangeLimit(min=0.0, max=1.0),
            native_max=power.range[1],
            native_unit=getattr(power, "unit", None),
        )


class OdemisFMFilterSet(FilterSet):
    """The filter set: the stream's excitation and emission bands."""

    def __init__(
        self, stream: Any, parent: Any = None, resources: Optional[Resources] = None
    ):
        super().__init__(name="filter_set", parent=parent, resources=resources)
        self._stream = stream

    def _excitation_choices_by_nm(self) -> Dict[float, tuple]:
        """Centre wavelength (nm) -> excitation choice, in one pass over the choices
        (an unordered set), so selection and assignment use the same choice."""
        return {c[2] * 1e9: c for c in self._stream.excitation.choices}

    def read_excitation_wavelength(self) -> float:
        return self._stream.excitation.value[2] * 1e9

    def write_excitation_wavelength(self, value: float) -> None:
        choices_by_nm = self._excitation_choices_by_nm()
        if not choices_by_nm:
            raise ValueError("No excitation wavelengths available.")
        closest_nm = min(choices_by_nm, key=lambda nm: abs(nm - value))
        choice = choices_by_nm[closest_nm]
        logging.info(
            f"Setting excitation wavelength to {closest_nm:.0f} nm "
            f"(requested: {value} nm, band: {choice})"
        )
        self._stream.excitation.value = choice

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._excitation_choices_by_nm()))

    def _emission_choices(self) -> List[Tuple[EmissionFilter, Any]]:
        return [(emission_filter_of(c), c) for c in self._stream.emission.choices]

    def read_emission_filter(self) -> EmissionFilter:
        return emission_filter_of(self._stream.emission.value)

    def write_emission_filter(self, value: EmissionFilter) -> None:
        choice = next((c for f, c in self._emission_choices() if f == value), None)
        if choice is None:
            raise ValueError(f"filter_set has no emission filter {value}")
        if value == REFLECTION:
            self._stream.emission.value = choice
            return
        if self._stream.emission.value == choice:
            return  # setting the emission filter is slow, skip if unchanged
        logging.info(f"Setting emission filter to {value.name} (band: {choice})")
        self._stream.emission.value = choice

    def metadata_emission_filter(self) -> ParameterMetadata:
        return ParameterMetadata(choices=[f for f, _ in self._emission_choices()])

    def select_fluorescence(self) -> None:
        """The band odemis matches to the current excitation: what the old class does
        for a channel that names its emission "Fluorescence" (TFS-style)."""
        from fibsem.fm.odemis import fluo, model

        bands = {
            c
            for c in self._stream.emission.choices
            if not (isinstance(c, str) and c == model.BAND_PASS_THROUGH)
        }
        if not bands:
            raise ValueError("No emission bands available for fluorescence mode.")
        choice = fluo.get_one_band_em(bands, self._stream.excitation.value)
        if self._stream.emission.value != choice:
            logging.info(
                f"Mapping emission 'Fluorescence' to band {choice} for the "
                f"current excitation"
            )
            self._stream.emission.value = choice
        # Through the parameter's own read, so the change is cached and signalled.
        self.emission_filter.get_value()


class OdemisFMObjective(Objective):
    """The objective: the ``focus`` actuator and the ``lens`` component.

    The favourite (inserted and retracted) positions are read from the focuser's
    metadata at connect and kept, as the old class caches them.
    """

    def __init__(
        self,
        focuser: Any,
        lens: Any,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        from fibsem.fm.odemis import model

        super().__init__(name="objective", parent=parent, resources=resources)
        self._focuser = focuser
        self._lens = lens
        self._metadata_cache: Dict[str, Any] = {}
        try:
            metadata = self._focuser.getMetadata()
            self._metadata_cache = {
                "active_position": metadata[model.MD_FAV_POS_ACTIVE],
                "deactive_position": metadata[model.MD_FAV_POS_DEACTIVE],
            }
        except Exception as e:
            logging.warning(f"Failed to cache metadata: {e}")

        # Where the objective focuses, for the FM API: the inserted position.
        active_position = self.active_position
        if active_position is not None:
            self.focus_position: float = active_position["z"]
        else:
            self.focus_position = self._focuser.position.value["z"]
            logging.warning(
                "Focuser has no active-position metadata; using the current "
                f"position ({self.focus_position * 1e3:.3f} mm) as the focus position."
            )
        # default user limit: no restriction beyond the hardware range
        self._limit_position = self._limits()[1]

    def _favourite(self, key: str, md_key: str, label: str) -> Optional[dict]:
        try:
            return self._metadata_cache[key]
        except (KeyError, TypeError):
            try:
                return self._focuser.getMetadata()[md_key]
            except Exception as e:
                logging.error(f"Failed to get {label} position: {e}")
                return None

    @property
    def active_position(self) -> Optional[dict]:
        """The inserted position, ``{"z": metres}``, or None when odemis has none."""
        from fibsem.fm.odemis import model

        return self._favourite("active_position", model.MD_FAV_POS_ACTIVE, "active")

    @property
    def deactive_position(self) -> Optional[dict]:
        """The retracted position, ``{"z": metres}``, or None when odemis has none."""
        from fibsem.fm.odemis import model

        return self._favourite(
            "deactive_position", model.MD_FAV_POS_DEACTIVE, "deactive"
        )

    def _limits(self) -> Tuple[float, float]:
        rng = self._focuser.axes["z"].range
        return (rng[0], rng[1])

    def read_position(self) -> float:
        return self._focuser.position.value["z"]

    def metadata_position(self) -> ParameterMetadata:
        low, high = self._limits()
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_state(self) -> InsertableDeviceState:
        from fibsem.fm.odemis import OBJECTIVE_POSITION_ATOL

        try:
            position = self._focuser.position.value["z"]
            active_position = self.active_position
            deactive_position = self.deactive_position
            if active_position is None or deactive_position is None:
                return objective_device_state("Other")
            if position > active_position["z"] - OBJECTIVE_POSITION_ATOL:
                return objective_device_state("Inserted")
            if position < deactive_position["z"] + OBJECTIVE_POSITION_ATOL:
                return objective_device_state("Retracted")
            return objective_device_state("Other")
        except Exception as e:
            logging.error(f"Failed to get objective lens state: {e}")
            return objective_device_state("Error")

    def read_magnification(self) -> float:
        return self._lens.magnification.value

    def write_magnification(self, value: float) -> None:
        self._lens.magnification.value = value

    def read_numerical_aperture(self) -> float:
        return self._lens.numericalAperture.value

    def write_numerical_aperture(self, value: float) -> None:
        self._lens.numericalAperture.value = value

    def read_limit_position(self) -> float:
        return self._limit_position

    def write_limit_position(self, value: float) -> None:
        self._limit_position = value
        logging.info(
            f"Objective user-defined position limit set to: "
            f"{self._limit_position * 1e3:.3f} mm"
        )

    def _move_absolute(self, position: float) -> None:
        # clip to the user-defined safety limit, then check the focuser's range
        # (odemis raises ValueError for an out-of-range move)
        if position > self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limit "
                f"{self._limit_position}"
            )
            position = self._limit_position
        limits = self._limits()
        if not limits[0] <= position <= limits[1]:
            raise ValueError(f"Position {position} outside focuser range {limits}")
        self._focuser.moveAbs({"z": position}).result()

    def _move_relative(self, delta: float) -> None:
        self._focuser.moveRel({"z": delta}).result()

    def _insert(self) -> Optional[bool]:
        # The favourite active position is calibrated (Delmic), so the user safety
        # limit is not applied here.
        active_position = self.active_position
        if active_position is None:
            logging.warning(
                "Cannot insert objective: no active-position metadata available."
            )
            return False
        self._focuser.moveAbs(active_position).result()
        return True

    def _retract(self) -> Optional[bool]:
        deactive_position = self.deactive_position
        if deactive_position is None:
            logging.warning(
                "Cannot retract objective: no deactive-position metadata available."
            )
            return False
        self._focuser.moveAbs(deactive_position).result()
        return True


class OdemisFM(FM):
    """The Odemis FM group: a channel set up on the stream, then a frame; and live
    view, as the stream running with the camera's frames pulled."""

    def __init__(
        self,
        stream: Any,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self._stream = stream

    def check_health(self) -> Optional[str]:
        """Whether odemis answers: one live read, the camera's exposure time."""
        self.camera.exposure_time.get_value()
        return None

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        """The old ``set_channel``: excitation, emission, power, exposure, gain."""
        if channel is None:
            return
        from fibsem.fm.microscope import emission_filter_named

        settings = ChannelSettings.from_dict(channel)
        filters = self.filter_set
        filters.excitation_wavelength.write_through(settings.excitation_wavelength)
        emission: Union[None, str, float] = settings.emission_wavelength
        if isinstance(emission, str):
            filters.select_fluorescence()
        else:
            filters.emission_filter.write_through(
                emission_filter_named(emission, filters.emission_filter.choices)
            )
        self.light_source.power.write_through(settings.power)
        camera = self.camera
        camera.exposure_time.write_through(settings.exposure_time)
        if settings.gain is not None and "gain" in camera.parameters:
            camera.gain.write_through(settings.gain)

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        return self.camera.acquire()

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        from fibsem.fm.odemis import _frame_metadata_from_data

        data = self._acquire_channel(channel)
        metadata = self._frame_metadata()
        # What odemis stamped on the frame at exposure time, over the parts' state:
        # its acquisition date among them, over the time `FM.acquire_frame` took.
        stamped = _frame_metadata_from_data(data) or {}
        metadata.update({key: to_wire(value) for key, value in stamped.items()})
        return Frame(data, metadata)

    # -- live view: the stream running, frames pulled -----------------------------------

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        self._apply_channel(channel)
        self._stream.is_active.value = True  # light on, filters set, streaming

    def _stop_live(self) -> None:
        try:
            self._stream.is_active.value = False  # light off, camera stopped
        except Exception as e:
            logging.error(f"Failed to deactivate stream after live view: {e}")


def bind_odemis_fm(
    parent: Any = None,
    resources: Optional[Resources] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Device]:
    """The METEOR's FM parts and group, by device name, from the odemis backend on
    this computer: its components by role, and one ``FluoStream`` over them, as
    the old ``OdemisFluorescenceMicroscope`` built it. *config* is the fm entry's own
    keys (``mount_transform``)."""
    from fibsem.fm.odemis import FluoStream, model

    resources = resources if resources is not None else resources_of(parent)
    camera = model.getComponent(role="ccd")
    light_source = model.getComponent(role="light")
    light_filter = model.getComponent(role="filter")
    focuser = model.getComponent(role="focus")
    stream = FluoStream(
        name="fm-stream",
        detector=camera,
        dataflow=camera.data,
        emitter=light_source,
        em_filter=light_filter,
        focuser=focuser,
    )
    common = {"parent": parent, "resources": resources}
    parts: Dict[str, Device] = {
        "objective": OdemisFMObjective(
            focuser, model.getComponent(role="lens"), **common
        ),
        "camera": OdemisFMCamera(camera, stream, **common),
        "light_source": OdemisFMLightSource(stream, **common),
        "filter_set": OdemisFMFilterSet(stream, **common),
    }
    parts["camera"].configure(config)
    group = OdemisFM(stream, **common).fill_roles(**parts)
    return {device.name: device.connect() for device in [group, *parts.values()]}

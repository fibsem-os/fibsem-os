"""The METEOR's FM through odemis as devices, beside the untouched Odemis FM.

``OdemisFMCamera``, ``OdemisFMLightSource``, ``OdemisFMFilterSet``,
``OdemisFMObjective`` and the ``OdemisFM`` group are ``OdemisFluorescenceMicroscope``'s
parts (``fibsem.fm.odemis``) moved onto the FM devices: each read, write and command
makes the odemis calls the old property or method makes, in the same order, on the same
components and the same ``FluoStream``. ``OdemisThermoMicroscope.fm`` is the FM API over
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

This module imports odemis, through ``fibsem.fm.odemis``, so it only loads where odemis
is installed.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from fibsem.devices.core import Device, ParameterMetadata, Resources, resources_of
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.wire import Frame, to_wire
from fibsem.fm.odemis import (
    OBJECTIVE_POSITION_ATOL,
    FluoStream,
    _frame_metadata_from_data,
    _odemis_bands_nm,
    acquire,
    fluo,
    model,
)
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    band_name,
    objective_device_state,
)
from fibsem.structures import InsertableDeviceState, RangeLimit


def emission_filter_of(choice: Any) -> EmissionFilter:
    """The filter an odemis emission choice is: pass-through is reflection; a band, or
    a multi-band filter's bands, in nm."""
    if isinstance(choice, str) and choice == model.BAND_PASS_THROUGH:
        return REFLECTION
    bands = _odemis_bands_nm(choice)
    return EmissionFilter(band_name(bands), bands=bands)


class OdemisFMCamera(Camera):
    """``OdemisCamera``: the ``ccd`` component, with acquisitions through the stream."""

    def __init__(
        self,
        camera: Any,
        stream: Any,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
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

    # A camera without a gain control has no gain parameter; the old class reads it as
    # None and ignores a write.
    def available_gain(self) -> bool:
        return model.hasVA(self._camera, "gain")

    # Gain is a fraction of the camera's maximum, as power is of the light's.
    def _max_gain(self) -> Optional[float]:
        """The top of the gain VA's range or choices; None when it gives neither."""
        va = self._camera.gain
        top = va.range[1] if getattr(va, "range", None) else None
        if top is None and getattr(va, "choices", None):
            top = max(va.choices)
        return float(top) if top is not None and top > 0 else None

    def read_gain(self) -> float:
        max_gain = self._max_gain()
        value = self._camera.gain.value
        return value if max_gain is None else value / max_gain

    def write_gain(self, value: float) -> None:
        max_gain = self._max_gain()
        if max_gain is None:
            self._camera.gain.value = value
            return
        if not 0.0 <= value <= 1.0:
            logging.warning(f"Gain fraction {value} outside [0, 1], clipping.")
            value = min(max(value, 0.0), 1.0)
        raw = value * max_gain
        choices = getattr(self._camera.gain, "choices", None)
        if choices:
            raw = min(choices, key=lambda c: abs(c - raw))
        self._camera.gain.value = raw

    def metadata_gain(self) -> ParameterMetadata:
        max_gain = self._max_gain()
        if max_gain is None:
            return ParameterMetadata()
        return ParameterMetadata(
            limits=RangeLimit(min=0.0, max=1.0),
            native_max=max_gain,
            native_unit=getattr(self._camera.gain, "unit", None),
        )

    def read_offset(self) -> float:
        """The camera's MD_BASELINE, read at connect. Read-only, as on the old class."""
        return self._offset

    def read_pixel_size(self) -> tuple:
        """Sample-plane, from the camera metadata: odemis republishes it on binning and
        magnification changes, so no binning scaling here."""
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
    """``OdemisLightSource``: the stream's power, as a fraction of its maximum."""

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
    """``OdemisFilterSet``: the stream's excitation and emission bands."""

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
    """``OdemisObjectiveLens``: the ``focus`` actuator and the ``lens`` component.

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
        return self._favourite("active_position", model.MD_FAV_POS_ACTIVE, "active")

    @property
    def deactive_position(self) -> Optional[dict]:
        """The retracted position, ``{"z": metres}``, or None when odemis has none."""
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
        parts: Dict[str, Device],
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self._stream = stream
        self.parts = parts

    def check_health(self) -> Optional[str]:
        """Whether odemis answers: one live read, the camera's exposure time."""
        self.parts["camera"].exposure_time.get_value()
        return None

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        """The old ``set_channel``: excitation, emission, power, exposure, gain."""
        if channel is None:
            return
        from fibsem.fm.api import emission_filter_named

        settings = ChannelSettings.from_dict(channel)
        filters = self.parts["filter_set"]
        filters.excitation_wavelength.write_through(settings.excitation_wavelength)
        emission: Union[None, str, float] = settings.emission_wavelength
        if isinstance(emission, str):
            filters.select_fluorescence()
        else:
            filters.emission_filter.write_through(
                emission_filter_named(emission, filters.emission_filter.choices)
            )
        self.parts["light_source"].power.write_through(settings.power)
        camera = self.parts["camera"]
        camera.exposure_time.write_through(settings.exposure_time)
        if settings.gain is not None and "gain" in camera.parameters:
            camera.gain.write_through(settings.gain)

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        return self.parts["camera"].acquire()

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        acquisition_date = datetime.now().isoformat()
        data = self._acquire_channel(channel)
        metadata = {"acquisition_date": acquisition_date, **self._frame_metadata()}
        # What odemis stamped on the frame at exposure time, over the parts' state.
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
    parent: Any = None, resources: Optional[Resources] = None
) -> Dict[str, Device]:
    """The METEOR's FM parts and group, by device name, from the odemis backend on
    this computer: its components by role, and one ``FluoStream`` over them, as
    ``OdemisFluorescenceMicroscope`` builds it."""
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
    group = OdemisFM(stream, parts, **common)
    return {device.name: device.connect() for device in [group, *parts.values()]}

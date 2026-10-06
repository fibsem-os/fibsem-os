"""The Thermo Fisher FM (Arctis, Hydra) as devices, beside the untouched Thermo FM.

``AutoscriptFMCamera``, ``AutoscriptFMLightSource``, ``AutoscriptFMFilterSet``,
``AutoscriptFMObjective`` and the ``AutoscriptFM`` group are
``ThermoFisherFluorescenceMicroscope``'s parts (``fibsem.fm.autoscript``) moved onto the
FM devices: each read, write and command makes the SDK calls the old property or method
makes, in the same order. Nothing builds them yet: ``ThermoMicroscope.fm`` is still the
old class, and pointing it at these is the next step.

The channel. The FM and the beams are one AutoScript connection with one active view,
so whoever sets the view last owns it (FIB-517). ``AutoscriptFMChannel`` is the old
class's ``active_channel()`` as it is: point the connection at the FM for a block, put
the view back after the outermost block, count the depth so a run isn't undone by each
step inside it, and take the microscope's ``imaging_channel`` lock (its
``_threading_lock``) for that bookkeeping only. A parameter in ``needs_channel`` runs in
that scope. The generic claim isn't used because it holds the lock through the read and
always takes it; the old scope skips the lock when the FM already has the view, which is
what keeps the objective movable while live view re-takes the lock every frame.

Live view keeps what the old fast acquisition does, with each frame pulled rather than
pushed: `AutoscriptFM.start_live` switches the light on and starts the acquisition, each
``acquire_frame`` while live is the next ``imaging.get_image()``, and stopping switches
the light off and stops the acquisition.

This module imports the SDK, through ``fibsem.fm.autoscript``, so it only loads where
AutoScript is installed.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from fibsem.devices.core import (
    IMAGING_CHANNEL,
    BoundParameter,
    Device,
    ParameterMetadata,
    Resources,
    resources_of,
)
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.devices.wire import Frame
from fibsem.fm.autoscript import (
    AVAILABLE_FM_WAVELENGTHS,
    COLOR_TO_WAVELENGTH,
    DEFAULT_CONFIGURATION,
    WAVELENGTH_TO_COLOR,
    CameraEmissionType,
    CameraFilterType,
    GrabFrameSettings,
    ImagingDevice,
    ImagingState,
)
from fibsem.fm.microscope import SIM_CAMERA_OFFSET
from fibsem.fm.structures import (
    REFLECTION,
    ChannelSettings,
    EmissionFilter,
    emission_filter_for,
    objective_device_state,
)
from fibsem.structures import InsertableDeviceState, RangeLimit

if TYPE_CHECKING:
    from fibsem.microscopes.autoscript import ThermoMicroscope

FM_ACTIVE_VIEW = 3
"""The FM's view on Arctis, as the old class sets it."""

MULTI_BAND = emission_filter_for("Fluorescence", {})
"""Thermo's one fluorescence filter, multi-band; the other choice is reflection."""


class AutoscriptFMChannel:
    """``ThermoFisherFluorescenceMicroscope.active_channel()``, shared by the FM devices.

    ``scope()`` is the old method unchanged: see its docstring in
    ``fibsem.fm.autoscript`` for why the lock covers the bookkeeping and not the body,
    and why a connection already on the FM takes no lock at all.
    """

    def __init__(self, microscope: ThermoMicroscope, lock: Any):
        self._microscope = microscope
        self.lock = lock
        self._depth = 0
        self._restore_view: Optional[int] = None

    @property
    def connection(self) -> Any:
        """The microscope's AutoScript client, looked up on each call."""
        return self._microscope.connection

    def set_active_channel(self) -> None:
        """Point the connection at the FM and leave it there (the old method)."""
        self.connection.imaging.set_active_view(FM_ACTIVE_VIEW)
        self.connection.imaging.set_active_device(
            ImagingDevice.FLUORESCENCE_LIGHT_MICROSCOPE
        )

    def settings(self) -> Any:
        """The FM camera's settings, selecting the FM first, as ``fm_settings`` does."""
        self.set_active_channel()
        return self.connection.detector.camera_settings

    def _is_ours(self) -> bool:
        return self.connection.imaging.get_active_view() == FM_ACTIVE_VIEW

    @contextmanager
    def scope(self) -> Iterator[None]:
        """Hold the connection on the FM for the block, then put the view back."""
        if self._depth == 0 and self._is_ours():
            yield
            return

        with self.lock:
            if self._depth == 0:
                self._restore_view = self.connection.imaging.get_active_view()
            self.set_active_channel()
            self._depth += 1
        try:
            yield
        finally:
            with self.lock:
                self._depth -= 1
                if self._depth == 0:
                    self.connection.imaging.set_active_view(self._restore_view)


class _OnTheFMChannel(Device):
    """A part whose ``needs_channel`` parameters run in the FM channel's scope."""

    def __init__(
        self,
        name: str,
        channel: AutoscriptFMChannel,
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name=name, parent=parent, resources=resources)
        self._channel = channel

    @contextmanager
    def _claim(self, param: BoundParameter) -> Iterator[None]:
        if not param.needs_channel:
            yield
            return
        with self._channel.scope():
            yield


class AutoscriptFMFilterSet(_OnTheFMChannel, FilterSet):
    """The excitation is held here, as the old filter set holds it: AutoScript's
    emission type is read-only, so the colour is applied when the light starts. The
    emission filter is the camera's filter type, reflection or fluorescence."""

    needs_channel = frozenset({"emission_filter"})

    def __init__(self, channel: AutoscriptFMChannel, **kwargs: Any):
        super().__init__("filter_set", channel, **kwargs)
        self.emission_type = CameraEmissionType.RED

    def read_excitation_wavelength(self) -> float:
        color = self.emission_type
        if color not in COLOR_TO_WAVELENGTH:
            raise ValueError(
                f"Invalid excitation color: {color}: must be one of "
                f"{list(COLOR_TO_WAVELENGTH.keys())}"
            )
        return COLOR_TO_WAVELENGTH[color]

    def write_excitation_wavelength(self, value: float) -> None:
        color = WAVELENGTH_TO_COLOR.get(value, None)
        if color is None:
            closest = min(COLOR_TO_WAVELENGTH.values(), key=lambda x: abs(x - value))
            color = WAVELENGTH_TO_COLOR[closest]
        self.emission_type = color

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        return ParameterMetadata(choices=sorted(AVAILABLE_FM_WAVELENGTHS))

    def read_emission_filter(self) -> EmissionFilter:
        mode = self._channel.settings().filter.type.value
        if mode is CameraFilterType.FLUORESCENCE:
            return MULTI_BAND
        return REFLECTION

    def write_emission_filter(self, value: EmissionFilter) -> None:
        if value not in (REFLECTION, MULTI_BAND):
            raise ValueError(f"filter_set has no emission filter {value}")
        settings = self._channel.settings()
        if value == REFLECTION:
            settings.filter.type.value = CameraFilterType.REFLECTION
        else:
            settings.filter.type.value = CameraFilterType.FLUORESCENCE

    def metadata_emission_filter(self) -> ParameterMetadata:
        return ParameterMetadata(choices=[REFLECTION, MULTI_BAND])


class AutoscriptFMCamera(_OnTheFMChannel, Camera):
    """The FM camera's settings. Gain is the detector's contrast; offset is a setting
    kept here, as the old camera keeps it; pixel size and resolution are the objective's
    configuration over the binning."""

    needs_channel = frozenset({"exposure_time", "binning", "gain"})

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        filter_set: AutoscriptFMFilterSet,
        **kwargs: Any,
    ):
        super().__init__("camera", channel, **kwargs)
        self._filter_set = filter_set
        self._offset = SIM_CAMERA_OFFSET
        self._pixel_size: Tuple[float, float] = DEFAULT_CONFIGURATION["pixel_size"]
        self._resolution: Tuple[int, int] = DEFAULT_CONFIGURATION["resolution"]

    def _exposure_time_limits(self) -> Tuple[float, float]:
        with self._channel.scope():
            limits = self._channel.settings().exposure_time.limits
            return (limits.min, limits.max)

    def read_exposure_time(self) -> float:
        return self._channel.settings().exposure_time.value

    def write_exposure_time(self, value: float) -> None:
        limits = self._exposure_time_limits()
        if not limits[0] <= value <= limits[1]:
            raise ValueError(
                f"Exposure time must be between {limits[0]} and {limits[1]}, got {value}"
            )
        self._channel.settings().exposure_time.value = value

    def metadata_exposure_time(self) -> ParameterMetadata:
        low, high = self._exposure_time_limits()
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def _available_binnings(self) -> Tuple[int, ...]:
        with self._channel.scope():
            return self._channel.settings().binning.available_values

    def _binning(self) -> int:
        with self._channel.scope():
            return self._channel.settings().binning.value

    def read_binning(self) -> int:
        return self._channel.settings().binning.value

    def write_binning(self, value: int) -> None:
        # The choices are read again for the message, as the old setter reads them.
        if value not in self._available_binnings():
            raise ValueError(
                f"Binning must be one of {self._available_binnings()}, got {value}"
            )
        self._channel.settings().binning.value = value

    def metadata_binning(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._available_binnings()))

    def read_gain(self) -> float:
        return self._channel.connection.detector.contrast.value

    def write_gain(self, value: float) -> None:
        self._channel.connection.detector.contrast.value = value

    def metadata_gain(self) -> ParameterMetadata:
        # The FM detector's contrast, which AutoScript keeps as a fraction already.
        with self._channel.scope():
            limits = self._channel.connection.detector.contrast.limits
            return ParameterMetadata(limits=RangeLimit(min=limits.min, max=limits.max))

    def read_offset(self) -> float:
        return self._offset

    def write_offset(self, value: float) -> None:
        if value < 0:
            raise ValueError("Offset must be non-negative.")
        self._offset = value

    # Binning read once per axis, as the old properties read it.

    def read_pixel_size(self) -> tuple:
        return (
            self._pixel_size[0] * self._binning(),
            self._pixel_size[1] * self._binning(),
        )

    def read_resolution(self) -> tuple:
        return (
            self._resolution[0] // self._binning(),
            self._resolution[1] // self._binning(),
        )

    def _acquire(self) -> np.ndarray:
        # The old camera grabs unscoped and leaves the FM selected, relying on the
        # acquisition around it to hold the channel. Here the grab holds it itself;
        # inside an acquisition that is the same calls, and a bare grab can't leave
        # the connection on the FM (FIB-517).
        with self._channel.scope():
            frame_settings = GrabFrameSettings(
                emission_type=self._filter_set.emission_type
            )
            self._channel.set_active_channel()
            image = self._channel.connection.imaging.grab_frame(frame_settings)
            return image.data

    # The acquisition live view runs, as the old camera starts and stops it.

    def _start_acquisition(self) -> None:
        self._channel.set_active_channel()
        self._channel.connection.imaging.start_acquisition()

    def _stop_acquisition(self) -> None:
        self._channel.set_active_channel()
        self._channel.connection.imaging.stop_acquisition()


class AutoscriptFMLightSource(_OnTheFMChannel, LightSource):
    """The light's brightness. AutoScript sets it for the colour that is emitting, so
    a write switches the light on in the filter set's colour, sets it, and switches it
    off again, as the old light source does."""

    needs_channel = frozenset({"power"})

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        filter_set: AutoscriptFMFilterSet,
        **kwargs: Any,
    ):
        super().__init__("light_source", channel, **kwargs)
        self._filter_set = filter_set

    def read_power(self) -> float:
        return self._channel.connection.detector.brightness.value

    def write_power(self, value: float) -> None:
        self.start_emission(self._filter_set.emission_type)
        self._channel.connection.detector.brightness.value = value
        self.stop_emission()

    def metadata_power(self) -> ParameterMetadata:
        with self._channel.scope():
            limits = self._channel.connection.detector.brightness.limits
            return ParameterMetadata(limits=RangeLimit(min=limits.min, max=limits.max))

    def start_emission(self, emission_type: Any) -> None:
        with self._channel.scope():
            self._channel.connection.detector.camera_settings.emission.start(
                emission_type=emission_type
            )

    def stop_emission(self) -> None:
        with self._channel.scope():
            self._channel.connection.detector.camera_settings.emission.stop()


class AutoscriptFMObjective(_OnTheFMChannel, Objective):
    """The objective: its focus is the camera's focus setting, and inserting or
    retracting it is the detector's. Magnification and numerical aperture are the
    configuration's; ``limit_position`` is the furthest a move may go in."""

    needs_channel = frozenset({"position", "state"})

    def __init__(self, channel: AutoscriptFMChannel, **kwargs: Any):
        super().__init__("objective", channel, **kwargs)
        self._magnification = DEFAULT_CONFIGURATION["magnification"]
        self._numerical_aperture = DEFAULT_CONFIGURATION["numerical_aperture"]
        self._limit_position: float = DEFAULT_CONFIGURATION.get(
            "limit_position", 8.6e-3
        )

    def _focus_limits(self) -> Tuple[float, float]:
        with self._channel.scope():
            limits = self._channel.settings().focus.limits
            return (limits.min, limits.max)

    def read_position(self) -> float:
        return self._channel.settings().focus.value

    def metadata_position(self) -> ParameterMetadata:
        low, high = self._focus_limits()
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_state(self) -> InsertableDeviceState:
        return objective_device_state(self._channel.connection.detector.state)

    def read_magnification(self) -> float:
        return self._magnification

    def read_numerical_aperture(self) -> float:
        return self._numerical_aperture

    def read_limit_position(self) -> float:
        return self._limit_position

    def write_limit_position(self, value: float) -> None:
        self._limit_position = value
        logging.info(
            f"Objective user-defined position limit set to: "
            f"{self._limit_position * 1e3:.3f} mm"
        )

    def _move_absolute(self, position: float) -> None:
        # The hardware limits are read for each bound, as the old move reads them.
        if not self._focus_limits()[0] <= position <= self._focus_limits()[1]:
            raise ValueError(
                f"Position {position} out of limits {self._focus_limits()}"
            )
        if not position <= self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limits "
                f"{self._limit_position}"
            )
            position = np.clip(position, 0, self._limit_position)
        with self._channel.scope():
            self._channel.settings().focus.value = position

    def _move_relative(self, delta: float) -> None:
        self._move_absolute(self.position.get_value() + delta)

    def _insert(self) -> bool:
        with self._channel.scope():
            if self.state.get_value() is InsertableDeviceState.INSERTED:
                logging.warning("Objective lens is already inserted.")
                return False
            self._channel.connection.detector.insert()
            return True

    def _retract(self) -> bool:
        with self._channel.scope():
            if self.state.get_value() is InsertableDeviceState.RETRACTED:
                logging.warning("Objective lens is already retracted.")
                return False
            self._channel.connection.detector.retract()
            return True


class AutoscriptFM(FM):
    """The Thermo FM group: a channel set up on its parts, then a frame, holding the
    FM channel throughout; and live view, as the old fast acquisition runs it."""

    def __init__(
        self,
        channel: AutoscriptFMChannel,
        parts: Dict[str, Device],
        parent: Any = None,
        resources: Optional[Resources] = None,
    ):
        super().__init__(name="fm", parent=parent, resources=resources)
        self._channel = channel
        self.parts = parts

    def check_health(self) -> Optional[str]:
        """Whether the FM answers: one live read, the camera's exposure time."""
        self.parts["camera"].exposure_time.get_value()
        return None

    def _apply_channel(self, channel: Optional[Dict[str, Any]]) -> None:
        """The old ``set_channel``: excitation, emission filter, power, exposure, gain."""
        if channel is None:
            return
        from fibsem.fm.api import emission_filter_named

        settings = ChannelSettings.from_dict(channel)
        filters = self.parts["filter_set"]
        filters.excitation_wavelength.write_through(settings.excitation_wavelength)
        filters.emission_filter.write_through(
            emission_filter_named(
                settings.emission_wavelength, filters.emission_filter.choices
            )
        )
        self.parts["light_source"].power.write_through(settings.power)
        camera = self.parts["camera"]
        camera.exposure_time.write_through(settings.exposure_time)
        if settings.gain is not None:
            camera.gain.write_through(settings.gain)

    def _frame(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        self._apply_channel(channel)
        if self.is_live:
            return self._live_frame()
        return self.parts["camera"].acquire()

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        with self._channel.scope():
            return self._frame(channel)

    def _acquire_frame(self, channel: Optional[Dict[str, Any]]) -> Frame:
        # The metadata is read inside the scope too, as the old acquisition reads it,
        # so it describes the state the frame was taken in.
        with self._channel.scope():
            acquisition_date = datetime.now().isoformat()
            data = self._frame(channel)
            metadata = {"acquisition_date": acquisition_date, **self._frame_metadata()}
            return Frame(data, metadata)

    # -- live view: the old fast acquisition, pulled ------------------------------------

    def _start_live(self, channel: Optional[Dict[str, Any]]) -> None:
        self._apply_channel(channel)
        connection = self._channel.connection
        with self._channel.lock:
            self._channel.set_active_channel()
            # The colour the hardware has, as the old fast acquisition starts it.
            emission_color = connection.detector.camera_settings.emission.type.value
            self.parts["light_source"].start_emission(emission_type=emission_color)
            self.parts["camera"]._start_acquisition()

    def _live_frame(self) -> np.ndarray:
        if self._channel.connection.imaging.state != ImagingState.ACQUIRING:
            # Stopped from outside (the microscope's own UI): end live view as the old
            # loop did, light off and acquisition stopped.
            self.stop_live()
            raise RuntimeError("The FM acquisition stopped; live view has ended.")
        with self._channel.lock:
            # Re-forced each frame, as the old loop does: something may have taken it.
            self._channel.set_active_channel()
            return self._channel.connection.imaging.get_image().data

    def _stop_live(self) -> None:
        self.parts["light_source"].stop_emission()
        self.parts["camera"]._stop_acquisition()


def bind_autoscript_fm(
    microscope: ThermoMicroscope,
    resources: Optional[Resources] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Device]:
    """The Thermo FM's parts and group for a connected Thermo microscope, by device
    name. They share the microscope's ``imaging_channel`` lock with the beams.
    *config* is the fm entry's own keys (``mount_transform``)."""
    resources = resources if resources is not None else resources_of(microscope)
    channel = AutoscriptFMChannel(microscope, resources.lock(IMAGING_CHANNEL))
    common = {"parent": microscope, "resources": resources}
    filter_set = AutoscriptFMFilterSet(channel, **common)
    parts: Dict[str, Device] = {
        "camera": AutoscriptFMCamera(channel, filter_set, **common),
        "light_source": AutoscriptFMLightSource(channel, filter_set, **common),
        "filter_set": filter_set,
        "objective": AutoscriptFMObjective(channel, **common),
    }
    parts["camera"].configure(config)
    group = AutoscriptFM(channel, parts, **common)
    return {device.name: device.connect() for device in [group, *parts.values()]}

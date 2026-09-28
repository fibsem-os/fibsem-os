"""A fluorescence microscope on another computer, behind today's FM API.

``RemoteFluorescenceMicroscope`` is a ``FluorescenceMicroscope`` like the AutoScript
and Odemis ones, so the FM UI, acquisition and workflows use it unchanged. Its parts
forward to the remote devices of ``fibsem.devices.drivers.remote``, served by
``fibsem.server.devices`` on the FM's computer (the METEOR PC, say):

    fm = RemoteFluorescenceMicroscope.connect("192.168.0.20", 8765, parent=microscope)
    fm.objective.insert()
    image = fm.acquire_image(channel_settings)

Every property is a live read or a write through the old API path, as the local
drivers' are: nothing here is cached behind the caller's back, so a guard reading
``fm.objective.state`` asks the FM's computer, and fails closed with
``RemoteDeviceUnreachable`` if it can't. Acquiring a channel is one call, run on the
FM's computer; only the metadata reads come back separately.

Two things stay on this computer, since they belong to the session rather than the
hardware: the objective's saved focus position, and the channel name and colour.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.fm.structures import ChannelSettings, FluorescenceImage

if TYPE_CHECKING:
    from fibsem.devices.core import Device
    from fibsem.devices.drivers.remote import DeviceClient
    from fibsem.microscope import FibsemMicroscope


class RemoteObjectiveLens(ObjectiveLens):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device
        self._focus_position = None  # a session setting, kept here

    @property
    def magnification(self) -> float:
        return self._device.magnification.get_value()

    @property
    def numerical_aperture(self) -> float:
        return self._device.numerical_aperture.get_value()

    @property
    def position(self) -> float:
        return self._device.position.get_value()

    @property
    def limit_position(self) -> float:
        return self._device.limit_position.get_value()

    @limit_position.setter
    def limit_position(self, position: float) -> None:
        self._device.limit_position.write_through(position)

    @property
    def limits(self) -> Tuple[float, float]:
        limits = self._device.position.limits
        return (limits.min, limits.max)

    @property
    def state(self) -> str:
        return self._device.state.get_value()

    def move_relative(self, delta: float) -> None:
        self._device.move_relative(delta)
        self._notify_moved()

    def move_absolute(self, position: float) -> None:
        self._device.move_absolute(position)
        self._notify_moved()

    def insert(self) -> None:
        self._device.insert()
        self._notify_moved()

    def retract(self) -> None:
        self._device.retract()
        self._notify_moved()


class RemoteCamera(Camera):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    def acquire_image(self) -> np.ndarray:
        return self._device.acquire()

    @property
    def exposure_time(self) -> float:
        return self._device.exposure_time.get_value()

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        self._device.exposure_time.write_through(value)

    @property
    def binning(self) -> int:
        return self._device.binning.get_value()

    @binning.setter
    def binning(self, value: int) -> None:
        self._device.binning.write_through(value)

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        return tuple(self._device.binning.choices or ())

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        limits = self._device.exposure_time.limits
        return (limits.min, limits.max)

    @property
    def gain(self) -> float:
        return self._device.gain.get_value()

    @gain.setter
    def gain(self, value: float) -> None:
        self._device.gain.write_through(value)

    @property
    def offset(self) -> float:
        return self._device.offset.get_value()

    @offset.setter
    def offset(self, value: float) -> None:
        self._device.offset.write_through(value)

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return tuple(self._device.pixel_size.get_value())

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(self._device.resolution.get_value())


class RemoteLightSource(LightSource):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def power(self) -> float:
        return self._device.power.get_value()

    @power.setter
    def power(self, value: float) -> None:
        self._device.power.write_through(value)

    @property
    def power_limits(self) -> Tuple[float, float]:
        limits = self._device.power.limits
        return (limits.min, limits.max)


class RemoteFilterSet(FilterSet):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return tuple(self._device.excitation_wavelength.choices or ())

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        return tuple(self._device.emission_wavelength.choices or ())

    @property
    def excitation_wavelength(self) -> float:
        return self._device.excitation_wavelength.get_value()

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        self._device.excitation_wavelength.write_through(value)

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        return self._device.emission_wavelength.get_value()

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        self._device.emission_wavelength.write_through(value)


class RemoteFluorescenceMicroscope(FluorescenceMicroscope):
    """Today's FM API over an FM served from another computer."""

    objective: RemoteObjectiveLens
    camera: RemoteCamera
    light_source: RemoteLightSource
    filter_set: RemoteFilterSet

    def __init__(
        self,
        devices: Dict[str, Device],
        parent: Optional[FibsemMicroscope] = None,
    ):
        super().__init__(parent=parent)
        self.devices = devices
        self.objective = RemoteObjectiveLens(devices["objective"], parent=self)
        self.camera = RemoteCamera(devices["camera"], parent=self)
        self.light_source = RemoteLightSource(devices["light_source"], parent=self)
        self.filter_set = RemoteFilterSet(devices["filter_set"], parent=self)

    @classmethod
    def connect(
        cls,
        host: str,
        port: int,
        parent: Optional[FibsemMicroscope] = None,
        client: Optional[DeviceClient] = None,
    ) -> RemoteFluorescenceMicroscope:
        """Connect to the FM's device server. Raises ``RemoteDeviceUnreachable``
        if it isn't running."""
        from fibsem.devices.drivers.remote import connect_remote_fm

        devices = connect_remote_fm(host, port, client=client)
        missing = {"fm", "camera", "light_source", "filter_set", "objective"} - set(
            devices
        )
        if missing:
            raise RuntimeError(f"{host}:{port} serves no FM {sorted(missing)}")
        logging.info(f"Connected to the fluorescence microscope at {host}:{port}")
        return cls(devices, parent=parent)

    @property
    def client(self) -> Any:
        return self.devices["fm"].client

    def acquire_image(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> FluorescenceImage:
        """One call to the FM's computer sets up the channel and takes the frame."""
        with self.active_channel():
            if channel_settings is None:
                data = self.devices["camera"].acquire()
            else:
                # The name and colour are this session's labels, not hardware.
                self.channel_name = channel_settings.name
                self.channel_color = channel_settings.color
                data = self.devices["fm"].acquire_channel(channel_settings.to_dict())
            return self._construct_image(data)

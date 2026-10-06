"""Today's FM API over FM devices, wherever the devices are.

``DeviceFluorescenceMicroscope`` is a ``FluorescenceMicroscope`` whose parts forward
to the FM's devices (``fibsem.devices.fm``): the camera, light source, filter set
and objective, and the ``fm`` group that runs a channel. It doesn't know which driver
built them, or whether they run in this process or on another computer, so the FM
UI, acquisition and workflows use it unchanged either way:

- the Demo's FM is this over the Demo FM devices (``fibsem.devices.drivers.demo``);
- ``RemoteFluorescenceMicroscope`` (``fibsem.fm.remote``) is this over remote devices,
  plus connecting to their server.

Every property is a live read, or a write through the old API path
(``write_through``): nothing is cached behind the caller's back, so a guard reading
``fm.objective.state`` asks the device, and a remote one fails closed with
``RemoteDeviceUnreachable`` when its server can't be reached. Acquiring a channel is
one command on the ``fm`` group, run next to the hardware.

What belongs to the session rather than the hardware stays here: the objective's
saved focus position, the channel name and colour, and the image transform.
"""

from __future__ import annotations

import logging
import threading
from copy import deepcopy
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from fibsem.devices.fm import mount_transform_from_name
from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.fm.progress import (
    FluorescenceAcquisitionProgress,
    FluorescenceAcquisitionStatus,
)
from fibsem.fm.structures import (
    REFLECTION,
    CameraImageTransform,
    ChannelSettings,
    EmissionFilter,
    FluorescenceChannelMetadata,
    FluorescenceImage,
    FluorescenceImageMetadata,
    ZParameters,
    ZStackOrder,
    emission_filter_for,
    objective_state_name,
    same_emission_value,
)

if TYPE_CHECKING:
    from fibsem.devices.core import BoundParameter, Device
    from fibsem.microscope import FibsemMicroscope

FM_DEVICE_NAMES = ("fm", "camera", "light_source", "filter_set", "objective")
"""The devices an FM is made of: the four parts and the group."""


def _param(device: Device, name: str) -> BoundParameter:
    """A part's parameter, failing closed while a remote part's server has never
    answered.

    A remote part built offline has no parameters yet, so a guard reading
    ``fm.objective.state`` gets the same ``RemoteDeviceUnreachable`` as one whose
    server went away later, not an ``AttributeError`` that reads like a missing
    feature. A local part is always online.
    """
    if not getattr(device, "online", True):
        from fibsem.devices.drivers.remote import RemoteDeviceUnreachable

        raise RemoteDeviceUnreachable(
            f"{device.name} at {device.client.base_url} has not connected yet"
        )
    return getattr(device, name)


class DeviceObjectiveLens(ObjectiveLens):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device
        self._focus_position = None  # a session setting, kept here

    @property
    def magnification(self) -> float:
        return _param(self._device, "magnification").get_value()

    @property
    def numerical_aperture(self) -> float:
        return _param(self._device, "numerical_aperture").get_value()

    @property
    def position(self) -> float:
        return _param(self._device, "position").get_value()

    @property
    def limit_position(self) -> float:
        return _param(self._device, "limit_position").get_value()

    @limit_position.setter
    def limit_position(self, position: float) -> None:
        _param(self._device, "limit_position").write_through(position)

    @property
    def limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "position").limits
        return (limits.min, limits.max)

    @property
    def state(self) -> str:
        return objective_state_name(_param(self._device, "state").get_value())

    def move_relative(self, delta: float) -> None:
        self._device.move_relative(delta)
        self._notify_moved()

    def move_absolute(self, position: float) -> None:
        self._device.move_absolute(position)
        self._notify_moved()

    # Announced unless the driver says nothing moved, as the FM classes announce
    # only a move (`_notify_moved`).

    def insert(self) -> None:
        if self._device.insert() is not False:
            self._notify_moved()

    def retract(self) -> None:
        if self._device.retract() is not False:
            self._notify_moved()


class DeviceCamera(Camera):
    """A camera without a gain control (a driver that offers no ``gain``) reads its
    gain as None and ignores a write, warning once."""

    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device
        self._gain_warning_logged = False

    def _has_no_gain(self) -> bool:
        # An offline remote camera has no parameters yet; reading it fails closed.
        return getattr(self._device, "online", True) and (
            "gain" not in self._device.parameters
        )

    def acquire_image(self) -> np.ndarray:
        return self._device.acquire()

    @property
    def exposure_time(self) -> float:
        return _param(self._device, "exposure_time").get_value()

    @exposure_time.setter
    def exposure_time(self, value: float) -> None:
        _param(self._device, "exposure_time").write_through(value)

    @property
    def binning(self) -> int:
        return _param(self._device, "binning").get_value()

    @binning.setter
    def binning(self, value: int) -> None:
        _param(self._device, "binning").write_through(value)

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        return tuple(_param(self._device, "binning").choices or ())

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "exposure_time").limits
        return (limits.min, limits.max)

    @property
    def gain(self) -> Optional[float]:
        if self._has_no_gain():
            return None
        return _param(self._device, "gain").get_value()

    @gain.setter
    def gain(self, value: float) -> None:
        if self._has_no_gain():
            if not self._gain_warning_logged:
                logging.warning("Camera has no gain control; ignoring gain settings.")
                self._gain_warning_logged = True
            return
        _param(self._device, "gain").write_through(value)

    @property
    def gain_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return _native_scale(self._device, "gain")

    @property
    def offset(self) -> float:
        return _param(self._device, "offset").get_value()

    @offset.setter
    def offset(self, value: float) -> None:
        _param(self._device, "offset").write_through(value)

    @property
    def pixel_size(self) -> Tuple[float, float]:
        return tuple(_param(self._device, "pixel_size").get_value())

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(_param(self._device, "resolution").get_value())


def _native_scale(device: Device, name: str) -> Optional[Tuple[float, Optional[str]]]:
    """A fraction parameter's full scale in hardware units, when the driver gives it."""
    if name not in device.parameters:
        return None
    metadata = _param(device, name).metadata
    if metadata.native_max is None:
        return None
    return (metadata.native_max, metadata.native_unit)


class DeviceLightSource(LightSource):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def power(self) -> float:
        return _param(self._device, "power").get_value()

    @power.setter
    def power(self, value: float) -> None:
        _param(self._device, "power").write_through(value)

    @property
    def power_limits(self) -> Tuple[float, float]:
        limits = _param(self._device, "power").limits
        return (limits.min, limits.max)

    @property
    def power_native_scale(self) -> Optional[Tuple[float, Optional[str]]]:
        return _native_scale(self._device, "power")


def _old_emission_value(found: EmissionFilter) -> Optional[Union[float, str]]:
    if found == REFLECTION:
        return None
    return found.name if found.low is None else found.low


def emission_filter_named(
    value: Optional[Union[float, str]], filters: Sequence[EmissionFilter]
) -> EmissionFilter:
    """The filter among ``filters`` that today's emission value names.

    ``None`` is reflection. A label is the filter of that name, or else the multi-band
    filter, as the FM classes take any label to mean fluorescence. A number is the
    band whose bottom edge is closest, as on Odemis; a filter set with no bands (Thermo,
    the simulator) has only its multi-band filter, which is what Thermo reports a
    number as.
    """
    multi_band = [f for f in filters if f != REFLECTION and f.low is None]
    if value is None:
        matches = [f for f in filters if f == REFLECTION]
    elif isinstance(value, str):
        matches = [f for f in filters if f.name == value] or multi_band
    else:
        banded = [f for f in filters if f.low is not None]
        matches = sorted(banded, key=lambda f: abs(f.low - value))[:1] or multi_band
    if not matches:
        raise ValueError(f"No emission filter for {value!r}")
    return matches[0]


class DeviceFilterSet(FilterSet):
    def __init__(self, device: Device, parent: Optional[FluorescenceMicroscope] = None):
        super().__init__(parent=parent)
        self._device = device

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        return tuple(_param(self._device, "excitation_wavelength").choices or ())

    # Today's API names a filter by one value: None for reflection, a label for a
    # multi-band filter, or the band's bottom edge in nm.

    @property
    def available_emission_wavelengths(self) -> Tuple[Union[None, str, float], ...]:
        return tuple(_old_emission_value(f) for f in self._emission_filters())

    def _emission_filters(self) -> Tuple[EmissionFilter, ...]:
        return tuple(_param(self._device, "emission_filter").choices or ())

    def emission_filter(self, value: Optional[Union[float, str]]) -> EmissionFilter:
        for found in self._emission_filters():
            if same_emission_value(_old_emission_value(found), value):
                return found
        return emission_filter_for(value, {})

    @property
    def excitation_wavelength(self) -> float:
        return _param(self._device, "excitation_wavelength").get_value()

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        _param(self._device, "excitation_wavelength").write_through(value)

    @property
    def emission_wavelength(self) -> Optional[Union[float, str]]:
        return _old_emission_value(_param(self._device, "emission_filter").get_value())

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[Union[float, str]]) -> None:
        found = emission_filter_named(value, self._emission_filters())
        _param(self._device, "emission_filter").write_through(found)


class DeviceFluorescenceMicroscope(FluorescenceMicroscope):
    """Today's FM API over FM devices, local or remote."""

    objective: DeviceObjectiveLens
    camera: DeviceCamera
    light_source: DeviceLightSource
    filter_set: DeviceFilterSet

    def __init__(
        self,
        devices: Dict[str, Device],
        parent: Optional[FibsemMicroscope] = None,
    ):
        super().__init__(parent=parent)
        self.devices = devices
        self.objective = DeviceObjectiveLens(devices["objective"], parent=self)
        self.camera = DeviceCamera(devices["camera"], parent=self)
        self.light_source = DeviceLightSource(devices["light_source"], parent=self)
        self.filter_set = DeviceFilterSet(devices["filter_set"], parent=self)

    def acquire_image(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> FluorescenceImage:
        """One command on the ``fm`` group sets up the channel and takes the frame,
        with what it was taken with, so building the image needs no further reads."""
        with self.active_channel():
            channel = None
            if channel_settings is not None:
                # The name and colour are this session's labels, not hardware.
                self.channel_name = channel_settings.name
                self.channel_color = channel_settings.color
                channel = channel_settings.to_dict()
            frame = self.devices["fm"].acquire_frame(channel)
            return self._construct_image(frame.data, frame.metadata)

    @property
    def mount_transform(self) -> CameraImageTransform:
        """How the camera is mounted, as its device reports it, so an FM on another
        computer brings its own. A camera without the parameter (a server from before
        it) is mounted straight, as every FM was. Read once and kept: every frame
        uses it, and a mount does not change."""
        camera = self.devices["camera"]
        if "mount_transform" not in camera.parameters:
            return CameraImageTransform.NONE
        return mount_transform_from_name(_param(camera, "mount_transform").cached)

    @property
    def runs_z_stack_on_device(self) -> bool:
        """Whether a z-stack is one command on the ``fm`` group: when the group runs
        on another computer and has the command. A local FM runs it step by step, so
        each slice is shown as it arrives."""
        group = self.devices["fm"]
        return getattr(group, "runs_elsewhere", False) and (
            "acquire_z_stack" in getattr(group, "server_commands", group.commands)
        )

    def acquire_z_stack_on_device(
        self,
        channel_settings: Union[ChannelSettings, List[ChannelSettings]],
        zparams: ZParameters,
        stop_event: Optional[threading.Event] = None,
    ) -> Optional[FluorescenceImage]:
        """``fibsem.fm.acquisition.acquire_z_stack`` as one ``fm`` group command:
        the same positions, order, progress and cancelling, with the frames coming
        back together at the end."""
        group = self.devices["fm"]
        channels = (
            channel_settings
            if isinstance(channel_settings, list)
            else [channel_settings]
        )
        with self.active_channel():
            z_init = self.objective.position
            positions = [float(z) for z in zparams.generate_positions(z_init=z_init)]
            order = "z" if zparams.order == ZStackOrder.Z_LEVEL else "channel"

            def on_changed(name: str, value: Any) -> None:
                if name != "progress" or not value:
                    return
                self.acquisition_progress_signal.emit(
                    FluorescenceAcquisitionProgress(
                        status=FluorescenceAcquisitionStatus.ACQUIRING_ZSTACK,
                        **value,
                    )
                )

            done = threading.Event()

            def watch_for_stop() -> None:
                while not done.wait(0.1):
                    if stop_event.is_set():
                        group.cancel()
                        return

            group.changed.connect(on_changed)
            if stop_event is not None:
                threading.Thread(
                    target=watch_for_stop, name="fm-z-stack-stop", daemon=True
                ).start()
            try:
                frames = group.acquire_z_stack(
                    channels=[ch.to_dict() for ch in channels],
                    positions=positions,
                    order=order,
                    restore_position=z_init,
                )
            finally:
                done.set()
                group.changed.disconnect(on_changed)
            # The objective moved on the FM's side; announce where it ended up.
            self.objective._notify_moved()
            if not frames:
                logging.info("Z-stack acquisition cancelled")
                return None

            images: List[FluorescenceImage] = []
            n = len(positions)
            for i, ch in enumerate(channels):
                self.channel_name, self.channel_color = ch.name, ch.color
                planes = [
                    self._construct_image(frame.data, frame.metadata)
                    for frame in frames[i * n : (i + 1) * n]
                ]
                images.append(FluorescenceImage.create_z_stack(planes))
            return FluorescenceImage.create_multi_channel_image(images)

    def _acquisition_worker(
        self, channel_settings: Optional[ChannelSettings] = None
    ) -> None:
        """Live view, pulled: the ``fm`` group keeps the hardware acquiring, and each
        frame is one `acquire_image` with the current settings. Stopping, or this
        process going away, ends it; the group stops by itself if no frame is asked
        for in its ``live_timeout``."""
        group = self.devices["fm"]
        try:
            if channel_settings is not None:
                self.set_channel(channel_settings)
            group.start_live()
            try:
                while not self._stop_acquisition_event.is_set():
                    self.acquire_image()
            finally:
                group.stop_live()
        except Exception as e:
            logging.error(f"Error in acquisition worker: {e}")

    def _metadata_for_frame(
        self, frame_metadata: Optional[dict]
    ) -> FluorescenceImageMetadata:
        """Built from what the ``fm`` group reported with the frame. Anything it didn't
        report (a server from before ``acquire_frame``) is read live, as before."""
        frame = frame_metadata or {}

        def reported(key: str, read: Callable[[], Any]) -> Any:
            return frame[key] if key in frame else read()

        if "emission_filter" in frame:
            emission = _old_emission_value(
                EmissionFilter.from_dict(frame["emission_filter"])
            )
        else:
            emission = self.filter_set.emission_wavelength
        camera, objective = self.camera, self.objective
        channel = FluorescenceChannelMetadata(
            name=self.channel_name,
            color=self.channel_color,
            excitation_wavelength=reported(
                "excitation_wavelength",
                lambda: self.filter_set.excitation_wavelength,
            ),
            emission_wavelength=emission,
            power=reported("power", lambda: self.light_source.power),
            exposure_time=reported("exposure_time", lambda: camera.exposure_time),
            gain=reported("gain", lambda: camera.gain),
            offset=reported("offset", lambda: camera.offset),
            binning=reported("binning", lambda: camera.binning),
            objective_position=reported(
                "objective_position", lambda: objective.position
            ),
            objective_magnification=reported(
                "objective_magnification", lambda: objective.magnification
            ),
            objective_numerical_aperture=reported(
                "objective_numerical_aperture", lambda: objective.numerical_aperture
            ),
        )
        pixel_size = reported("pixel_size", lambda: camera.pixel_size)
        resolution = reported("resolution", lambda: camera.resolution)
        parent = self.parent
        # The coordinator's own state, as `get_metadata` stamps it.
        return FluorescenceImageMetadata(
            acquisition_date=reported(
                "acquisition_date", lambda: datetime.now().isoformat()
            ),
            pixel_size_x=pixel_size[0],
            pixel_size_y=pixel_size[1],
            resolution=(resolution[0], resolution[1]),
            stage_position=parent.get_stage_position() if parent else None,
            geometry=parent.fm_image_geometry() if parent else None,
            experiment=deepcopy(parent.experiment) if parent else None,
            channels=[channel],
        )

"""FM parts as devices, over the existing ``fibsem.fm`` classes.

The FM classes (``fibsem.fm.microscope``, with their AutoScript, Odemis and simulated
implementations) are already split into parts, so this driver only adapts them: each
parameter reads and writes the property it always did, and each command calls the
method it always did. It works for every FM backend at once. An FM class may offer
more than today's API needs (an ``emission_bands`` map of each band's edges); the
driver uses it where it's there.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

from fibsem.devices.core import Device, ParameterMetadata, Resources
from fibsem.devices.fm import FM, Camera, FilterSet, LightSource, Objective
from fibsem.fm.structures import REFLECTION, EmissionFilter, objective_device_state
from fibsem.structures import InsertableDeviceState, RangeLimit

if TYPE_CHECKING:
    from fibsem.fm.microscope import FluorescenceMicroscope


class FMCamera(Camera):
    def __init__(self, camera: Any, **kwargs: Any):
        super().__init__(name="camera", **kwargs)
        self._camera = camera

    def read_exposure_time(self) -> float:
        return self._camera.exposure_time

    def write_exposure_time(self, value: float) -> None:
        self._camera.exposure_time = value

    def metadata_exposure_time(self) -> ParameterMetadata:
        low, high = self._camera.exposure_time_limits
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_binning(self) -> int:
        return self._camera.binning

    def write_binning(self, value: int) -> None:
        self._camera.binning = value

    def metadata_binning(self) -> ParameterMetadata:
        return ParameterMetadata(choices=list(self._camera.available_binnings))

    def read_gain(self) -> float:
        return self._camera.gain

    def write_gain(self, value: float) -> None:
        self._camera.gain = value

    def read_offset(self) -> float:
        return self._camera.offset

    def write_offset(self, value: float) -> None:
        self._camera.offset = value

    def read_pixel_size(self) -> tuple:
        return tuple(self._camera.pixel_size)

    def read_resolution(self) -> tuple:
        return tuple(self._camera.resolution)

    def _acquire(self) -> np.ndarray:
        return self._camera.acquire_image()


class FMLightSource(LightSource):
    def __init__(self, light_source: Any, **kwargs: Any):
        super().__init__(name="light_source", **kwargs)
        self._light = light_source

    def read_power(self) -> float:
        return self._light.power

    def write_power(self, value: float) -> None:
        self._light.power = value

    def metadata_power(self) -> ParameterMetadata:
        low, high = self._light.power_limits
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))


def emission_filter_for(
    value: Any, edges: Dict[float, Tuple[float, float]]
) -> EmissionFilter:
    """The filter an FM class's emission value names. ``edges`` maps a band's bottom
    edge to both edges, for FM classes that know them."""
    if value is None:
        return REFLECTION
    if isinstance(value, str):
        return EmissionFilter(value)
    low = float(value)
    high = next(
        (h for bottom, (_, h) in edges.items() if math.isclose(bottom, low)), None
    )
    if high is None:
        return EmissionFilter(f"{low:.0f} nm", low=low)
    return EmissionFilter(f"{low:.0f}–{high:.0f} nm", low=low, high=float(high))


def same_emission_value(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6)
    return a == b


class FMFilterSet(FilterSet):
    def __init__(self, filter_set: Any, **kwargs: Any):
        super().__init__(name="filter_set", **kwargs)
        self._filters = filter_set

    def read_excitation_wavelength(self) -> float:
        return self._filters.excitation_wavelength

    def write_excitation_wavelength(self, value: float) -> None:
        self._filters.excitation_wavelength = value

    def metadata_excitation_wavelength(self) -> ParameterMetadata:
        return ParameterMetadata(
            choices=list(self._filters.available_excitation_wavelengths)
        )

    # The FM classes name a filter by one value: None for reflection, a label such
    # as "Fluorescence" for a multi-band filter, or the band's bottom edge in nm.

    def _emission_filters(self) -> List[Tuple[Any, EmissionFilter]]:
        edges = getattr(self._filters, "emission_bands", {})
        return [
            (value, emission_filter_for(value, edges))
            for value in self._filters.available_emission_wavelengths
        ]

    def read_emission_filter(self) -> EmissionFilter:
        value = self._filters.emission_wavelength
        filters = self._emission_filters()
        for old, found in filters:
            if same_emission_value(old, value):
                return found
        # Thermo reports its multi-band filter as the excitation wavelength.
        multi_band = [found for old, found in filters if isinstance(old, str)]
        if multi_band and value is not None:
            return multi_band[0]
        return emission_filter_for(value, {})

    def write_emission_filter(self, value: EmissionFilter) -> None:
        for old, found in self._emission_filters():
            if found == value:
                self._filters.emission_wavelength = old
                return
        raise ValueError(f"filter_set has no emission filter {value}")

    def metadata_emission_filter(self) -> ParameterMetadata:
        return ParameterMetadata(choices=[f for _, f in self._emission_filters()])


class FMObjective(Objective):
    def __init__(self, objective: Any, **kwargs: Any):
        super().__init__(name="objective", **kwargs)
        self._objective = objective

    def read_position(self) -> float:
        return self._objective.position

    def metadata_position(self) -> ParameterMetadata:
        low, high = self._objective.limits
        return ParameterMetadata(limits=RangeLimit(min=low, max=high))

    def read_state(self) -> InsertableDeviceState:
        return objective_device_state(self._objective.state)

    def read_magnification(self) -> float:
        return self._objective.magnification

    def read_numerical_aperture(self) -> float:
        return self._objective.numerical_aperture

    def read_limit_position(self) -> float:
        return self._objective.limit_position

    def write_limit_position(self, value: float) -> None:
        self._objective.limit_position = value

    # A move changes position and state; read them back so both are signalled.
    def _moved(self) -> None:
        for name in ("position", "state"):
            if name in self.parameters:
                self.parameters[name].get_value()

    def _insert(self) -> None:
        self._objective.insert()
        self._moved()

    def _retract(self) -> None:
        self._objective.retract()
        self._moved()

    def _move_absolute(self, position: float) -> None:
        self._objective.move_absolute(position)
        self._moved()

    def _move_relative(self, delta: float) -> None:
        self._objective.move_relative(delta)
        self._moved()


class FMGroup(FM):
    def __init__(self, fm: FluorescenceMicroscope, **kwargs: Any):
        super().__init__(name="fm", **kwargs)
        self._fm = fm
        self.channel_parts: List[Device] = []
        """The parts a channel sets up; read back after a sequence changes them."""

    def check_health(self) -> Optional[str]:
        """Whether the FM answers: one live read, the camera's exposure time.

        Read through the FM class rather than a cached value, so a backend that has
        stopped answering (odemis no longer running, a dropped camera) raises here
        and the device server reports the FM down. None means it answered.
        """
        _ = self._fm.camera.exposure_time
        return None

    def _acquire_channel(self, channel: Optional[Dict[str, Any]]) -> np.ndarray:
        from fibsem.fm.structures import ChannelSettings

        settings = ChannelSettings.from_dict(channel) if channel is not None else None
        data = self._fm.acquire_image(settings).data
        if settings is not None:
            # The FM class set the parts directly, past their devices: read them back
            # so each change is cached and signalled, here and on any remote client.
            for part in self.channel_parts:
                for param in part.parameters.values():
                    param.get_value()
        return data


def bind_fm_devices(
    fm: FluorescenceMicroscope, resources: Optional[Resources] = None
) -> Dict[str, Device]:
    """The FM's parts and its group, by device name, for any ``fibsem.fm`` backend."""
    resources = resources if resources is not None else Resources()
    group = FMGroup(fm, resources=resources)
    parts = [
        FMCamera(fm.camera, resources=resources),
        FMLightSource(fm.light_source, resources=resources),
        FMFilterSet(fm.filter_set, resources=resources),
    ]
    objective = FMObjective(fm.objective, resources=resources)
    group.channel_parts = parts
    return {device.name: device.connect() for device in [group, *parts, objective]}

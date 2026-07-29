import numpy as np
import logging
import time

from datetime import datetime
from typing import Dict, List, Literal, Optional, Tuple, Union
from fibsem.fm.microscope import (
    Camera,
    FilterSet,
    FluorescenceMicroscope,
    LightSource,
    ObjectiveLens,
)
from fibsem.fm.structures import FluorescenceImage
from fibsem.microscopes.odemis_microscope import add_odemis_path

add_odemis_path()

from odemis import model
from odemis.acq.acqmng import acquire
from odemis.acq.stream import FluoStream
from odemis.util import fluo

# NOTES: needed to install shapely, pylibtiff, and odemis

# position tolerance for deciding the objective is at the active/deactive position
OBJECTIVE_POSITION_ATOL = 100e-6  # m


def _frame_metadata_from_data(data) -> Optional[dict]:
    """Translate odemis DataArray metadata to the neutral frame-metadata keys
    consumed by FluorescenceMicroscope._construct_image.

    Odemis DataArrays carry metadata captured at exposure time (pixel size,
    acquisition date, exposure time), which is authoritative over a state
    snapshot taken when the frame is processed.
    """
    md = getattr(data, "metadata", None)
    if not md:
        return None

    frame_metadata = {}
    pixel_size = md.get(model.MD_PIXEL_SIZE)
    if pixel_size is not None:
        frame_metadata["pixel_size"] = tuple(pixel_size)
    acquisition_date = md.get(model.MD_ACQ_DATE)
    if acquisition_date is not None:
        frame_metadata["acquisition_date"] = datetime.fromtimestamp(
            acquisition_date
        ).isoformat()
    exposure_time = md.get(model.MD_EXP_TIME)
    if exposure_time is not None:
        frame_metadata["exposure_time"] = exposure_time
    return frame_metadata or None


class OdemisObjectiveLens(ObjectiveLens):
    """Odemis objective lens implementation for fluorescence microscopy.

    Provides control over objective lens positioning, state management, and configuration
    for Odemis-compatible fluorescence microscope systems. Implements metadata caching
    for improved performance when accessing frequently used position values.
    """

    def __init__(
        self,
        parent: "OdemisFluorescenceMicroscope",
        focuser: Optional[model.Actuator] = None,
    ):
        """Initialize the objective lens with optional focuser component.

        Args:
            parent: The parent fluorescence microscope instance
            focuser: Optional focuser actuator component. If None, will be
                    automatically detected using the 'focus' role.
        """
        super().__init__(parent)
        self.parent = parent
        if focuser is None:
            focuser = model.getComponent(role="focus")
        self._focuser = focuser
        self._lens = model.getComponent(role="lens")

        # Cache metadata on initialization
        self._metadata_cache = {}
        self._cache_metadata()

        active_position = self.active_position
        if active_position is not None:
            self._focus_position = active_position["z"]
        else:
            self._focus_position = self.position
            logging.warning(
                "Focuser has no active-position metadata; using the current "
                f"position ({self._focus_position * 1e3:.3f} mm) as the focus position."
            )
        # default user limit: no restriction beyond the hardware range
        self._limit_position = self.limits[1]

    def _cache_metadata(self):
        """Cache frequently accessed metadata for improved performance.

        Caches the active and deactive positions from the focuser metadata
        to reduce hardware communication overhead during repeated access.

        Note:
            If caching fails, the cache will be empty and methods will
            fall back to direct hardware access.
        """
        try:
            metadata = self._focuser.getMetadata()
            self._metadata_cache = {
                "active_position": metadata[model.MD_FAV_POS_ACTIVE],
                "deactive_position": metadata[model.MD_FAV_POS_DEACTIVE],
                "last_updated": time.time(),
            }
        except Exception as e:
            logging.warning(f"Failed to cache metadata: {e}")
            self._metadata_cache = {}

    @property
    def active_position(self) -> Optional[dict]:
        """Get the active (inserted) position of the objective lens.

        Uses cached metadata when available for improved performance,
        with fallback to direct hardware access if needed.

        Returns:
            The active position coordinates as a dictionary with 'z' key,
            or None if the position cannot be determined.
        """
        try:
            return self._metadata_cache["active_position"]
        except (KeyError, TypeError):
            # Fallback to direct hardware call if cache fails
            try:
                return self._focuser.getMetadata()[model.MD_FAV_POS_ACTIVE]
            except Exception as e:
                logging.error(f"Failed to get active position: {e}")
                return None

    @property
    def deactive_position(self) -> Optional[dict]:
        """Get the deactive (retracted) position of the objective lens.

        Uses cached metadata when available for improved performance,
        with fallback to direct hardware access if needed.

        Returns:
            The deactive position coordinates as a dictionary with 'z' key,
            or None if the position cannot be determined.
        """
        try:
            return self._metadata_cache["deactive_position"]
        except (KeyError, TypeError):
            # Fallback to direct hardware call if cache fails
            try:
                return self._focuser.getMetadata()[model.MD_FAV_POS_DEACTIVE]
            except Exception as e:
                logging.error(f"Failed to get deactive position: {e}")
                return None

    @property
    def magnification(self):
        """Get the current magnification of the objective lens.

        Returns:
            The magnification value as a float
        """
        return self._lens.magnification.value

    @magnification.setter
    def magnification(self, value: float):
        """Set the magnification of the objective lens.

        Args:
            value: The desired magnification value
        """
        self._lens.magnification.value = value

    @property
    def numerical_aperture(self) -> float:
        """Get the numerical aperture of the objective lens.

        Returns:
            The numerical aperture value as a float
        """
        return self._lens.numericalAperture.value

    @numerical_aperture.setter
    def numerical_aperture(self, value: float):
        """Set the numerical aperture of the objective lens.

        Args:
            value: The numerical aperture value (must be numeric)

        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Numerical aperture must be an integer or float.")
        self._lens.numericalAperture.value = value

    @property
    def position(self) -> float:
        """Get the current focus position of the objective lens.

        Returns:
            The current z-axis position of the focuser in meters
        """
        pos = self._focuser.position.value
        return pos["z"]

    def move_relative(self, delta):
        """Move the objective lens by a relative distance.

        Args:
            delta: The distance to move in meters (positive = towards sample)

        Raises:
            TypeError: If delta is not a number
        """
        if not isinstance(delta, (int, float)):
            raise TypeError("Delta must be an integer or float.")
        f = self._focuser.moveRel({"z": delta})
        f.result()

    @property
    def limits(self) -> Tuple[float, float]:
        """Get the z-axis position limits of the objective lens.

        Returns:
            A tuple of (minimum, maximum) positions in metres, from the
            focuser axis range.
        """
        rng = self._focuser.axes["z"].range
        return (rng[0], rng[1])

    def move_absolute(self, position: float):
        """Move the objective lens to an absolute focus position.

        The position is clipped to the user-defined safety limit and
        validated against the focuser axis range before moving (odemis
        raises ValueError for out-of-range moves).

        Args:
            position: The target z-axis position in meters

        Raises:
            TypeError: If position is not a number
            ValueError: If the position is outside the focuser axis range
        """
        if not isinstance(position, (int, float)):
            raise TypeError("Position must be an integer or float.")

        # clip to the user-defined safety limit
        if position > self._limit_position:
            logging.warning(
                f"Clipping position {position} to user-defined limit {self._limit_position}"
            )
            position = self._limit_position

        limits = self.limits
        if not limits[0] <= position <= limits[1]:
            raise ValueError(f"Position {position} outside focuser range {limits}")

        f = self._focuser.moveAbs({"z": position})
        f.result()

    def insert(self):
        """Move the objective lens to the active (inserted) position.

        Moves the objective lens to the predefined active position for imaging.
        The favourite active position is calibrated (Delmic), so the user
        safety limit is not applied here.
        """
        active_position = self.active_position
        if active_position is None:
            logging.warning(
                "Cannot insert objective: no active-position metadata available."
            )
            return
        f = self._focuser.moveAbs(active_position)
        f.result()

    def retract(self):
        """Move the objective lens to the deactive (retracted) position.

        Moves the objective lens to the predefined deactive position for safety.
        """
        deactive_position = self.deactive_position
        if deactive_position is None:
            logging.warning(
                "Cannot retract objective: no deactive-position metadata available."
            )
            return
        f = self._focuser.moveAbs(deactive_position)
        f.result()

    @property
    def state(self) -> Literal["Inserted", "Retracted", "Busy", "Error", "Other"]:
        """Get the current state of the objective lens.

        Determines the lens state by comparing the current position with the
        favourite active/deactive positions (within OBJECTIVE_POSITION_ATOL).

        Returns:
            'Inserted' or 'Retracted' when at the corresponding favourite
            position, 'Other' when in between or if the favourite positions
            are unavailable, and 'Error' if the state cannot be read.
            Values match the FluorescenceMicroscope base contract
            (consumers compare against 'Inserted'/'Retracted').
        """
        try:
            position = self._focuser.position.value["z"]
            active_position = self.active_position
            deactive_position = self.deactive_position

            if active_position is None or deactive_position is None:
                return "Other"

            if position > active_position["z"] - OBJECTIVE_POSITION_ATOL:
                return "Inserted"
            if position < deactive_position["z"] + OBJECTIVE_POSITION_ATOL:
                return "Retracted"
            return "Other"
        except Exception as e:
            logging.error(f"Failed to get objective lens state: {e}")
            return "Error"


class OdemisCamera(Camera):
    """Odemis camera implementation for fluorescence microscopy.

    Provides control over camera settings and image acquisition for Odemis-compatible
    fluorescence microscope systems. Manages exposure time, binning, and image capture
    through the Odemis streaming interface.
    """

    def __init__(
        self,
        parent: "OdemisFluorescenceMicroscope",
        camera: Optional[model.DigitalCamera] = None,
    ):
        """Initialize the camera with optional camera component.

        Args:
            parent: The parent fluorescence microscope instance
            camera: Optional camera component. If None, will be automatically
                   detected using the 'ccd' role.
        """
        super().__init__(parent)
        self.parent = parent
        self._stream: FluoStream = self.parent._stream
        if camera is None:
            camera: model.DigitalCamera = model.getComponent(role="ccd")
        self._camera = camera
        camera_md = self._camera.getMetadata()
        # fallback values only: pixel_size/resolution are read live (see properties)
        self._resolution = tuple(self._camera.resolution.value)  # (width, height)

        pixel_size = camera_md.get(model.MD_PIXEL_SIZE)  # (x, y) sample-plane
        if pixel_size is None:
            # no lens/magnification wiring in the microscope file
            pixel_size = self._camera.pixelSize.value  # sensor pixel size
            logging.warning(
                "Camera metadata has no MD_PIXEL_SIZE; falling back to the "
                f"sensor pixel size {pixel_size} - image scale will not "
                "account for magnification."
            )
        self._pixel_size = tuple(pixel_size)

        # not all cameras publish a baseline (only some drivers set MD_BASELINE)
        self._offset = camera_md.get(model.MD_BASELINE, 0)
        self._gain_warning_logged = False

    # other attributes: depthOfField, readoutRate, pointSpreadFunctionSize

    def acquire_image(self) -> np.ndarray:
        """Acquire a single image from the camera.

        While live acquisition is running (the stream is active), a fresh
        frame is taken from the camera dataflow without stopping the stream
        (the light stays on). Otherwise a full acquisition is run through
        the Odemis acquisition manager.

        Returns:
            A numpy array containing the image data

        Raises:
            RuntimeError: If the acquisition fails or returns no data
        """
        if self._stream.is_active.value:
            # asap=False guarantees the frame is acquired after this call
            return self._camera.data.get(asap=False)

        da: List[model.DataArray]
        f = acquire([self._stream])
        da, err = f.result()
        if err:
            raise RuntimeError(f"Error acquiring image: {err}")
        if not da:
            raise RuntimeError("Acquisition returned no data.")
        return da[0]  # model.DataArray -> np.ndarray

    def _start_fast_acquisition(self):
        """Continuous live acquisition via the odemis camera dataflow.

        Activates the fluorescence stream once (light on, filters set,
        camera in continuous mode) and subscribes a listener to the camera
        dataflow, so frames are pushed at the exposure-limited rate instead
        of paying a full acquisition round-trip (with light toggling) per
        frame. Blocks until stop_acquisition() sets the stop event; called
        by the base _acquisition_worker from the acquisition thread.

        The stream is always deactivated (light off) on exit, even if the
        frame handler or unsubscribe fails.
        """

        def on_frame(dataflow, data):
            try:
                # emits acquisition_signal, rate-limited by the base class
                self.parent._construct_image(data)
            except Exception as e:
                logging.error(f"Error handling live frame: {e}")

        stream = self._stream
        try:
            stream.is_active.value = True  # light on, filters set, streaming
            self._camera.data.subscribe(on_frame)
            self.parent._stop_acquisition_event.wait()
        finally:
            try:
                self._camera.data.unsubscribe(on_frame)
            except Exception as e:
                logging.warning(f"Failed to unsubscribe live-view listener: {e}")
            try:
                stream.is_active.value = False  # light off, camera stopped
            except Exception as e:
                logging.error(f"Failed to deactivate stream after live view: {e}")

    @property
    def exposure_time(self) -> float:
        """Get the current exposure time of the camera.

        Returns:
            The exposure time in seconds
        """
        return self._camera.exposureTime.value

    @exposure_time.setter
    def exposure_time(self, value: float):
        """Set the exposure time of the camera.

        Args:
            value: The exposure time in seconds (must be numeric)

        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Exposure time must be an integer or float.")
        self._camera.exposureTime.value = value

    @property
    def binning(self) -> int:
        """Get the current binning setting of the camera.

        Returns:
            The binning value (assumes symmetric binning, returns x-axis value)
        """
        return self._camera.binning.value[0]  # Assuming binning is a tuple (x, y)

    @binning.setter
    def binning(self, value: int):
        """Set the binning of the camera.

        Args:
            value: The binning value (must be 1, 2, 4, or 8)

        Raises:
            ValueError: If the binning value is not supported

        Note:
            Sets symmetric binning (same value for both x and y axes)
        """
        if value not in self.available_binnings:
            raise ValueError(
                f"Binning must be one of {self.available_binnings}, got {value}"
            )
        self._camera.binning.value = (value, value)

    @property
    def available_binnings(self) -> Tuple[int, ...]:
        """Get the supported binning values from the camera binning VA.

        Uses the VA choices when enumerated; otherwise powers of two within
        the VA range.
        """
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

    @property
    def exposure_time_limits(self) -> Tuple[float, float]:
        """Get the valid exposure time range from the camera VA.

        Returns:
            A tuple of (minimum, maximum) exposure times in seconds
        """
        rng = self._camera.exposureTime.range
        return (rng[0], rng[1])

    @property
    def gain(self) -> Optional[float]:
        """Get the camera gain, or None when the camera has no gain control.

        Only reports values that reflect hardware state: if the camera
        exposes no gain VA, None is returned (and recorded in metadata)
        rather than echoing back a software-only value.
        """
        if model.hasVA(self._camera, "gain"):
            return self._camera.gain.value
        return None

    @gain.setter
    def gain(self, value: float) -> None:
        """Set the camera gain when the camera supports it.

        Ignored (with a one-time warning) on cameras without a gain VA, so
        applying a saved channel with a gain value does not fail.
        """
        if model.hasVA(self._camera, "gain"):
            self._camera.gain.value = value
            return
        if not self._gain_warning_logged:
            logging.warning("Camera has no gain control; ignoring gain settings.")
            self._gain_warning_logged = True

    @property
    def pixel_size(self) -> Tuple[float, float]:
        """Get the sample-plane pixel size in metres, read live from camera metadata.

        The odemis backend computes MD_PIXEL_SIZE = sensor pixel size x binning
        / magnification and republishes it whenever binning or magnification
        change, so no binning scaling is applied here (unlike the base class).
        """
        try:
            pixel_size = self._camera.getMetadata().get(model.MD_PIXEL_SIZE)
        except Exception as e:
            logging.warning(
                f"Failed to read camera metadata, using cached pixel size: {e}"
            )
            return tuple(self._pixel_size)
        if pixel_size is None:
            return tuple(self._pixel_size)  # missing key already warned at init
        return tuple(pixel_size)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get the current readout resolution in pixels, read live from the camera.

        The odemis resolution VA already accounts for binning, so no binning
        scaling is applied here (unlike the base class).
        """
        return tuple(self._camera.resolution.value)

    @property
    def offset(self) -> float:
        """Get the baseline offset of the camera.

        Read-only: the offset comes from the camera MD_BASELINE metadata;
        the base-class setter is intentionally not supported here.

        Returns:
            The camera's baseline offset value from metadata
        """
        return self._offset


class OdemisLightSource(LightSource):
    """Odemis light source implementation for fluorescence microscopy.

    Provides control over light source power and intensity for Odemis-compatible
    fluorescence microscope systems. Manages power control through the fluorescence
    stream interface.
    """

    def __init__(
        self,
        parent: "OdemisFluorescenceMicroscope",
        light_source: Optional[model.Emitter] = None,
    ):
        """Initialize the light source with optional emitter component.

        Args:
            parent: The parent fluorescence microscope instance
            light_source: Optional light source emitter component. If None,
                         will be automatically detected using the 'light' role.
        """
        super().__init__(parent)
        self.parent = parent
        self._stream = self.parent._stream
        if light_source is None:
            light_source = model.getComponent(role="light")
        self._light_source = light_source

    @property
    def power(self) -> float:
        """Get the current power of the light source as a fraction (0-1) of maximum.

        The odemis stream power is in watts; it is normalised against the
        stream power range so ChannelSettings.power keeps identical semantics
        across drivers (fraction of maximum power, displayed as % in the UI).

        Returns:
            The power level as a fraction of maximum power (0.0-1.0)
        """
        max_power = self._stream.power.range[1]
        if max_power <= 0:
            return 0.0
        return self._stream.power.value / max_power

    @power.setter
    def power(self, value: float):
        """Set the power of the light source as a fraction (0-1) of maximum.

        Args:
            value: The power level as a fraction of maximum power (0.0-1.0).
                   Values outside [0, 1] are clipped with a warning.

        Raises:
            TypeError: If the value is not a number
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Power must be an integer or float.")
        if not 0.0 <= value <= 1.0:
            logging.warning(f"Power fraction {value} outside [0, 1], clipping.")
            value = min(max(value, 0.0), 1.0)
        max_power = self._stream.power.range[1]
        self._stream.power.value = value * max_power

    @property
    def power_limits(self) -> Tuple[float, float]:
        """Get the valid power range for the light source.

        Power is normalised to a fraction of maximum across all drivers;
        the underlying watt range remains available on the stream power VA
        for diagnostics.

        Returns:
            A tuple of (minimum, maximum) power fractions (0.0, 1.0)
        """
        return (0.0, 1.0)

    # spectra property?


# NOTE: we also need to determine which filter set is active, as the light source power returns the power for each emitter...
# power, excitation, and emission should represent the 'currentlly selected' rather than the full available?
# QUERY: should excitiation be a property of the filter set or the light source?


class OdemisFilterSet(FilterSet):
    """Odemis filter set implementation for fluorescence microscopy.

    Manages excitation and emission wavelength selection for Odemis-compatible
    fluorescence microscope systems. Handles wavelength conversion between
    nanometers (external interface) and meters (internal Odemis units).

    Note:
        Wavelengths are stored internally in meters but exposed in nanometers
        for user convenience. None values represent reflection/pass-through mode.
    """

    def __init__(self, parent: Optional["OdemisFluorescenceMicroscope"] = None):
        """Initialize the filter set with parent microscope.

        Args:
            parent: The parent fluorescence microscope instance
        """
        super().__init__(parent)
        self.parent = parent
        self._stream: FluoStream = self.parent._stream

    # NOTE on internal representation:
    # images are acquired via the public stream interface, as it contains the necessary components

    # excitation wavelength is 5D tuple: (bottom 99%, bottom 25%, centre, top 25%, top 99%)
    # emission wavelength is a 2D tuple: (bottom, top)
    # to set the exciation/emission wavelngth you need to set this tuple

    # externally, we pass in the centre wavelength for excitation and the bottom wavelength for emission
    # this may change to use a str or enum for the excitation and emission wavelengths in the future, to be more structured.
    # None -> reflection

    @property
    def _excitation_choices_by_nm(self) -> Dict[float, tuple]:
        """Map centre wavelength (nm) -> excitation choice (5-tuple, in metres).

        Built in a single pass over the stream choices so selection and
        assignment always refer to the same choice object (the underlying
        VA choices are an unordered set).
        """
        return {c[2] * 1e9: c for c in self._stream.excitation.choices}

    @property
    def _emission_choices_by_nm(self) -> Dict[Optional[float], Union[tuple, str]]:
        """Map bottom wavelength (nm) -> emission choice (2-tuple, in metres).

        The pass-through (reflection) choice is keyed as None.
        """
        mapping: Dict[Optional[float], Union[tuple, str]] = {}
        for c in self._stream.emission.choices:
            # special case for pass-through (reflection -> None)
            if isinstance(c, str) and c == model.BAND_PASS_THROUGH:
                mapping[None] = c
                continue
            mapping[c[0] * 1e9] = c  # convert from m to nm
        return mapping

    @property
    def available_excitation_wavelengths(self) -> Tuple[float, ...]:
        """Get the available excitation wavelengths for the filter set.

        Returns:
            A tuple of available centre excitation wavelengths in nanometers
            (converted from Odemis internal metre units)
        """
        return tuple(self._excitation_choices_by_nm.keys())

    @property
    def available_emission_wavelengths(self) -> Tuple[Optional[float], ...]:
        """Get the available emission wavelengths for the filter set.

        Returns:
            A tuple of available emission wavelengths in nanometers.
            Includes None for reflection/pass-through mode.

        Note:
            Returns the bottom wavelength of each emission filter range.
        """
        return tuple(self._emission_choices_by_nm.keys())

    @property
    def excitation_wavelength(self) -> float:
        """Get the current excitation wavelength of the filter set.

        Returns:
            The center excitation wavelength in nanometers
            (converted from Odemis internal meter units)
        """
        return self._stream.excitation.value[2] * 1e9  # centre wavelength (nm)

    @excitation_wavelength.setter
    def excitation_wavelength(self, value: float) -> None:
        """Set the excitation wavelength of the filter set.

        Args:
            value: The desired excitation wavelength in nanometers

        Raises:
            TypeError: If the value is not a number
            ValueError: If no excitation wavelengths are available

        Note:
            Automatically selects the closest available wavelength if an
            exact match is not found.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("Excitation wavelength must be an integer or float.")

        choices_by_nm = self._excitation_choices_by_nm
        if not choices_by_nm:
            raise ValueError("No excitation wavelengths available.")

        closest_nm = min(choices_by_nm, key=lambda nm: abs(nm - value))
        choice = choices_by_nm[closest_nm]
        logging.info(
            f"Setting excitation wavelength to {closest_nm:.0f} nm "
            f"(requested: {value} nm, band: {choice})"
        )
        self._stream.excitation.value = choice

    @property
    def emission_wavelength(self) -> Optional[float]:
        """Get the current emission wavelength of the filter set.

        Returns:
            The bottom emission wavelength in nanometers, or None for
            reflection/pass-through mode

        Note:
            Converts from Odemis internal meter units to nanometers.
        """
        value = self._stream.emission.value
        if isinstance(value, str) and value == model.BAND_PASS_THROUGH:
            return None  # pass-through means reflection, so we return None
        return self._stream.emission.value[0] * 1e9  # convert from m to nm

    @emission_wavelength.setter
    def emission_wavelength(self, value: Optional[float]) -> None:
        """Set the emission wavelength of the filter set.

        Args:
            value: The desired emission wavelength in nanometers, or None
                  for reflection/pass-through mode

        Raises:
            TypeError: If the value is not a number or None
            ValueError: If the requested wavelength or pass-through mode
                       is not available

        Note:
            Setting to None enables reflection/pass-through mode.
            Automatically selects the closest available wavelength.
            Performance note: Setting emission wavelength can be slow.
        """
        choices_by_nm = self._emission_choices_by_nm

        # None means reflection, so we set the emission to pass-through (reflection)
        if value is None:
            if None not in choices_by_nm:
                raise ValueError(
                    f"Pass-through (reflection) is not available in the current "
                    f"filter set. Available: {tuple(choices_by_nm.keys())}"
                )
            self._stream.emission.value = choices_by_nm[None]
            return

        if isinstance(value, str):
            # TFS-style channel settings use 'Fluorescence' for multi-band
            # emission; pick the band matching the current excitation
            bands = {c for nm, c in choices_by_nm.items() if nm is not None}
            if not bands:
                raise ValueError("No emission bands available for fluorescence mode.")
            choice = fluo.get_one_band_em(bands, self._stream.excitation.value)
            if self._stream.emission.value != choice:
                logging.info(
                    f"Mapping emission '{value}' to band {choice} for the "
                    f"current excitation"
                )
                self._stream.emission.value = choice
            return

        if not isinstance(value, (int, float)):
            raise TypeError("Emission wavelength must be a number, str, or None.")

        numeric = {nm: c for nm, c in choices_by_nm.items() if nm is not None}
        if not numeric:
            raise ValueError("No emission wavelengths available.")

        closest_nm = min(numeric, key=lambda nm: abs(nm - value))
        choice = numeric[closest_nm]
        if self._stream.emission.value == choice:
            return  # setting the emission filter is slow, skip if unchanged
        logging.info(
            f"Setting emission wavelength to {closest_nm:.0f} nm "
            f"(requested: {value} nm, band: {choice})"
        )
        self._stream.emission.value = choice


class OdemisFluorescenceMicroscope(FluorescenceMicroscope):
    """Odemis fluorescence microscope implementation.

    Provides integrated control over Odemis-compatible fluorescence microscope systems.
    Manages objective lens, camera, light source, and filter set components through
    the Odemis framework using a unified fluorescence stream interface.

    Attributes:
        objective: The objective lens controller
        camera: The camera controller
        light_source: The light source controller
        filter_set: The filter set controller

    Note:
        Requires Odemis to be properly installed and configured with the
        appropriate hardware components (ccd, light, filter, focus).
    """

    objective: OdemisObjectiveLens
    camera: OdemisCamera
    light_source: OdemisLightSource
    filter_set: OdemisFilterSet

    def __init__(self, parent: "FibsemMicroscope"):
        """Initialize the Odemis fluorescence microscope.

        Args:
            parent: The parent FibsemMicroscope instance

        Note:
            Automatically detects and configures hardware components using
            Odemis role-based component discovery (ccd, light, filter, focus).
        """
        super().__init__()

        self.parent = parent
        camera = model.getComponent(role="ccd")
        light_source = model.getComponent(role="light")
        light_filter = model.getComponent(role="filter")
        focuser = model.getComponent(role="focus")

        self._stream = FluoStream(
            name="fm-stream",
            detector=camera,
            dataflow=camera.data,
            emitter=light_source,
            em_filter=light_filter,
            focuser=focuser,
        )

        self.objective = OdemisObjectiveLens(self, focuser=focuser)
        self.camera = OdemisCamera(self, camera=camera)
        self.light_source = OdemisLightSource(self, light_source=light_source)
        self.filter_set = OdemisFilterSet(self)

    def _construct_image(
        self, data: np.ndarray, frame_metadata: Optional[dict] = None
    ) -> FluorescenceImage:
        """Construct a FluorescenceImage, preferring per-frame odemis metadata.

        Odemis DataArrays carry authoritative metadata captured at exposure
        time (pixel size, acquisition date, exposure time); when present it
        overrides the state snapshot, so frames stay correctly stamped even
        if settings change during live view.
        """
        if frame_metadata is None:
            frame_metadata = _frame_metadata_from_data(data)
        return super()._construct_image(data, frame_metadata=frame_metadata)

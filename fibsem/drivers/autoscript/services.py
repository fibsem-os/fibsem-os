"""ThermoFisher's services.

`AutoScriptMilling` mills on AutoScript's ``connection.patterning``: the per-pattern
application file, the Serial mode cross-sections need, and the patterning state read
in the milling view, under the imaging channel's lock.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from difflib import get_close_matches
from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Type

import numpy as np
from packaging.version import parse as parse_version
from skimage import transform

from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.autoscript import microscope as autoscript
from fibsem.drivers.autoscript.microscope import TFS_SCAN_DIRECTIONS
from fibsem.services.milling import Milling, bind_milling
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
    BeamMillingSettings,
    CrossSectionPattern,
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemLineSettings,
    FibsemMillingSettings,
    FibsemPatternSettings,
    FibsemPolygonSettings,
    FibsemRectangleSettings,
    MillingState,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fibsem.drivers.autoscript.connection import AutoScriptConnection
    from fibsem.drivers.autoscript.microscope import ThermoMicroscope

_PATTERNING_MODES = ("Serial", "Parallel")


def match_application_file(
    application_file: str, application_files: List[str], strict: bool = True
) -> str:
    """The application file to use for `application_file`, from those available.

    With `strict`, a name that is not available raises; otherwise the closest
    available name is used.

    Raises:
        ValueError: If no available application file matches.
    """
    if application_file not in application_files:
        if strict:
            raise ValueError(
                f"Application file {application_file} not available. Available files: {application_files}"
            )
        closest_match = get_close_matches(application_file, application_files, n=1)
        if not closest_match:
            raise ValueError(
                f"Application file {application_file} not available. Available files: {application_files}"
            )
        application_file = str(closest_match[0])

    return application_file


class AutoScriptMilling(Milling):
    """ThermoFisher milling, on ``connection.patterning``."""

    parent: ThermoMicroscope

    settings_class = BeamMillingSettings
    scan_directions = tuple(TFS_SCAN_DIRECTIONS)

    def __init__(
        self, connection: AutoScriptConnection, name: str = "milling", **kwargs: Any
    ):
        super().__init__(name, **kwargs)
        self.connection = connection
        # The patterns drawn since the last clear: their times are the estimate.
        self._patterns: List = []
        # The recipe's application file, which each pattern's own is reset to after
        # it is drawn, and the one set last.
        self._default_application_file = "Si"
        self._current_application_file = self._default_application_file

    @property
    def _sdk(self) -> Any:
        return self.connection.sdk

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self._sdk.patterning.list_all_application_files()
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: BeamMillingSettings, name: Optional[str]) -> None:
        # the milling view, the application file, the patterning mode, then hfw,
        # voltage and current on the milling beam
        microscope = self.parent
        channel = settings.milling_channel
        microscope.milling_channel = channel
        self.connection.set_channel(channel)
        self._sdk.patterning.set_default_beam_type(channel.value)
        self._set_application_file(settings.application_file, default=True)
        self._set_patterning_mode(settings.patterning_mode)
        self._clear()  # clear any existing patterns
        beam = self.beam(channel)
        beam.hfw.write_through(settings.hfw)
        # voltage before current: the available ion currents are calibrated per voltage
        beam.voltage.write_through(settings.milling_voltage)
        beam.current.write_through(settings.milling_current)

    def _set_application_file(
        self, application_file: str, default: bool = False, strict: bool = True
    ) -> str:
        """Set the application file new patterns are made with: the closest match
        to *application_file* when not *strict*. AutoScript needs a valid one set
        before a pattern is made."""
        application_file = match_application_file(
            application_file,
            self._sdk.patterning.list_all_application_files(),
            strict,
        )
        self._sdk.patterning.set_default_application_file(application_file)
        self._current_application_file = application_file
        if default:
            self._default_application_file = application_file
        logging.debug(
            {
                "msg": "set_application_file",
                "application_file": application_file,
                "default": default,
            }
        )
        return application_file

    def _set_patterning_mode(self, mode: str) -> str:
        if mode not in _PATTERNING_MODES:
            raise ValueError(
                f"Patterning mode {mode} not supported. Supported modes: Serial, Parallel"
            )
        self._sdk.patterning.mode = mode
        logging.debug({"msg": "set_patterning_mode", "mode": mode})
        return mode

    @classmethod
    def supported_pattern_types(cls) -> Tuple[Type[FibsemPatternSettings], ...]:
        types = super().supported_pattern_types()
        version = autoscript.AUTOSCRIPT_VERSION
        if version is not None and version < parse_version("4.12"):
            # polygon patterning came in AutoScript 4.12
            types = tuple(t for t in types if t is not FibsemPolygonSettings)
        return types

    def _draw(self, pattern: FibsemPatternSettings, hook: str) -> None:
        # Each pattern starts on the recipe's application file, and one a pattern
        # needs for itself does not outlast it.
        self._set_application_file(self._default_application_file)
        try:
            super()._draw(pattern, hook)
        finally:
            self._set_application_file(self._default_application_file)

    def _draw_rectangle(
        self,
        pattern_settings: FibsemRectangleSettings,
    ):
        """
        Draws a rectangle pattern using the current ion beam.

        Args:
            pattern_settings (FibsemRectangleSettings): the settings for the pattern to draw.

        Returns:
            Pattern: the created pattern.

        Raises:
            AutoscriptError: if an error occurs while creating the pattern.
        """

        # get patterning api
        patterning_api = self._sdk.patterning
        if pattern_settings.cross_section is CrossSectionPattern.RegularCrossSection:
            create_pattern_function = patterning_api.create_regular_cross_section
            self._set_patterning_mode(
                "Serial"
            )  # parallel mode not supported for regular cross section
            self._set_application_file("Si-multipass", strict=False)
        elif pattern_settings.cross_section is CrossSectionPattern.CleaningCrossSection:
            create_pattern_function = patterning_api.create_cleaning_cross_section
            self._set_patterning_mode(
                "Serial"
            )  # parallel mode not supported for cleaning cross section
            self._set_application_file("Si-ccs", strict=False)
        else:
            create_pattern_function = patterning_api.create_rectangle
            # ensure a rectangle-compatible application file is set; the stage's
            # application file may be a cross-section-only file (e.g. Si-ccs) that
            # AutoScript rejects for a plain Rectangle pattern.
            self._set_application_file("Si", strict=False)

        # create pattern
        pattern = create_pattern_function(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        if not np.isclose(pattern_settings.time, 0.0):
            logging.debug(f"Setting pattern time to {pattern_settings.time}.")
            pattern.time = pattern_settings.time

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        # set scan direction
        available_scan_directions = TFS_SCAN_DIRECTIONS

        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(
                f"Scan direction {pattern_settings.scan_direction} not supported. Using TopToBottom instead."
            )
            logging.warning(
                f"Supported scan directions are: {available_scan_directions}"
            )

        # set passes
        if pattern_settings.passes:  # not zero
            if isinstance(pattern, autoscript.RegularCrossSectionPattern):
                pattern.multi_scan_pass_count = pattern_settings.passes
                pattern.scan_method = 1  # multi scan
            else:
                pattern.dwell_time = pattern.dwell_time * (
                    pattern.pass_count / pattern_settings.passes
                )

                # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
                # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
                # NB: the current must be set before doing this, otherwise it will be out of range

        logging.debug(
            {"msg": "draw_rectangle", "pattern_settings": pattern_settings.to_dict()}
        )

        self._patterns.append(pattern)

        return pattern

    def _draw_line(self, pattern_settings: FibsemLineSettings):
        """
        Draws a line pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemLineSettings): A data class object specifying the pattern parameters,
                including the start and end points, and the depth of the pattern.

        Returns:
            LinePattern: A line pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """
        pattern = self._sdk.patterning.create_line(
            start_x=pattern_settings.start_x,
            start_y=pattern_settings.start_y,
            end_x=pattern_settings.end_x,
            end_y=pattern_settings.end_y,
            depth=pattern_settings.depth,
        )
        logging.debug(
            {"msg": "draw_line", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    def _draw_circle(self, pattern_settings: FibsemCircleSettings):
        """
        Draws a circle pattern on the current imaging view of the microscope.

        Args:
            pattern_settings (FibsemCircleSettings): A data class object specifying the pattern parameters,
                including the centre point, radius and depth of the pattern.

        Returns:
            CirclePattern: A circle pattern object, which can be used to configure further properties or to add the
                pattern to the milling list.

        Raises:
            autoscript.exceptions.InvalidArgumentException: if any of the pattern parameters are invalid.
        """

        outer_diameter = 2 * pattern_settings.radius
        inner_diameter = 0
        if pattern_settings.thickness != 0:
            inner_diameter = outer_diameter - 2 * pattern_settings.thickness

        fallback_application_file = "Si"
        try:
            pattern = self._sdk.patterning.create_circle(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                outer_diameter=outer_diameter,
                inner_diameter=inner_diameter,
                depth=pattern_settings.depth,
            )
        except Exception:
            if self._current_application_file == fallback_application_file:
                # No need to try again with the same application file
                raise
            logging.warning(
                "Failed to draw circle pattern, falling back on application file %s",
                fallback_application_file,
            )
            self._set_application_file(fallback_application_file)
            pattern = self._sdk.patterning.create_circle(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                outer_diameter=outer_diameter,
                inner_diameter=inner_diameter,
                depth=pattern_settings.depth,
            )
        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug(
            {"msg": "draw_circle", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    def _draw_bitmap(self, pattern_settings: FibsemBitmapSettings):
        # Avoid modifying the original pattern_settings object
        pattern_settings = deepcopy(pattern_settings)

        if pattern_settings.bitmap is None:
            logging.warning("Bitmap pattern will be skipped as no bitmap has been set")
            return None

        # Get bitmap from pattern settings
        bitmap_pattern = autoscript.BitmapPatternDefinition()

        if pattern_settings.flip_y:
            pattern_settings.bitmap = np.flip(pattern_settings.bitmap, axis=0)

        points = pattern_settings.bitmap

        fallback_application_file = "Si"
        try:
            if pattern_settings.interpolate is not None:
                points = self._resize_bitmap_to_pattern(pattern_settings)
            bitmap_pattern.points = points
            pattern = self._sdk.patterning.create_bitmap(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
                bitmap_pattern_definition=bitmap_pattern,
            )
        except Exception:
            if self._current_application_file == fallback_application_file:
                # No need to try again with the same application file
                raise
            logging.warning(
                "Failed to draw bitmap pattern, falling back on application file %s",
                fallback_application_file,
            )
            self._set_application_file(fallback_application_file)

            if pattern_settings.interpolate is not None:
                points = self._resize_bitmap_to_pattern(pattern_settings)
            bitmap_pattern.points = points
            pattern = self._sdk.patterning.create_bitmap(
                center_x=pattern_settings.centre_x,
                center_y=pattern_settings.centre_y,
                width=pattern_settings.width,
                height=pattern_settings.height,
                depth=pattern_settings.depth,
                bitmap_pattern_definition=bitmap_pattern,
            )

        if not np.isclose(pattern_settings.time, 0.0):
            logging.debug("Setting pattern time to %f", pattern_settings.time)
            pattern.time = pattern_settings.time

        # set pattern rotation
        pattern.rotation = pattern_settings.rotation

        # set exclusion
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        # set scan direction
        available_scan_directions = TFS_SCAN_DIRECTIONS

        if pattern_settings.scan_direction in available_scan_directions:
            pattern.scan_direction = pattern_settings.scan_direction
        else:
            pattern.scan_direction = "TopToBottom"
            logging.warning(
                "Scan direction %s not supported. Using TopToBottom instead.",
                pattern_settings.scan_direction,
            )
            logging.warning(
                "Supported scan directions are: %s", str(available_scan_directions)
            )

        # set passes
        if pattern_settings.passes:  # not zero
            pattern.dwell_time = pattern.dwell_time * (
                pattern.pass_count / pattern_settings.passes
            )

            # NB: passes, time, dwell time are all interlinked, therefore can only adjust passes indirectly
            # if we adjust passes directly, it just reduces the total time to compensate, rather than increasing the dwell_time
            # NB: the current must be set before doing this, otherwise it will be out of range

        logging.debug(
            {
                "msg": "draw_bitmap_pattern",
                "pattern_settings": pattern_settings.to_dict(),
            }
        )
        self._patterns.append(pattern)
        return pattern

    def _resize_bitmap_to_pattern(
        self, pattern_settings: FibsemBitmapSettings
    ) -> NDArray[np.float64 | np.uint8]:
        points = pattern_settings.bitmap

        if points is None:
            raise ValueError(
                "Unable to resize bitmap as FibsemBitmapSettings.bitmap is None"
            )

        # Get pitch to calculate expected pixel size
        rectangle = self._sdk.patterning.create_rectangle(
            center_x=pattern_settings.centre_x,
            center_y=pattern_settings.centre_y,
            width=pattern_settings.width,
            height=pattern_settings.height,
            depth=pattern_settings.depth,
        )

        new_shape = (
            int(round(pattern_settings.height / rectangle.pitch_y)),
            int(round(pattern_settings.width / rectangle.pitch_x)),
        )

        # Disable after calculations just in case values are cleared
        rectangle.enabled = False

        if pattern_settings.interpolate == "bicubic":
            order = 3
        elif pattern_settings.interpolate == "bilinear":
            order = 1
        elif pattern_settings.interpolate == "nearest":
            order = 0
        else:
            raise ValueError(
                f"Invalid interpolate option '{pattern_settings.interpolate}'"
            )

        resized_points = np.empty((*new_shape, 2), dtype=object)

        resized_points[:, :, 0] = transform.resize(
            points[:, :, 0]
            .reshape(points.shape[0], points.shape[1])
            .astype(np.float64),
            output_shape=new_shape,
            order=order,
            preserve_range=True,
        ).astype(np.float64)
        resized_points[:, :, 1] = transform.resize(
            points[:, :, 1].reshape(points.shape[0], points.shape[1]).astype(np.uint8),
            output_shape=new_shape,
            order=0,
            preserve_range=True,
        ).astype(np.uint8)

        return resized_points

    def _draw_polygon(self, pattern_settings: FibsemPolygonSettings) -> None:
        """Draw a polygon pattern on the current imaging view of the microscope."""

        pattern = self._sdk.patterning.create_polygon(
            pattern_settings.vertices, depth=pattern_settings.depth
        )
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug(
            {"msg": "draw_polygon", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    def read_state(self) -> MillingState:
        with self.connection.lock:
            self.connection.set_channel(self.parent.milling_channel)
            return MillingState[self._sdk.patterning.state.upper()]

    def _start(self) -> None:
        with self.connection.lock:
            if self.read_state() is MillingState.IDLE:
                self._sdk.patterning.start()
                logging.info("Starting milling...")

    def _stop(self) -> None:
        with self.connection.lock:
            if self.read_state() in ACTIVE_MILLING_STATES:
                logging.info("Stopping milling...")
                self._sdk.patterning.stop()
                logging.info("Milling stopped.")

    def _pause(self) -> None:
        with self.connection.lock:
            if self.read_state() == MillingState.RUNNING:
                logging.info("Pausing milling...")
                self._sdk.patterning.pause()
                logging.info("Milling paused.")

    def _resume(self) -> None:
        with self.connection.lock:
            if self.read_state() == MillingState.PAUSED:
                logging.info("Resuming milling...")
                self._sdk.patterning.resume()
                logging.info("Milling resumed.")

    def _estimate(self) -> float:
        return sum(pattern.time for pattern in self._patterns)

    def _clear(self) -> None:
        self._sdk.patterning.clear_patterns()
        self._patterns = []

    def _restore(self) -> None:
        # The patterning mode persists in xT, so a stage left in Parallel would carry
        # over to the next one unless reset.
        self._set_patterning_mode("Serial")


def bind_autoscript_milling(
    microscope: ThermoMicroscope,
) -> Optional[AutoScriptMilling]:
    """Build ``milling`` for a connected Thermo microscope whose beams are built."""
    return bind_milling(
        AutoScriptMilling, microscope, connection=microscope._connection
    )

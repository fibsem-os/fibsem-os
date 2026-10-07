"""ThermoFisher's services.

`AutoScriptMilling` mills on AutoScript's ``connection.patterning``: the per-pattern
application file, the Serial mode cross-sections need, and the patterning state read
in the milling view, under the imaging channel's lock.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import numpy as np
from packaging.version import parse as parse_version
from skimage import transform

from fibsem.devices.core import ParameterMetadata
from fibsem.drivers.autoscript import microscope as autoscript
from fibsem.drivers.autoscript.microscope import TFS_SCAN_DIRECTIONS
from fibsem.services.milling import Milling, bind_milling
from fibsem.structures import (
    ACTIVE_MILLING_STATES,
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
from fibsem.util.application_file import match_application_file

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from fibsem.drivers.autoscript.microscope import ThermoMicroscope

_PATTERNING_MODES = ("Serial", "Parallel")


class AutoScriptMilling(Milling):
    """ThermoFisher milling, on ``connection.patterning``."""

    parent: ThermoMicroscope

    setting_names = (
        "milling_channel",
        "hfw",
        "milling_current",
        "milling_voltage",
        "application_file",
        "patterning_mode",
    )
    scan_directions = tuple(TFS_SCAN_DIRECTIONS)

    def __init__(self, name: str = "milling", **kwargs: Any):
        super().__init__(name, **kwargs)
        # The patterns drawn since the last clear: their times are the estimate.
        self._patterns: List = []
        # The recipe's application file, which each pattern's own is reset to after
        # it is drawn, and the one set last.
        self._default_application_file = "Si"
        self._current_application_file = self._default_application_file

    @property
    def _connection(self) -> Any:
        return self.parent.connection

    def _setting_metadata(self, name: str) -> ParameterMetadata:
        if name == "application_file":
            files = self._connection.patterning.list_all_application_files()
            return ParameterMetadata(choices=tuple(files))
        return super()._setting_metadata(name)

    def _setup(self, settings: FibsemMillingSettings, name: Optional[str]) -> None:
        # the milling view, the application file, the patterning mode, then hfw,
        # voltage and current on the milling beam
        microscope = self.parent
        channel = settings.milling_channel
        microscope.milling_channel = channel
        microscope.set_channel(channel)
        self._connection.patterning.set_default_beam_type(channel.value)
        self._set_application_file(settings.application_file, default=True)
        self._set_patterning_mode(settings.patterning_mode)
        self._clear()  # clear any existing patterns
        microscope.set_field_of_view(hfw=settings.hfw, beam_type=channel)
        # voltage before current: the available ion currents are calibrated per voltage
        microscope.set_beam_voltage(voltage=settings.milling_voltage, beam_type=channel)
        microscope.set_beam_current(current=settings.milling_current, beam_type=channel)
        logging.debug({"msg": "setup_milling", "mill_settings": settings.to_dict()})

    def _set_application_file(
        self, application_file: str, default: bool = False, strict: bool = True
    ) -> str:
        """Set the application file new patterns are made with: the closest match
        to *application_file* when not *strict*. AutoScript needs a valid one set
        before a pattern is made."""
        application_file = match_application_file(
            application_file,
            self._connection.patterning.list_all_application_files(),
            strict,
        )
        self._connection.patterning.set_default_application_file(application_file)
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
        self._connection.patterning.mode = mode
        logging.debug({"msg": "set_patterning_mode", "mode": mode})
        return mode

    def _draw(self, pattern: FibsemPatternSettings) -> None:
        if isinstance(pattern, FibsemRectangleSettings):
            draw: Callable[[Any], Any] = self._draw_rectangle
        elif isinstance(pattern, FibsemLineSettings):
            draw = self._draw_line
        elif isinstance(pattern, FibsemCircleSettings):
            draw = self._draw_circle
        elif isinstance(pattern, FibsemBitmapSettings):
            draw = self._draw_bitmap_pattern
        elif isinstance(pattern, FibsemPolygonSettings):
            draw = self._draw_polygon
        else:
            return
        # Each pattern starts on the recipe's application file, and one a pattern
        # needs for itself does not outlast it.
        self._set_application_file(self._default_application_file)
        try:
            draw(pattern)
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
        patterning_api = self._connection.patterning
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
        pattern = self._connection.patterning.create_line(
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
            pattern = self._connection.patterning.create_circle(
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
            pattern = self._connection.patterning.create_circle(
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

    def _draw_bitmap_pattern(self, pattern_settings: FibsemBitmapSettings):
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
            pattern = self._connection.patterning.create_bitmap(
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
            pattern = self._connection.patterning.create_bitmap(
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
        rectangle = self._connection.patterning.create_rectangle(
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

        if autoscript.AUTOSCRIPT_VERSION < parse_version("4.12"):
            raise NotImplementedError(
                "Polygon patterning is only supported in Autoscript 4.12 or higher."
            )

        pattern = self._connection.patterning.create_polygon(
            pattern_settings.vertices, depth=pattern_settings.depth
        )
        pattern.is_exclusion_zone = pattern_settings.is_exclusion

        logging.debug(
            {"msg": "draw_polygon", "pattern_settings": pattern_settings.to_dict()}
        )
        self._patterns.append(pattern)
        return pattern

    def read_state(self) -> MillingState:
        microscope = self.parent
        with microscope._threading_lock:
            microscope.set_channel(channel=microscope.milling_channel)
            return MillingState[self._connection.patterning.state.upper()]

    def _start(self) -> None:
        with self.parent._threading_lock:
            if self.read_state() is MillingState.IDLE:
                self._connection.patterning.start()
                logging.info("Starting milling...")

    def _stop(self) -> None:
        with self.parent._threading_lock:
            if self.read_state() in ACTIVE_MILLING_STATES:
                logging.info("Stopping milling...")
                self._connection.patterning.stop()
                logging.info("Milling stopped.")

    def _pause(self) -> None:
        with self.parent._threading_lock:
            if self.read_state() == MillingState.RUNNING:
                logging.info("Pausing milling...")
                self._connection.patterning.pause()
                logging.info("Milling paused.")

    def _resume(self) -> None:
        with self.parent._threading_lock:
            if self.read_state() == MillingState.PAUSED:
                logging.info("Resuming milling...")
                self._connection.patterning.resume()
                logging.info("Milling resumed.")

    def _estimate(self) -> float:
        return sum(pattern.time for pattern in self._patterns)

    def _clear(self) -> None:
        self._connection.patterning.clear_patterns()
        self._patterns = []

    def _restore(self) -> None:
        # The patterning mode persists in xT, so a stage left in Parallel would carry
        # over to the next one unless reset.
        self._set_patterning_mode("Serial")


def bind_autoscript_milling(
    microscope: ThermoMicroscope,
) -> Optional[AutoScriptMilling]:
    """Build ``milling`` for a connected Thermo microscope whose beams are built."""
    return bind_milling(AutoScriptMilling, microscope)

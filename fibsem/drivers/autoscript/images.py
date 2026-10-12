"""AutoScript images: an SDK ``AdornedImage`` as a ``FibsemImage``.

This module imports nothing from the SDK; the Thermo backend's guarded import is the
only one.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

from fibsem.util.timestamps import to_datetime

if TYPE_CHECKING:
    from fibsem.structures import (
        BeamType,
        FibsemImage,
        ImageSettings,
        MicroscopeState,
    )


def image_settings_from_adorned_image(
    image: Any,
    beam_type: Optional["BeamType"] = None,
) -> "ImageSettings":
    """Create ImageSettings from an AutoScript AdornedImage.

    Args:
        image: AutoScript AdornedImage.
        beam_type: Beam type for the image settings.

    Returns:
        ImageSettings: Converted image settings.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    from fibsem.drivers.autoscript import microscope as thermo

    if not thermo.THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AdornedImage."
        )

    from fibsem.structures import BeamType, ImageSettings
    from fibsem.utils import current_timestamp

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    return ImageSettings(
        resolution=(image.width, image.height),
        dwell_time=image.metadata.scan_settings.dwell_time,
        hfw=image.width * image.metadata.binary_result.pixel_size.x,
        autocontrast=True,
        beam_type=beam_type,
        save=False,
        path="path",
        filename=current_timestamp(),
        reduced_area=None,
    )


def fibsem_image_from_adorned_image(
    adorned: Any,
    image_settings: Optional["ImageSettings"] = None,
    state: Optional["MicroscopeState"] = None,
    beam_type: Optional["BeamType"] = None,
) -> "FibsemImage":
    """Create a FibsemImage from an AutoScript AdornedImage.

    Args:
        adorned: AutoScript AdornedImage.
        image_settings: Image settings. Defaults to None (derived from adorned).
        state: Microscope state. Defaults to None (derived from adorned).
        beam_type: Beam type for the image. Defaults to BeamType.ELECTRON.

    Returns:
        FibsemImage: Converted image.

    Raises:
        ImportError: If AutoScript libraries are not available.
    """
    from fibsem.drivers.autoscript import microscope as thermo

    if not thermo.THERMO_API_AVAILABLE:
        raise ImportError(
            "AutoScript libraries not available. Cannot convert from AdornedImage."
        )

    from fibsem.structures import (
        BeamSettings,
        BeamType,
        FibsemImage,
        FibsemImageMetadata,
        FibsemStagePosition,
        MicroscopeState,
        Point,
    )

    if beam_type is None:
        beam_type = BeamType.ELECTRON

    # AutoScript's acquisition_datetime is the instrument PC's clock time with no zone
    # (`07/16/2026 11:07:27` in a saved file). Read as this machine's local time --
    # the instrument PC's, or one in the same building -- and recorded with its offset
    # (FIB-1190). A state passed in keeps its own timestamp, when it was read:
    # overwriting it with this string is what put a string in a float field (FIB-487).
    acquired = to_datetime(adorned.metadata.acquisition.acquisition_datetime)
    if acquired is not None:
        acquired = acquired.astimezone()

    if state is None:
        state = MicroscopeState(
            # Built from the frame's own metadata, so read when the frame was.
            timestamp=acquired.timestamp() if acquired is not None else time.time(),
            stage_position=FibsemStagePosition(
                adorned.metadata.stage_settings.stage_position.x,
                adorned.metadata.stage_settings.stage_position.y,
                adorned.metadata.stage_settings.stage_position.z,
                adorned.metadata.stage_settings.stage_position.r,
                adorned.metadata.stage_settings.stage_position.t,
            ),
            electron_beam=BeamSettings(beam_type=BeamType.ELECTRON),
            ion_beam=BeamSettings(beam_type=BeamType.ION),
        )

    if image_settings is None:
        image_settings = image_settings_from_adorned_image(adorned, beam_type)

    pixel_size = Point(
        adorned.metadata.binary_result.pixel_size.x,
        adorned.metadata.binary_result.pixel_size.y,
    )

    metadata = FibsemImageMetadata(
        image_settings=image_settings,
        pixel_size=pixel_size,
        microscope_state=state,
        acquisition_datetime=acquired,
    )
    return FibsemImage(data=adorned.data, metadata=metadata)

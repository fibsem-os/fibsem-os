"""Auto charge neutralisation for FIB-SEM imaging."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.microscope import FibsemMicroscope
    from fibsem.structures import ImageSettings

logger = logging.getLogger(__name__)


def auto_charge_neutralisation(
    microscope: "FibsemMicroscope",
    image_settings: "ImageSettings",
    discharge_settings: "ImageSettings" = None,
    n_iterations: int = 10,
) -> None:
    """Discharge sample charging by acquiring a rapid sequence of electron images.

    Acquires n_iterations fast probe images to neutralise accumulated charge,
    then acquires the final image with the provided image_settings.

    Args:
        microscope: FibsemMicroscope instance.
        image_settings: Settings for the final image acquisition.
        discharge_settings: Settings for the discharge images. Defaults to a
            fast 768x512 electron image at the same HFW.
        n_iterations: Number of discharge images to acquire.
    """
    from fibsem import acquire
    from fibsem.structures import BeamType, ImageSettings

    if discharge_settings is None:
        discharge_settings = ImageSettings(
            resolution=[768, 512],
            dwell_time=200e-9,
            hfw=image_settings.hfw,
            beam_type=BeamType.ELECTRON,
            save=False,
            autocontrast=False,
            autogamma=False,
            filename=None,
        )

    for i in range(n_iterations):
        acquire.new_image(microscope, discharge_settings)

    acquire.new_image(microscope, image_settings)
    logger.info("Auto charge neutralisation complete (%d iterations).", n_iterations)

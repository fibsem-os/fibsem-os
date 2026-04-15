from __future__ import annotations
import logging
from pathlib import Path

from fibsem import acquire, config as fcfg
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemImage,
    ImageSettings,
)
from fibsem.utils import current_timestamp_v2

########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
):
    """Setup Microscope for FIB Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_stage (FibsemMillingStage): Milling Stage
    """

    # acquire reference image for drift correction
    if milling_stage.alignment.enabled:
        reference_image = get_stage_reference_image(
            microscope=microscope, milling_stage=milling_stage
        )

    # set up milling settings
    microscope.setup_milling(mill_settings=milling_stage.milling)

    # align at the milling current to correct for shift
    if milling_stage.alignment.enabled:
        from fibsem import alignment

        logging.info(
            f"FIB Aligning at Milling Current: {milling_stage.milling.milling_current:.2e}"
        )
        alignment.multi_step_alignment_v2(
            microscope=microscope,
            ref_image=reference_image,
            beam_type=milling_stage.milling.milling_channel,
            steps=milling_stage.alignment.steps,
            use_autocontrast=milling_stage.alignment.use_autocontrast,
            use_autofocus=milling_stage.alignment.use_autofocus,
            plot_title=f"{milling_stage.name} - {milling_stage.milling.milling_current * 1e9:.2e}nA",
        )  # high current -> damaging


def get_stage_reference_image(
    microscope: FibsemMicroscope, milling_stage: FibsemMillingStage
) -> FibsemImage:
    ref_image = milling_stage.reference_image
    if isinstance(ref_image, FibsemImage):
        return ref_image
    elif ref_image is None:
        path = milling_stage.imaging.path
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)
        image_settings = ImageSettings(
            hfw=milling_stage.milling.hfw,
            dwell_time=1e-6,
            resolution=(1536, 1024),
            beam_type=milling_stage.milling.milling_channel,
            reduced_area=milling_stage.alignment.rect,
            save=True,
            path=path,
            filename=f"ref_{milling_stage.name}_initial_alignment_{current_timestamp_v2()}",
        )
        return acquire.acquire_image(microscope, image_settings)
    raise TypeError(f"Invalid ref_image type '{type(ref_image)}'")

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from fibsem import config as fcfg
from fibsem.cancellation import raise_if_cancelled
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemImage,
    ImageSettings,
)
from fibsem.utils import current_timestamp_v2

if TYPE_CHECKING:
    # Type-checking only, so `fibsem.milling` stays microscope-free -- see the note
    # in `base.py`. Deferred annotations are already on at the top of this file.
    from fibsem.microscope import FibsemMicroscope

########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
    stop_event: Optional[threading.Event] = None,
):
    """Setup Microscope for FIB Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_stage (FibsemMillingStage): Milling Stage
        stop_event: threaded into the drift-correction alignment so a Stop during
            "Preparing" aborts it between steps (the strategy then raises before the
            beam starts).
    """

    # acquire reference image for drift correction
    alignment_settings = milling_stage.alignment
    full_frame_reference: Optional[FibsemImage] = None
    if alignment_settings.enabled:
        raise_if_cancelled(stop_event)  # no beam after a stop (FIB-1430)
        reference_image = get_stage_reference_image(
            microscope=microscope, milling_stage=milling_stage
        )
        # Taken now, at the imaging current, as the area's reference is: whether
        # it is used is only known once the milling current is set.
        if alignment_settings.full_frame_below > 0:
            full_frame_reference = get_stage_reference_image(
                microscope=microscope, milling_stage=milling_stage, full_frame=True
            )

    # set up milling settings
    microscope.setup_milling(mill_settings=milling_stage.milling)

    # align at the milling current to correct for shift
    if alignment_settings.enabled:
        from fibsem import alignment

        if full_frame_reference is not None:
            # The current the column reads back, not the configured one: a
            # Tescan sets its ion current by preset and ignores milling_current.
            current = microscope.get_beam_current(milling_stage.milling.milling_channel)
            if current <= alignment_settings.full_frame_below:
                reference_image = full_frame_reference
            used = (
                "full frame"
                if reference_image is full_frame_reference
                else "alignment area"
            )
            logging.info(
                f"{milling_stage.name}: aligning on the {used} at "
                f"{current * 1e12:.0f} pA (full frame at or below "
                f"{alignment_settings.full_frame_below * 1e12:.0f} pA)"
            )

        logging.debug(
            f"FIB Aligning at Milling Current: {milling_stage.milling.milling_current:.2e}"
        )
        # The stage's run folder when the milling task gave it one -- the same
        # <folder>/Alignment the reference image's own path led to before. A
        # stage run on its own keeps finding it from the reference image.
        alignment_path = None
        if milling_stage.output_dir is not None:
            alignment_path = os.path.join(
                milling_stage.output_dir, alignment.ALIGNMENT_SUBDIR
            )
        alignment.multi_step_alignment_v2(
            microscope=microscope,
            ref_image=reference_image,
            steps=milling_stage.alignment.steps,
            use_autocontrast=milling_stage.alignment.use_autocontrast,
            use_autofocus=milling_stage.alignment.use_autofocus,
            run_name=milling_stage.name,
            acquire_final_image=True,
            stop_event=stop_event,  # abort between alignment steps
            path=alignment_path,
        )  # high current -> damaging


def get_stage_reference_image(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
    full_frame: bool = False,
) -> FibsemImage:
    """The stage's alignment reference: the one its task took, else one taken
    now. ``full_frame`` asks for the whole-frame reference instead of the
    alignment area's."""
    ref_image = (
        milling_stage.reference_image_full_frame
        if full_frame
        else milling_stage.reference_image
    )
    if isinstance(ref_image, FibsemImage):
        return ref_image
    elif ref_image is None:
        path = milling_stage.output_dir
        if path is None:
            path = milling_stage.imaging.path
        if path is None:
            path = Path(fcfg.DATA_CC_PATH)
        image_settings = ImageSettings(
            hfw=milling_stage.milling.hfw,
            dwell_time=1e-6,
            resolution=(1536, 1024),
            beam_type=milling_stage.milling.milling_channel,
            reduced_area=None if full_frame else milling_stage.alignment.rect,
            save=True,
            path=path,
            filename=f"ref_{milling_stage.name}_initial_alignment{'_full-frame' if full_frame else ''}_{current_timestamp_v2()}",
        )
        # Imported here rather than at module scope: `fibsem.acquire` imports
        # `fibsem.microscope`, which is the other half of the cycle the
        # TYPE_CHECKING block above avoids. One call site, so one local import.
        from fibsem import acquire

        return acquire.acquire_image(microscope, image_settings)
    raise TypeError(f"Invalid ref_image type '{type(ref_image)}'")

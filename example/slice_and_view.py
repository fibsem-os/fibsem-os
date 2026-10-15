# slice and view

import logging
import os
from pprint import pprint

import numpy as np

import fibsem
from fibsem import acquire, alignment, utils
from fibsem.alignment import AlignmentSubsystem
from fibsem.structures import (
    BeamType,
    CrossSectionPattern,
    FibsemMillingSettings,
    FibsemRectangleSettings,
    ImageSettings,
)


def main():

    PROTOCOL_PATH = os.path.join(
        os.path.dirname(__file__), "protocol_slice_and_view.yaml"
    )
    microscope, settings = utils.setup_session(protocol_path=PROTOCOL_PATH)

    # Milling parameters for this run. These used to come from the configuration's
    # `milling:` block, which no longer exists -- milling belongs to a milling stage,
    # not to the instrument's configuration.
    milling_settings = FibsemMillingSettings(
        milling_current=2.0e-9,
        milling_voltage=30.0e3,
        patterning_mode="Serial",
    )

    pattern_settings = FibsemRectangleSettings(
        width=settings.protocol["milling"]["width"],
        height=settings.protocol["milling"]["height"],
        depth=settings.protocol["milling"]["depth"],
        centre_x=0.0,
        centre_y=0.0,
        scan_direction=settings.protocol["milling"]["scan_direction"],
        cross_section=CrossSectionPattern[
            settings.protocol["milling"]["cross_section"]
        ],
    )

    # angle correction, on the electron beam (where the instrument has it)
    sem = microscope.beams[BeamType.ELECTRON]
    if "tilt_correction" in sem.parameters:
        sem.tilt_correction.set_value(True)
        sem.angular_correction.set_value(np.deg2rad(-38))

    # update image settings
    settings.image.filename = "reference"
    settings.image.save = True
    settings.image.beam_type = BeamType.ELECTRON

    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    ref_eb, ref_ib = None, None

    for slice_idx in range(int(settings.protocol["steps"])):
        # slice
        logging.info("------------------------ SLICE ------------------------")

        microscope.setup_milling(milling_settings)
        microscope.draw_pattern(pattern_settings)
        logging.info(f"Estimated milling time: {microscope.estimate_milling_time()} s")
        microscope.run_milling()
        # clear the pattern and put the ion beam back as setup found it
        microscope.finish_milling()

        # view
        logging.info("------------------------ VIEW ------------------------")
        settings.image.filename = f"slice_{slice_idx:04d}"
        eb_image = acquire.acquire_image(microscope, settings.image)

        # align to the previous slice (stage movement)
        if ref_eb is not None:
            alignment.multi_step_alignment_v2(
                microscope=microscope,
                ref_image=ref_eb,
                steps=3,
                subsystem=AlignmentSubsystem.STAGE,
            )
        ref_eb = eb_image

        # update patterns
        pattern_settings.centre_y += settings.protocol["step_size"]


if __name__ == "__main__":
    main()

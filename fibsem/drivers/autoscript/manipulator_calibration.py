"""Centre the EasyLift needle in both beams, as part of the xT EasyLift alignment.

The xT procedure is run by hand up to its step 5; this then detects the needle tip at
a series of field widths in each beam and moves the needle onto the image centre. Run
it from a terminal on the microscope PC::

    python -m fibsem.drivers.autoscript.manipulator_calibration [debug]

``debug`` first moves the needle near the eucentric position and shows a reference
image pair.
"""

from __future__ import annotations

import sys

import numpy as np

from fibsem import acquire
from fibsem.microscope import FibsemMicroscope
from fibsem.structures.settings import BeamType, FibsemManipulatorPosition
from fibsem.structures.system import MicroscopeSettings

FIELD_WIDTHS = [2000e-6, 900e-6, 400e-6, 150e-6]


def calibrate_manipulator(
    microscope: FibsemMicroscope, settings: MicroscopeSettings
) -> None:
    """Centre the needle in both beams, asking at the terminal before each move."""
    from fibsem.detection import detection
    from fibsem.segmentation.model import load_model

    input(
        "Please complete the EasyLift alignment procedure in the xT UI until Step 5. "
        "Press Enter to proceed."
    )

    model = load_model(settings.protocol["options"]["checkpoint"])
    settings.image.autocontrast = True

    working_distance = microscope.get_working_distance(BeamType.ELECTRON)
    microscope.set_working_distance(
        microscope.system.electron.eucentric_height, BeamType.ELECTRON
    )

    for hfw in FIELD_WIDTHS:
        for beam_type in [BeamType.ELECTRON, BeamType.ION]:
            settings.image.hfw = hfw
            settings.image.beam_type = beam_type

            features = (
                [detection.NeedleTip(), detection.ImageCentre()]
                if np.isclose(microscope.get_scan_rotation(beam_type), 0)
                else [detection.NeedleTipBottom(), detection.ImageCentre()]
            )

            image = acquire.acquire_image(microscope, settings.image)
            det = detection.detect_features(
                image,
                model,
                features=features,
                pixelsize=image.metadata.pixel_size.x,
            )
            detection.plot_detection(det)
            if input("continue? (y/n)") != "y":
                return

            move_x = bool(beam_type == BeamType.ELECTRON)  # ION calibration only in z
            detection.move_based_on_detection(
                microscope,
                settings,
                det,
                beam_type,
                move_x=move_x,
                _move_system="manipulator",
            )

    microscope.set_working_distance(working_distance, BeamType.ELECTRON)
    print("The manipulator should now be centred in both beams.")


def main() -> None:
    import matplotlib.pyplot as plt

    from fibsem import utils

    microscope, settings = utils.setup_session()

    if len(sys.argv) > 1 and sys.argv[1] == "debug":
        microscope.set_scan_rotation(np.deg2rad(0), BeamType.ION)
        microscope.move_manipulator_to_position_offset(
            offset=FibsemManipulatorPosition(), name="EUCENTRIC"
        )
        microscope.move_manipulator_corrected(
            dx=-50e-6, dy=-150e-6, beam_type=BeamType.ELECTRON
        )

        settings.image.autocontrast = True
        settings.image.hfw = 900e-6
        eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(eb_image.data, cmap="gray")
        ax[1].imshow(ib_image.data, cmap="gray")
        plt.show()

    calibrate_manipulator(microscope, settings)


if __name__ == "__main__":
    main()

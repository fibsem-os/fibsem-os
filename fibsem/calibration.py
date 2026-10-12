import numpy as np

from fibsem import acquire
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    MicroscopeSettings,
    MicroscopeState,
)


def auto_home_and_link_v2(
    microscope: FibsemMicroscope, state: MicroscopeState = None
) -> None:

    # home the stage and return the linked state
    if state is None:
        state = microscope.get_microscope_state()

    # home the stage
    microscope.home()

    # move to saved linked state
    microscope.set_microscope_state(state)


def _calibrate_manipulator_thermo(
    microscope: FibsemMicroscope, settings: MicroscopeSettings, parent_ui=None
):
    from fibsem.applications.autolamella.workflows.ui import (
        ask_user,
        update_detection_ui,
    )
    from fibsem.detection import detection
    from fibsem.segmentation.model import load_model

    if parent_ui:
        ret = ask_user(
            parent_ui,
            msg="Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Continue to proceed.",
            pos="Continue",
            neg="Cancel",
        )
        if ret is False:
            return
    else:
        input(
            "Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Enter to proceed."
        )

    def align_manipulator_to_eucentric(
        microsscope: FibsemMicroscope,
        settings: MicroscopeSettings,
        parent_ui,
        validate: bool,
    ) -> None:
        return NotImplemented

    settings.protocol["options"].get("checkpoint", cfg.DEFAULT_CHECKPOINT)
    model = load_model(settings.protocol["options"]["checkpoint"])
    settings.image.autocontrast = True

    hfws = [2000e-6, 900e-6, 400e-6, 150e-6]

    # set working distance
    wd = microscope.get_working_distance(BeamType.ELECTRON)
    microscope.set_working_distance(
        microscope.system.electron.eucentric_height, BeamType.ELECTRON
    )

    for hfw in hfws:
        for beam_type in [BeamType.ELECTRON, BeamType.ION]:
            settings.image.hfw = hfw
            settings.image.beam_type = beam_type

            features = (
                [detection.NeedleTip(), detection.ImageCentre()]
                if np.isclose(microscope.get_scan_rotation(beam_type), 0)
                else [detection.NeedleTipBottom(), detection.ImageCentre()]
            )

            if parent_ui:
                det = update_detection_ui(
                    microscope,
                    settings,
                    features,
                    parent_ui,
                    validate=True,
                    msg="Confirm Feature Detection. Press Continue to proceed.",
                )
            else:
                image = acquire.new_image(microscope, settings.image)
                det = detection.detect_features(
                    image,
                    model,
                    features=features,
                    pixelsize=image.metadata.pixel_size.x,
                )
                detection.plot_detection(det)
                ret = input("continue? (y/n)")

                if ret != "y":
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

    # restore working distance
    microscope.set_working_distance(wd, BeamType.ELECTRON)

    if parent_ui:
        ask_user(
            parent_ui,
            msg="Alignment of EasyLift complete. Please complete the procedure in xT UI. Press Continue to proceed.",
            pos="Continue",
        )
    print("The manipulator should now be centred in both beams.")

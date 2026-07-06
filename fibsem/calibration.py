import logging

import numpy as np

from fibsem import acquire, utils
from fibsem import config as cfg
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    ImageSettings,
    MicroscopeSettings,
    MicroscopeState,
)



# def auto_needle_calibration(
#     microscope: FibsemMicroscope, settings: MicroscopeSettings, validate: bool = True
# ):

#     if TESCAN:
#         raise NotImplementedError

#     # set coordinate system
#     microscope.connection.specimen.manipulator.set_default_coordinate_system(
#         ManipulatorCoordinateSystem.STAGE
#     )

#     # current working distance
#     wd = microscope.connection.beams.electron_beam.working_distance.value
#     needle_wd_eb = 4.0e-3

#     # focus on the needle
#     microscope.connection.beams.electron_beam.working_distance.value = needle_wd_eb
#     microscope.connection.specimen.stage.link()

#     settings.image.hfw = 2700e-6
#     acquire.take_reference_images(microscope, settings.image)

#     # very low res alignment
#     hfws = [2700e-6, 900e-6, 400e-6, 150e-6]
#     for hfw in hfws:
#         settings.image.hfw = hfw
#         align_needle_to_eucentric_position(microscope, settings, validate=validate)

#     # restore working distance
#     microscope.connection.beams.electron_beam.working_distance.value = wd
#     microscope.connection.specimen.stage.link()

#     logging.info(f"Finished automatic needle calibration.")


# def align_needle_to_eucentric_position(
#     microscope: FibsemMicroscope,
#     settings: MicroscopeSettings,
#     validate: bool = False,
# ) -> None:
#     """Move the needle to the eucentric position, and save the updated position to disk

#     Args:
#         microscope (FibsemMicroscope): OpenFIBSEM microscope instance
#         settings (MicroscopeSettings): microscope settings
#         validate (bool, optional): validate the alignment. Defaults to False.
#     """

#     from fibsem.ui import windows as fibsem_ui_windows
#     from fibsem.detection import detection

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ELECTRON

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope,
#         settings=settings,
#         features=[
#             NeedleTip(),
#             ImageCentre(),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope, settings, det, beam_type=settings.image.beam_type
#     )

#     # take reference images
#     settings.image.save = False
#     settings.image.beam_type = BeamType.ION

#     image = acquire.new_image(microscope, settings.image)

#     det = fibsem_ui_windows.detect_features_v2(
#         microscope=microscope,
#         settings=settings,
#         features=[
#             NeedleTip(),
#             ImageCentre(),
#         ],
#         validate=validate,
#     )
#     detection.move_based_on_detection(
#         microscope, settings, det, beam_type=settings.image.beam_type, move_x=False
#     )

#     # take image
#     acquire.take_reference_images(microscope, settings.image)


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


def _calibrate_manipulator_thermo(microscope:FibsemMicroscope, settings:MicroscopeSettings, parent_ui = None):
    from fibsem.applications.autolamella.workflows.ui import ask_user, update_detection_ui

    from fibsem.detection import detection
    from fibsem.segmentation.model import load_model

    if parent_ui:
        ret = ask_user(parent_ui, 
            msg="Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Continue to proceed.",
            pos="Continue", neg="Cancel")
        if ret is False:
            return
    else:
        input("Please complete the EasyLift alignment procedure in the xT UI until Step 5. Press Enter to proceed.")


    def align_manipulator_to_eucentric(microsscope: FibsemMicroscope, settings:MicroscopeSettings, parent_ui, validate: bool) -> None:
        return NotImplemented

    settings.protocol["options"].get("checkpoint", cfg.DEFAULT_CHECKPOINT)
    model = load_model(settings.protocol["options"]["checkpoint"])
    settings.image.autocontrast = True

    hfws = [2000e-6, 900e-6, 400e-6, 150e-6]

    # set working distance
    wd = microscope.get("working_distance", BeamType.ELECTRON)
    microscope.set("working_distance", microscope.system.electron.eucentric_height, BeamType.ELECTRON)

    for hfw in hfws:
        for beam_type in [BeamType.ELECTRON, BeamType.ION]:
            settings.image.hfw = hfw
            settings.image.beam_type = beam_type

            features = [detection.NeedleTip(), detection.ImageCentre()] if np.isclose(microscope.get("scan_rotation", beam_type), 0) else [detection.NeedleTipBottom(), detection.ImageCentre()]
            
            if parent_ui:
                det = update_detection_ui(microscope, settings, features, parent_ui, validate = True, msg = "Confirm Feature Detection. Press Continue to proceed.")
            else:
                image = acquire.new_image(microscope, settings.image)
                det = detection.detect_features(image, model, features=features, pixelsize=image.metadata.pixel_size.x)
                detection.plot_detection(det)
                ret  = input("continue? (y/n)")
                
                if ret != "y":
                    return

            move_x = bool(beam_type == BeamType.ELECTRON) # ION calibration only in z
            detection.move_based_on_detection(microscope, settings, det, beam_type, move_x=move_x, _move_system="manipulator")

    # restore working distance
    microscope.set("working_distance", wd, BeamType.ELECTRON)

    if parent_ui:
        ask_user(parent_ui, 
            msg="Alignment of EasyLift complete. Please complete the procedure in xT UI. Press Continue to proceed.",
            pos="Continue")
    print("The manipulator should now be centred in both beams.")
import logging
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import os
import numpy as np
from fibsem import acquire, alignment, calibration
from fibsem import utils as fibsem_utils
from fibsem import validation
from fibsem.detection import detection
from fibsem.detection.detection import (
    ImageCentre,
    LamellaCentre,
    LamellaLeftEdge,
    LamellaRightEdge,
    LandingPost,
    NeedleTip,
    LandingPost,
    LamellaTopEdge,
    LamellaBottomEdge,
    detect_features,
    DetectedFeatures,
)
from fibsem.imaging import utils as image_utils
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage, get_milling_stages, get_protocol_from_stages
from fibsem.structures import (
    BeamType,
    FibsemRectangle,
    FibsemStagePosition,
    FibsemImage,
    MicroscopeSettings,
    MicroscopeState,
    Point,
    ImageSettings
)

from autolamella.workflows import actions
from autolamella.structures import AutoLamellaStage, Experiment, Lamella
from autolamella.ui import AutoLamellaUI
from fibsem import config as fcfg


from autolamella.workflows.core import (log_status_message, log_status_message_raw,
                                        start_of_stage_update, end_of_stage_update, 
                                        mill_trench, mill_undercut, mill_lamella, 
                                        setup_lamella)
from autolamella.workflows.ui import (update_milling_ui, update_status_ui, 
                                      set_images_ui, ask_user, update_detection_ui, 
                                      update_experiment_ui)


# autoliftout workflow functions

def liftout_lamella(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:
    # bookkeeping
    validate = bool(settings.protocol["options"]["supervise"]["liftout"])
    settings.image.path = lamella.path

    # get ready to do liftout by moving to liftout angle (flat to eb)
    # actions.move_to_liftout_angle(microscope, settings)

    log_status_message(lamella, "ALIGN_REF_UNDERCUT")

    # detect
    log_status_message(lamella, "ALIGN_TRENCH")
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_trench_align_ml"
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    features = [LamellaCentre()] 
    # TODO: FAIL HERE
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

    microscope.stable_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
        beam_type=settings.image.beam_type
    )

    # Align ion so it is coincident with the electron beam
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH

    features = [LamellaCentre()] 
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)
    
    # align vertical
    microscope.vertical_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
    )

    # lamella should now be centred in ion beam


    # reference images for needle location
    settings.image.save = True
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_needle_inserted"
    acquire.take_reference_images(microscope, settings.image)

    # land needle on lamella
    lamella = land_needle_on_milled_lamella(
        microscope, settings, lamella, validate=validate, parent_ui=parent_ui
    )

    log_status_message(lamella, "NEEDLE_JOIN_LAMELLA")
    update_status_ui(
        parent_ui, f"{lamella.info} Joining Needle to Lamella..."
    )


    _JOINING_METHOD = settings.protocol["options"]["liftout_joining_method"].upper()
    logging.info(f"Using {_JOINING_METHOD} joining method")

    # joining options
    if  _JOINING_METHOD == "WELD":
        settings.image.beam_type = BeamType.ION

        features = [LamellaLeftEdge()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)


        # mill weld
        stages = get_milling_stages(
            "weld", settings.protocol, det.features[0].feature_m
        )
        stages = update_milling_ui(microscope, stages, parent_ui, 
            msg=f"Press Run Milling to mill the weld for {lamella.petname}. Press Continue when done.", 
            validate=validate)
        
        lamella.protocol["join"] = deepcopy(get_protocol_from_stages(stages))
        lamella.protocol["join"]["point"] = stages[0].pattern.point.to_dict()

    logging.info(
        f"{lamella.state.stage.name}: lamella to needle joining complete."
    )

    settings.image.save = True
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_manipulator_contact"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    log_status_message(lamella, "NEEDLE_SEVER_LAMELLA")
    update_status_ui(parent_ui, f"{lamella.info} Severing Lamella...")

    # sever lamella
    lamella = mill_lamella_edge(
        microscope=microscope,
        settings=settings,
        parent_ui=parent_ui,
        lamella=lamella,
        validate=validate,
        x_shift=1e-6,
    )

    # take reference images
    settings.image.save = True
    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.filename = f"ref_{lamella.state.stage.name}_sever"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    log_status_message(lamella, "NEEDLE_TRENCH_REMOVAL")
    update_status_ui(
        parent_ui, f"{lamella.info} Removing Needle from trench..."
    )

    # Raise needle 30um from trench
    # move needle back from trench x
    dx = -1.5e-6
    microscope.move_manipulator_corrected(dx=dx, dy=0, beam_type=BeamType.ION)

    for i in range(3):
        microscope.move_manipulator_corrected(dx=0, dy=10e-6, beam_type=BeamType.ION)
        settings.image.filename = f"liftout_trench_{i:02d}"
        acquire.take_reference_images(microscope, settings.image)
        time.sleep(1)

    # take reference images
    log_status_message(lamella, "REFERENCE_IMAGES")
    reference_images = acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_{lamella.state.stage.name}_final",
    )
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    # move needle to park position
    microscope.retract_manipulator()  # retracted needle not supported on tescan

    return lamella


def land_needle_on_milled_lamella(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    lamella: Lamella,
    validate: bool,
    parent_ui: AutoLamellaStage,
) -> Lamella:
    # bookkeeping
    settings.image.path = lamella.path

    # validate needle insertion conditions
    # validate_needle_insertion(
    #     microscope, settings.system.stage.needle_stage_height_limit
    # )

    log_status_message(lamella, "INSERT_NEEDLE")
    update_status_ui(parent_ui, f"{lamella.info} Inserting Needle...")

    # insert the needle for liftout
    actions.move_needle_to_liftout_position(microscope)

    # align needle to side of lamella
    log_status_message(lamella, "NEEDLE_EB_DETECTION")

    settings.image.beam_type = BeamType.ELECTRON

    features = [NeedleTip(), LamellaLeftEdge()]
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

    # offset from lamella
    logging.info(f"DISTANCE: {det.distance}")
    H_OFFSET = 2.5e-6
    det._offset = Point(-H_OFFSET, 0) 
    logging.info(f"OFFSET: {det._offset}")
    logging.info(f"DISTANCE: {det.distance}")
    
    detection.move_based_on_detection(
        microscope, settings, det, beam_type=settings.image.beam_type
    )

    log_status_message(lamella, "NEEDLE_IB_DETECTION")

    # need to do vertical movement twice, because needle isnt moving correctly in z
    # TODO: remove when needle movement is fixed.
    for i in range(2):
        settings.image.beam_type = BeamType.ION

        features = [NeedleTip(), LamellaLeftEdge()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

        detection.move_based_on_detection(
            microscope, settings, det, beam_type=settings.image.beam_type, move_x=False
        )

    # reference images
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.save = True
    settings.image.filename = f"ref_{lamella.state.stage.name}_manipulator_start_position"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    if validate:
        ask_user(
            parent_ui,
            msg="Confirm Needle is positioned to left of the Lamella Left Edge.",
            pos="Continue",
        )

    # charge neutralisation  to charge lamella
    settings.image.beam_type = BeamType.ION
    n_iter = int(
        settings.protocol["options"].get("liftout_charge_neutralisation_iterations", 35)
    )

    calibration.auto_charge_neutralisation(
        microscope=microscope, 
        image_settings=settings.image, 
        n_iterations=n_iter, 
        discharge_settings = ImageSettings(
            resolution=[768, 512],
            dwell_time=200e-9,
            hfw=settings.image.hfw,
            beam_type=BeamType.ION,
            save=False,
            autocontrast=False,
            autogamma=False,
            filename=None,
        )
    )

    X_LIFTOUT_CONTACT_OFFSET = settings.protocol["options"].get("liftout_contact_offset", 0.25e-6)
    features = [NeedleTip(), LamellaLeftEdge()]
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)
    det._offset = Point(x=X_LIFTOUT_CONTACT_OFFSET, y=0)
    detection.move_based_on_detection(microscope, settings, det, beam_type=settings.image.beam_type, move_y=False)

    _USE_CONTACT_DETECTION = settings.protocol["options"].get("liftout_contact_detection", False)
    if _USE_CONTACT_DETECTION:
        lamella = _liftout_contact_detection(microscope, settings, lamella, parent_ui, validate=validate)
                
    # move needle up in z to prevent bottoming out
    dz = 0.5e-6  # positive is away from sample (up)
    microscope.move_manipulator_corrected(dx=0, dy=dz, beam_type=BeamType.ION)

    # restore imaging settings
    settings.image.autogamma = True
    settings.image.reduced_area = None

    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER],
        filename=f"ref_{lamella.state.stage.name}_manipulator_landed",
    )

    return lamella

def _liftout_contact_detection(microscope: FibsemMicroscope, settings: MicroscopeSettings, 
                                lamella: Lamella, parent_ui: AutoLamellaUI, validate: bool = True) -> Lamella:
    

    # measure brightness
    BRIGHTNESS_FACTOR = 1.2
    settings.image.beam_type = BeamType.ION
    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.filename = f"ref_{lamella.state.stage.name}_manipulator_land_initial"
    settings.image.save = True
    settings.image.autogamma = False
    reduced_area = FibsemRectangle(
        0.2, 0.2, 0.70, 0.70
    )  # TODO: improve contact detection
    settings.image.reduced_area = reduced_area
    ib_image = acquire.new_image(microscope, settings.image)
    previous_brightness = image_utils.measure_brightness(ib_image)

    brightness_history = [previous_brightness]
    mean_brightness = np.mean(brightness_history)

    iteration_count = 0
    MAX_ITERATIONS = 15

    log_status_message(lamella, "NEEDLE_CONTACT_DETECTION")

    while True:
        # move needle down
        dx = 0.5e-6
        dy = 0.0e-6

        microscope.move_manipulator_corrected(dx=dx, dy=dy, beam_type=BeamType.ION)

        # calculate brightness
        settings.image.filename = f"ref_{lamella.state.stage.name}_contact_brightness_{iteration_count}"
        ib_image = acquire.new_image(microscope, settings.image)
        brightness = image_utils.measure_brightness(ib_image)
        set_images_ui(parent_ui, None, ib_image)

        logging.info(
            f"iter: {iteration_count}: brightness: {brightness}, prevs: {previous_brightness}, MEAN BRIGHTNESS: {mean_brightness}"
        )

        above_brightness_threshold = brightness > mean_brightness * BRIGHTNESS_FACTOR

        if above_brightness_threshold and validate is False:
            break  # exit loop

        if above_brightness_threshold or validate:
            # needle has landed...
            if above_brightness_threshold:
                logging.info("BRIGHTNESS THRESHOLD REACHED STOPPPING")

            msg = f"Has the needle landed on the lamella? Above Threshold: {above_brightness_threshold} ({iteration_count+1}/{MAX_ITERATIONS})"
            response = ask_user(parent_ui, msg=msg, pos="Yes", neg="No")
            if response is True:
                break

        previous_brightness = brightness
        brightness_history.append(brightness)
        mean_brightness = np.mean(brightness_history)

        iteration_count += 1
        if iteration_count >= MAX_ITERATIONS:
            break
    return lamella


def land_lamella(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:
    # bookkeeping
    validate = bool(settings.protocol["options"]["supervise"]["landing"])
    settings.image.path = lamella.path
    settings.image.save = False

    # move to landing coordinate
    microscope.set_microscope_state(lamella.landing_state)

    # align to ref
    log_status_message(lamella, "ALIGN_REF_LANDING")
    reference_images = lamella.get_reference_images("ref_landing")
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    # align to ref
    alignment.correct_stage_drift(
        microscope,
        settings,
        reference_images=reference_images,
        alignment=(BeamType.ION, BeamType.ION),
        rotate=False,
        xcorr_limit=(512, 100),
    )

    # confirm eucentricity
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    if validate:
        ask_user(
            parent_ui,
            msg=f"Confirm {lamella.info} is centred in both Beams. Press Continue to proceed.",
            pos="Continue",
        )

    ############################## LAND_LAMELLA ##############################
    # validate_needle_insertion(
    #     microscope, settings.system.stage.needle_stage_height_limit
    # )

    # landing entry
    log_status_message(lamella, "LAND_LAMELLA_ENTRY")
    update_status_ui(
        parent_ui, f"{lamella.info} Landing Entry Procedure..."
    )
    # TODO tescan warning to insert needle
    landing_entry_procedure(microscope, settings, lamella=lamella, 
    validate=validate,parent_ui=parent_ui)
    # land lamella on post
    log_status_message(lamella, "LAND_LAMELLA_ON_POST")
    response = False
    _repeated_landing_attempt = False
    while response is False:
        if _repeated_landing_attempt:
            mill_lamella_edge(
                microscope,
                settings,
                parent_ui=parent_ui,
                lamella=lamella,
                validate=validate,
            )
        # land the lamella on the post
        land_lamella_on_post(
            microscope=microscope,
            settings= settings,
            parent_ui=parent_ui,
            lamella=lamella,
            validate=validate)

        # confirm with user
        eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
        set_images_ui(parent_ui, eb_image, ib_image)

        # TODO: reenable this
        # features = [LamellaRightEdge(), LandingPost()]
        # det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)
        # successful_landing = (det.distance.x >= 2.5e-6)

        if validate:
            response = ask_user(
                parent_ui,
                msg=f"Confirm {lamella.info} has landed on post. \nPress Continue to proceed, or Repeat to attempt landing again.",
                pos="Continue",
                neg="Repeat",
            )
        else:
            response = True

        _repeated_landing_attempt = True

    log_status_message(lamella, "LAND_LAMELLA_REMOVE_NEEDLE")
    # move needle out of trench slowly at first (QUERY: is this required?)
    for i in range(3):
        microscope.move_manipulator_corrected(dx=0, dy=10e-6, beam_type=BeamType.ION)
        time.sleep(1)

    # move needle to park position
    microscope.retract_manipulator()

    # take reference images
    log_status_message(lamella, "REFERENCE_IMAGES")
    reference_images = acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_{lamella.state.stage.name}_final",
    )
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    # reset manipulator
    RESET_REQUIRED = bool(settings.protocol["options"]["liftout_joining_method"].capitalize() != "None")

    response = True
    if validate:
        response = ask_user(parent_ui, msg="Do you want to Reset the manipulator?", pos="Reset", neg="Skip")

    if response and RESET_REQUIRED:
        lamella = reset_needle(microscope, settings, lamella,  parent_ui,)


    return lamella


def mill_lamella_edge(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    parent_ui: AutoLamellaUI,
    lamella: Lamella = None,
    validate: bool = True,
    x_shift: float = 0,
):
    settings.image.save = True
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_mill_lamella_edge"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    log_status_message(
        lamella, f"{lamella.state.stage.name.upper()}_MILL_LAMELLA_EDGE"
    )
    update_status_ui(parent_ui, f"{lamella.info} Mill Lamella Edge...")

    settings.image.beam_type = BeamType.ION

    features = [LamellaRightEdge()]
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

    # adjustment
    point = det.features[0].feature_m
    point.x += x_shift

    stages = get_milling_stages("sever", settings.protocol["milling"], point=point)
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the sever for {lamella.petname}. Press Continue when done.", 
        validate=validate)

    lamella.protocol[f"{lamella.state.stage.name}_sever"] = deepcopy(get_protocol_from_stages(stages[0]))
    lamella.protocol[f"{lamella.state.stage.name}_sever"]["point"] = stages[0].pattern.point.to_dict()

    # take reference images
    settings.image.save = True
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_lamella_edge_milled"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    return lamella


def land_lamella_on_post(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    parent_ui: AutoLamellaUI,
    lamella: Lamella,
    validate: bool = True,
):
    # done

    # repeat final movement until user confirms landing
    response = False
    i = 0
    while response is False:
        #### X-MOVE
        settings.image.hfw = (
            fcfg.REFERENCE_HFW_HIGH if i == 0 else fcfg.REFERENCE_HFW_SUPER
        )
        settings.image.beam_type = BeamType.ION
        settings.image.save = True
        settings.image.filename = f"ref_{lamella.state.stage.name}_needle_pre_move_{i}"

        log_status_message(lamella, "LAND_LAMELLA_IB_DETECTION")

        features = [LamellaRightEdge(), LandingPost()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

        logging.info(f"OFFSET: {det._offset}")
        logging.info(f"DISTANCE: {det.distance}")
        det._offset = Point(x=settings.protocol["options"].get("landing_post_x_offset", 0.75e-6), y=0)
        logging.info(f"OFFSET: {det._offset}")
        logging.info(f"DISTANCE: {det.distance}")
        
        detection.move_based_on_detection(
            microscope, settings, det, beam_type=settings.image.beam_type
        )

        # final reference images
        settings.image.save = True
        settings.image.filename = f"ref_{lamella.state.stage.name}_lamella_contact_{i}"
        eb_image, ib_image = acquire.take_reference_images(
            microscope=microscope, image_settings=settings.image
        )
        set_images_ui(parent_ui, eb_image, ib_image)

        if validate:
            response =ask_user(
                parent_ui,
                msg=f"Confirm {lamella.info} has made contact with post. \nPress Continue to proceed or Repeat to repeat the landing attempt.",
                pos="Continue",
                neg="Repeat",
            )
        else:
            response = True

            # TODO: add a check where if the lamella edge and post are close enough, exit the loop

        # increment count
        i += 1

    #################################################################################################

    ############################## WELD TO LANDING POST #############################################

    log_status_message(lamella, "LAND_LAMELLA_WELD_TO_POST")

    settings.image.beam_type = BeamType.ION

    features = [LamellaRightEdge()]
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)

    point = det.features[0].feature_m 
    point.x += settings.protocol["milling"]["weld"].get("width", 5e-6) / 2

    stages = get_milling_stages("weld", settings.protocol["milling"], point)
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the weld for {lamella.petname}. Press Continue when done.", 
        validate=validate)
    
    lamella.protocol["weld"] = deepcopy(get_protocol_from_stages(stages[0]))
    lamella.protocol["weld"]["point"] = stages[0].pattern.point.to_dict()

    # final reference images
    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.save = True
    settings.image.filename = f"ref_{lamella.state.stage.name}_lamella_final_weld_high_res"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    #################################################################################################

    ###################################### REMOVE NEEDLE ######################################

    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.beam_type = BeamType.ION
    settings.image.save = True
    settings.image.filename = f"ref_{lamella.state.stage.name}_lamella_needle_removal"

    # charge neutralisation # discharge to unlock lamella
    response = False
    n_iter = int(
        settings.protocol["options"].get(
            "landing_charge_neutralisation_iterations", 100
        )
    )
    while response is False:
        settings.image.beam_type = BeamType.ELECTRON
        calibration.auto_charge_neutralisation(
            microscope, settings.image, n_iterations=n_iter
        )

        if validate:
            response = ask_user(
                parent_ui,
                msg=f"Repeat charge procedure? iter: {n_iter}. \nPress Continue to proceed or Repeat to repeat the dis-charge attempt.",
                pos="Continue",
                neg="Repeat",
            )
        else:
            response = True

    # optional? makes it not repeatable
    log_status_message(lamella, "LAND_LAMELLA_REMOVE_NEEDLE")

    logging.info(f"{lamella.state.stage.name}: removing needle from lamella")

    # back out needle from lamella , no cut required?
    for i in range(10):
        # move needle back
        microscope.move_manipulator_corrected(dx=-0.5e-6, dy=0, beam_type=BeamType.ION)

    # reference images
    acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER],
        filename=f"ref_{lamella.state.stage.name}_lamella_needle_removal",
    )

    return


def landing_entry_procedure(
    microscope: FibsemMicroscope, settings: MicroscopeSettings, lamella: Lamella, validate: bool = True,parent_ui=AutoLamellaUI
):
    # entry procedure, align vertically to post
    actions.move_needle_to_landing_position(microscope)

    hfws = [
        fcfg.REFERENCE_HFW_LOW,
        fcfg.REFERENCE_HFW_MEDIUM,
        fcfg.REFERENCE_HFW_HIGH,
        fcfg.REFERENCE_HFW_HIGH,
    ]
    beam_types = [BeamType.ION, BeamType.ELECTRON, BeamType.ION, BeamType.ION]

    for i, (hfw, beam_type) in enumerate(zip(hfws, beam_types)):
        # needle starting position
        settings.image.hfw = hfw
        settings.image.beam_type = beam_type
        settings.image.save = True
        settings.image.filename = f"ref_{lamella.state.stage.name}_needle_start_position_{i}"
        eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
        set_images_ui(parent_ui, eb_image, ib_image)
        
        features = [LamellaRightEdge(), LandingPost()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg=lamella.info)
       
        det._offset = Point(-30e-6, 0)
        logging.info(f"DISTANCE: {det.distance}, OFFSET: {det._offset}")
        detection.move_based_on_detection(
            microscope, settings, det, beam_type=settings.image.beam_type
        )

    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_{lamella.state.stage.name}_needle_ready_position"
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

def reset_needle(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:
    # bookkeeping
    validate = bool(settings.protocol["options"]["supervise"]["reset"])

    settings.image.path = lamella.path

    ask_user(msg="Reset manipulator is currently unavailble, please use the manual controls to reset the manipulator.", pos="OK")

    return lamella

    # TODO: re-implement

    # move sample stage out
    actions.move_sample_stage_out(microscope, settings)

    ###################################### SHARPEN_NEEDLE ######################################

    validation.validate_stage_height_for_needle_insertion(
        microscope, settings.system.stage.needle_stage_height_limit
    )

    # move needle in
    actions.move_needle_to_reset_position(microscope)

    # needle imagesV
    settings.image.save = True
    settings.image.filename = f"ref_{lamella.state.stage.name}_needle_start_position"
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)
    settings.image.beam_type = BeamType.ION

    # TODO: move needle to the centre, because it has been cut off...
    calibration.align_needle_to_eucentric_position(microscope, settings, validate=False)

    log_status_message_raw(workflow_stage, "MOVE_TO_EUCENTRIC")
    for beam_type in [BeamType.ELECTRON, BeamType.ION]:
        
        settings.image.hfw = fcfg.REFERENCE_HFW_HIGH 
        settings.image.beam_type = beam_type

        # detect manipulator and user defined feature
        features = [detection.NeedleTip(), detection.CoreFeature()] if np.isclose(scan_rotation, 0) else [detection.NeedleTipBottom(), detection.CoreFeature()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg="Prepare Manipulator")

        # move manipulator to target position
        detection.move_based_on_detection(microscope, settings, det, beam_type, _move_system="manipulator")

    
    # TODO: validate this movement

    # # create sharpening patterns
    ask_user(
        parent_ui,
        msg="Please move the needle to the centre of the view and press OK",
        pos="OK",
    )

    stages = get_milling_stages("sharpen", settings.protocol)
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the sharpen for {lamella.petname}. Press Continue when done.", 
        validate=validate)

    lamella.protocol["reset"] = deepcopy(get_protocol_from_stages(stages[0]))
    lamella.protocol["reset"]["point"] = stages[0].pattern.point.to_dict()

    #################################################################################################

    # reset the "eucentric position" for the needle, centre needle in both views
    calibration.align_needle_to_eucentric_position(microscope, settings, validate=True)

    # take reference images
    settings.image.filename = f"ref_{lamella.state.stage.name}_final"
    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.save = True
    acquire.take_reference_images(microscope=microscope, image_settings=settings.image)

    # retract needle
    microscope.retract_manipulator()

    # # reset stage position
    # move_settings = MoveSettings(rotate_compucentric=True)
    # stage.absolute_move(StagePosition(t=np.deg2rad(0)), move_settings)
    # stage.absolute_move(StagePosition(x=0.0, y=0.0), move_settings)
    microscope.move_stage_absolute(x=0.0, y=0.0, t=0.0)

    # TODO: test this
    if lamella.landing_selected:
        calibration.set_microscope_state(microscope, lamella.landing_state)

    return lamella


def run_setup_autoliftout(
    microscope: FibsemMicroscope,	
    settings: MicroscopeSettings,	
    experiment: Experiment,	
    parent_ui: AutoLamellaUI,	
) -> Experiment:
    
    log_status_message_raw(f"{AutoLamellaStage.SetupTrench.name}", "STARTED")

    # select the lamella and landing positions
    experiment = select_lamella_positions(microscope, settings, experiment, parent_ui)

    return experiment

# autoliftout_workflow
WORKFLOW_STAGES = {
    # AutoLamellaStage.SetupTrench: run_setup_autoliftout, # TODO: split this further
    AutoLamellaStage.MillTrench: mill_trench,
    AutoLamellaStage.MillUndercut: mill_undercut,
    AutoLamellaStage.LiftoutLamella: liftout_lamella,
    AutoLamellaStage.LandLamella: land_lamella,
    AutoLamellaStage.SetupLamella: setup_lamella,
    AutoLamellaStage.MillRough: mill_lamella,
    AutoLamellaStage.MillPolishing: mill_lamella,
}

def run_autoliftout_workflow(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
) -> Experiment:


    update_status_ui(parent_ui, "Starting AutoLiftout Workflow...")
    logging.info(
        f"AutoLiftout Workflow started for {len(experiment.positions)} lamellae."
    )
    settings.image.save = False
    settings.image.path = experiment.path
    settings.image.filename = f"{fibsem_utils.current_timestamp()}"

    
    # batch mode workflow
    if True:
        for terminal_stage in [
            AutoLamellaStage.MillTrench,
            AutoLamellaStage.MillUndercut, # TODO: maybe add this to config?
        ]:
            lamella: Lamella
            for lamella in experiment.positions:
                if lamella.is_failure:
                    continue  # skip failures

                while lamella.state.stage.value < terminal_stage.value:
                    next_stage = AutoLamellaStage(lamella.state.stage.value + 1)

                    # update image settings (save in correct directory)
                    settings.image.path = lamella.path

                    # reset to the previous state
                    lamella = start_of_stage_update(
                        microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                    )

                    # run the next workflow stage
                    lamella = WORKFLOW_STAGES[next_stage](
                        microscope=microscope,
                        settings=settings,
                        lamella=lamella,
                        parent_ui=parent_ui,
                    )
                    # advance workflow
                    experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui)

                    # update ui
                    update_experiment_ui(parent_ui, experiment)

    # standard workflow
    lamella: Lamella
    for lamella in experiment.positions:
        if lamella.is_failure:
            continue  # skip failures

        while lamella.state.stage.value < AutoLamellaStage.LandLamella.value:

            next_stage = AutoLamellaStage(lamella.state.stage.value + 1)
            if True:
                msg = (
                    f"""Continue Lamella {(lamella.petname)} from {next_stage.name}?"""
                )
                response = ask_user(parent_ui, msg=msg, pos="Continue", neg="Skip")

            else:
                response = True

            # update image settings (save in correct directory)
            settings.image.path = lamella.path

            if response:
                # reset to the previous state
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                )

                # run the next workflow stage
                lamella = WORKFLOW_STAGES[next_stage](
                    microscope=microscope,
                    settings=settings,
                    lamella=lamella,
                    parent_ui=parent_ui,
                )

                # advance workflow
                experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui)
                
                # update ui
                update_experiment_ui(parent_ui, experiment)
            else:
                break  # go to the next lamella

    return experiment

# TODO: separate the _run_setup_lamella_workflow so we dont have to do sily passthrough step
def run_thinning_workflow(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
) -> Experiment:

    update_status_ui(parent_ui, "Starting MillRough Workflow...")
    lamella: Lamella
    for next_stage in [
        AutoLamellaStage.SetupLamella,
        # AutoLamellaStage.ReadyLamella,
        AutoLamellaStage.MillRough,
        AutoLamellaStage.MillPolishing,
    ]:
        for lamella in experiment.positions:
            if lamella.is_failure:
                continue

            if lamella.state.stage.value == next_stage.value - 1:

                restore_state = True # next_stage != AutoLamellaStage.ReadyLamella
                save_state = True # next_stage != AutoLamellaStage.ReadyLamella

                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage, parent_ui=parent_ui, 
                    restore_state=restore_state
                )
                WORKFLOW_STAGES[next_stage](microscope, settings, lamella,parent_ui)
                experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui, 
                                                 save_state=save_state)
                update_experiment_ui(parent_ui, experiment)

    # finish the experiment
    for lamella in experiment.positions:
        if lamella.state.stage == AutoLamellaStage.MillPolishing:
            lamella = start_of_stage_update(microscope, lamella, next_stage=AutoLamellaStage.Finished, parent_ui=parent_ui, restore_state=False)
            experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui, save_state=False)
            update_experiment_ui(parent_ui, experiment)


    return experiment

def get_current_lamella(
    experiment: Experiment, parent_ui: AutoLamellaUI
) -> bool:
    select_another_lamella = (
        ask_user(
            parent_ui,
            msg=f"Do you want to select another trench? {len(experiment.positions)} currentlly selected.",
            pos="Yes",
            neg="No",
        )
        if experiment.positions
        else True
    )
    return select_another_lamella


def select_initial_lamella_positions(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI = None,
) -> Lamella:
    """Select the initial experiment positions for liftout"""

    # create lamella
    lamella_no = max(len(experiment.positions) + 1, 1)
    lamella = Lamella(path=experiment.path, number=lamella_no)
    log_status_message(lamella, "STARTED")

    # reference images
    settings.image.hfw = fcfg.REFERENCE_HFW_MEDIUM
    settings.image.save = False
    eb_image,ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    log_status_message(lamella, "SELECT_LAMELLA_POSITION")
    stages = get_milling_stages("trench", settings.protocol["milling"])
    stages = update_milling_ui(microscope, stages, parent_ui,
        msg=f"Select a position and milling pattern for {lamella.petname}. Press Continue when done.",
        validate=True,
        milling_enabled=False
    )
    
    # log the protocol
    lamella.protocol["trench"] = deepcopy(get_protocol_from_stages(stages))
    lamella.protocol["trench"]["point"] = stages[0].pattern.point.to_dict()
    
    # need to set the imaging settings too?
    
    # save microscope state   

    lamella.state.start_timestamp = datetime.timestamp(datetime.now())
    lamella.state.microscope_state = microscope.get_microscope_state()

    log_status_message(lamella, "LAMELLA_REFERENCE_IMAGES")

    settings.image.hfw = fcfg.REFERENCE_HFW_LOW
    settings.image.save = True
    settings.image.path = lamella.path

    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_{lamella.state.stage.name}_start",
    )

    return lamella


def select_landing_positions(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
):
    """Select landing positions for autoliftout"""

    ####################################
    # # move to landing grid
    landing_start_position = fibsem_utils._get_position(settings.protocol["options"]["landing_start_position"])
    microscope.safe_absolute_stage_movement(landing_start_position)

    settings.image.save = False
    ####################################

    # select corresponding experiment landing positions
    lamella: Lamella
    for lamella in experiment.positions:
        # check if landing position already selected? so it doesnt overwrite
        if lamella.landing_selected is False:
            settings.image.hfw = fcfg.REFERENCE_HFW_LOW
            lamella = select_landing_sample_positions(
                microscope, settings, lamella, parent_ui
            )

            experiment.save()

    return experiment


def select_landing_sample_positions(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:
    """Select the landing coordinates for a lamella."""
    logging.info(f"Selecting Landing Position: {lamella.petname}")

    # update image path
    settings.image.path = lamella.path
    settings.image.hfw = fcfg.REFERENCE_HFW_LOW

    # eb_image, ib_image = acquire.take_reference_images(microscope=microscope, image_settings=settings.image)
    # set_images_ui(parent_ui, eb_image, ib_image)

    # # select landing coordinates
    # ask_user(
    #     parent_ui,
    #     msg=f"Select the landing coordinate for {lamella.petname}.",
    #     pos="Continue",
    # )  # enable movement, imaging
    # lamella.landing_state = microscope.get_microscope_state()

    # mill the landing edge flat
    log_status_message(lamella, "SELECT_LANDING_POSITION")
    settings.image.hfw = fcfg.REFERENCE_HFW_MEDIUM
    settings.image.beam_type = BeamType.ION
    settings.image.save = False

    eb_image, ib_image = acquire.take_reference_images(microscope=microscope, image_settings=settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    
    # log the protocol
    # TODO: change to prepare-landing protocol
    stages = get_milling_stages("flatten", settings.protocol["milling"])
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Select the landing position and prepare (mill) the area for {lamella.petname}. Press Continue when done.", 
        validate=True)
    
    lamella.protocol["flatten"] = deepcopy(get_protocol_from_stages(stages))
    lamella.protocol["flatten"]["point"] = stages[0].pattern.point.to_dict()
    lamella.landing_state = microscope.get_microscope_state()

    # take reference images
    log_status_message(lamella, "LANDING_REFERENCE_IMAGES")
    acquire.take_set_of_reference_images(
        microscope,
        settings.image,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename="ref_landing",
    )

    lamella.landing_selected = True

    return lamella


def select_lamella_positions(
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
):

    # reference images
    settings.image.hfw = fcfg.REFERENCE_HFW_LOW
    settings.image.beam_type = BeamType.ELECTRON
    settings.image.save = True
    settings.image.path = experiment.path
    settings.image.filename = f"initial_setup_grid_{fibsem_utils.current_timestamp_v2()}"
    eb_image,ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    select_another = get_current_lamella(experiment, parent_ui)

    if select_another:
        trench_start_position = fibsem_utils._get_position(settings.protocol["options"]["trench_start_position"])
        microscope.safe_absolute_stage_movement(trench_start_position)

    # allow the user to select additional lamella positions
    while select_another:
        logging.info(f"Selecting Lamella Position: {select_another}")
        lamella = select_initial_lamella_positions(
            microscope, settings, experiment, parent_ui
        )

        # save lamella data
        experiment.positions.append(deepcopy(lamella))
        experiment.save()

        # select another?
        select_another = get_current_lamella(experiment, parent_ui)

    # select landing positions
    select_landing_positions(microscope, settings, experiment, parent_ui)

    # finish setup
    finish_setup_autoliftout(microscope, experiment, parent_ui)

    return experiment


def finish_setup_autoliftout(
    microscope: FibsemMicroscope,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
):
    """Finish the setup stage for autolifout/autolamella"""

    parent_ui.set_instructions_msg(msg="Ready for AutoLiftout", pos=None, neg=None)

    for lamella in experiment.positions:
        if lamella.state.stage == AutoLamellaStage.SetupTrench:
            experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui, save_state=False)

            # administrative details        
            lamella = start_of_stage_update(microscope, lamella, next_stage=AutoLamellaStage.ReadyTrench, parent_ui=parent_ui, restore_state=False)
            experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui, save_state=False)

        

    logging.info(f"Selected {len(experiment.positions)} lamella for autoliftout.")
    log_status_message_raw(f"{AutoLamellaStage.SetupTrench.name}", "FINISHED")
    



def validate_needle_insertion(
    microscope: FibsemMicroscope, needle_stage_height_limit: float = 3.7e-3
) -> None:
    while validation.validate_stage_height_for_needle_insertion(
        microscope, needle_stage_height_limit
    ):
        ask_user(
            msg=f"""The system has identified the distance between the sample and the pole piece is less than {needle_stage_height_limit * 1000}mm. "
            "The needle will contact the sample, and it is unsafe to insert the needle. "
            "\nPlease manually refocus and link the stage, then press OK to continue. """, pos="Continue."
        )



# 
        
def prepare_manipulator_surface(microscope: FibsemMicroscope, settings: MicroscopeSettings, 
                                parent_ui: AutoLamellaUI, validate: bool = True,):

    workflow_stage = "PrepareManipulator"
    scan_rotation = microscope.get("scan_rotation", BeamType.ION)

    # save microscope state
    initial_state = microscope.get_microscope_state()
    
    # tilt stage flat
    microscope.safe_absolute_stage_movement(FibsemStagePosition(t = 0))
    
    # insert manipulator to eucentric z=-10
    log_status_message_raw(workflow_stage, "INSERT_MANIPULATOR")
    update_status_ui(parent_ui, f"Inserting Manipulator...")
    actions.move_needle_to_prepare_position(microscope)

    # move manipulator to centre of image
    beam_type = BeamType.ION
    settings.image.beam_type = beam_type

    features = [detection.NeedleTip(), detection.ImageCentre()] if np.isclose(scan_rotation, 0) else [detection.NeedleTipBottom(), detection.ImageCentre()]
    det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg="Prepare Manipulator")

    detection.move_based_on_detection(microscope, settings, det, beam_type, _move_system="manipulator")

    # mill prepare-manipulator (clean the manipulator surface)
    log_status_message_raw(workflow_stage, "MILL_PREPARE_MANIPULATOR_SURFACE")
    settings.image.filename = f"ref_prepare_manipulator_surface"
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    update_status_ui(parent_ui, f"Preparing Manipulator Surface...")

    # create rectangle pattern at tip (horizontal rect)
    stages = get_milling_stages("prepare-manipulator", ["milling"])

    # move pattern based on preparation method
    method = settings.protocol.get("method", "autolamella-serial-liftout")
    if method == "autolamella-serial-liftout":
        point = Point(0, 10e-6)
    if method == "autolamella-liftout":
        point = Point(-10e-6, 0)

    if not np.isclose(scan_rotation, 0):
        point.y *= -1.0

    stages = update_milling_ui(microscope, stages=stages, 
                msg=f"Press Run Milling to mill the manipulator. Press Continue when done.", 
                parent_ui=parent_ui, validate=validate)

    # reference images
    log_status_message_raw(workflow_stage, "REFERENCE_IMAGES")
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.filename = f"ref_prepare_manipulator_surface_final"
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    # retract manipulator
    microscope.retract_manipulator()

    # restore state
    microscope.set_microscope_state(initial_state)

    return 


def _prepare_manipulator_autoliftout(microscope: FibsemMicroscope, 
                                     settings: MicroscopeSettings, 
                                     parent_ui: AutoLamellaUI, 
                                     experiment: Experiment):

    # bookkeeping
    workflow_stage = "PrepareManipulator"
    log_status_message_raw(workflow_stage, "STARTED")
    validate = bool(settings.protocol["options"]["supervise"].get("prepare_manipulator", True))

    if experiment is not None:
        path = experiment.path
    else: 
        path = os.getcwd()

    settings.image.path = os.path.join(path, "prepare_manipulator")
    os.makedirs(settings.image.path, exist_ok=True)

    # assume manipulator is calibrated
    ret = ask_user(parent_ui=parent_ui, 
                   msg="Do you want to prepare the manipulator for autoliftout? Please ensure the manipulator is calibrated before starting.",
                    pos="Yes", neg="No")

    if ret is False:
        logging.info(f"Exiting prepare manipulator workflow. Manipulator is not calibrated")
        return
    
    # prepare manipulator surface
    prepare_manipulator_surface(microscope, settings, parent_ui, validate=validate)

    return


def _prepare_manipulator_serial_liftout(microscope: FibsemMicroscope, settings: MicroscopeSettings, parent_ui: AutoLamellaUI, experiment: Experiment = None):


    # bookkeeping
    workflow_stage = "PrepareManipulator"
    log_status_message_raw(workflow_stage, "STARTED")
    validate = bool(settings.protocol["options"]["supervise"].get("prepare_manipulator", True))
    scan_rotation = microscope.get("scan_rotation", BeamType.ION)

    if experiment is not None:
        path = experiment.path
    else: 
        path = os.getcwd()

    settings.image.path = os.path.join(path, "prepare_manipulator")
    os.makedirs(settings.image.path, exist_ok=True)

    # assume manipulator is calibrated
    ret = ask_user(parent_ui=parent_ui, 
                   msg="Do you want to prepare the manipulator for serial-liftout)? Please ensure the manipulator is calibrated before starting.",
                    pos="Yes", neg="No")

    if ret is False:
        logging.info(f"Exiting prepare manipulator workflow. Manipulator is not calibrated")
        return

    # move to landing grid
    log_status_message_raw(workflow_stage, "MOVE_TO_LANDING_GRID")
    update_status_ui(parent_ui, f"Moving to Landing Grid...")
    position = fibsem_utils._get_position(settings.protocol["options"]["landing_start_position"])
    microscope.safe_absolute_stage_movement(position)

    # move to milling orientation (18 degrees)
    t=np.deg2rad(settings.protocol["options"].get("lamella_tilt_angle", 18))
    microscope.safe_absolute_stage_movement(FibsemStagePosition(t=t))

    # ask the user to navigate to the desired location
    ask_user(
        parent_ui,
        msg=f"Please navigate to the desired location for preparing the copper adaptors. Press Continue when ready.",
        pos="Continue",
    )


    # mill prepare-copper-grid (clean the grid surface)
    settings.image.filename = f"ref_prepare_copper_grid"
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    update_status_ui(parent_ui, f"Preparing Copper Grid...")

    log_status_message_raw(workflow_stage, "MILL_PREPARE_COPPER_GRID")
    stages = get_milling_stages("prepare-copper-grid", settings.protocol["milling"])
    stages = update_milling_ui(microscope, stages=stages,
            msg=f"Press Run Milling to mill the grid preparation milling. Press Continue when done.", 
            parent_ui=parent_ui, validate=validate)
    
    # refernce images
    log_status_message_raw(workflow_stage, "REFERENCE_IMAGES")
    reference_images = acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=settings.image,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_prepare_copper_grid_final",
    )
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    # get milling state for return later
    milling_state = microscope.get_microscope_state()

    # rotate flat to ion
    log_status_message_raw(workflow_stage, "ROTATE_FLAT_TO_ION")
    update_status_ui(parent_ui, f"Rotating to Ion Beam...")
    microscope.move_flat_to_beam(beam_type=BeamType.ION)

    # mill prepare-copper-blocks (chain of blocks)
    settings.image.filename = f"ref_prepare_copper_blocks"
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    update_status_ui(parent_ui, f"Preparing Copper Blocks...")


    # get top pattern position
    log_status_message_raw(workflow_stage, "MILL_PREPARE_COPPER_BLOCKS")
    h1 = settings.protocol["milling"]["prepare-copper-blocks"]["stages"][0]["height"]
    h2 = settings.protocol["milling"]["prepare-copper-blocks"]["stages"][1]["height"]
    dy = h1/2 - h2/2
    points = [Point(0, 0), Point(0, dy)]
    
    # mill prepare-copper-blocks (chain of blocks)
    stages = get_milling_stages("prepare-copper-blocks", settings.protocol["milling"], point=points)
    stages = update_milling_ui(microscope, stages=stages, 
                msg=f"Press Run Milling to mill the copper blocks. Press Continue when done.", 
                parent_ui=parent_ui, validate=validate)
    

    # move back to milling orientation
    log_status_message_raw(workflow_stage, "MOVE_TO_MILLING_ORIENTATION")
    update_status_ui(parent_ui, f"Moving to Milling Orientation...")
    microscope.set_microscope_state(milling_state)

    # align coincidence
    if validate:
        ask_user(
            parent_ui,
            msg=f"Please align the coincidence of the beam. Press Continue when ready.",
            pos="Continue",
        )

    # update saved milling state
    milling_state = microscope.get_microscope_state()
    
    # optional, if manipulator already prepped
    ret = ask_user(parent_ui=parent_ui, 
                msg="Do you want to prepare the manipulator surface (mill the surface flat)?",
                pos="Yes", neg="Skip")

    if ret is True:

        # prepare manipulator surface
        prepare_manipulator_surface(microscope, settings, parent_ui, validate=validate)

    # insert manipulator to eucentric z=-10
    log_status_message_raw(workflow_stage, "INSERT_MANIPULATOR")
    update_status_ui(parent_ui, f"Inserting Manipulator...")
    actions.move_needle_to_prepare_position(microscope)

    # polish surfaces flat, cleaning cross section?
    ask_user(
        parent_ui,
        msg=f"Confirm that both surfaces are flat. Polish with milling if required. Press Continue when ready.",
        pos="Continue",
    )

    # move manipulator to centre of image
    log_status_message_raw(workflow_stage, "MOVE_TO_WELD_POSITION")
    for beam_type in [BeamType.ELECTRON, BeamType.ION]:
        
        settings.image.hfw = fcfg.REFERENCE_HFW_HIGH 
        settings.image.beam_type = beam_type

        # detect manipulator and user defined feature
        features = [detection.NeedleTip(), detection.CoreFeature()] if np.isclose(scan_rotation, 0) else [detection.NeedleTipBottom(), detection.CoreFeature()]
        det = update_detection_ui(microscope, settings, features, parent_ui, validate, msg="Prepare Manipulator")

        # move manipulator to target position
        detection.move_based_on_detection(microscope, settings, det, beam_type, _move_system="manipulator")

    # reference images
    log_status_message_raw(workflow_stage, "REFERENCE_IMAGES")
    settings.image.filename = f"ref_prepare_weld_position"
    settings.image.hfw = fcfg.REFERENCE_HFW_ULTRA
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    # respositon weld
    # TODO: detect the weld position
    log_status_message_raw(workflow_stage, "MILL_COPPER_WELD")
    stages = get_milling_stages("prepare-copper-weld", settings.protocol["milling"])
    stages = update_milling_ui(microscope, stages=stages,
                        msg=f"Press Run Milling to weld the copper block. Press Continue when done.", 
                        parent_ui=parent_ui, validate=validate)

    # reference images
    log_status_message_raw(workflow_stage, "REFERENCE_IMAGES")
    settings.image.filename = f"ref_prepare_copper_release"
    settings.image.hfw = fcfg.REFERENCE_HFW_SUPER
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)
    
    # release copper block
    log_status_message_raw(workflow_stage, "MILL_COPPER_RELEASE")
    ret = False
    while ret is False:

        # mill prepare-copper-release (release the copper block)
        stages = get_milling_stages("prepare-copper-release", settings.protocol["milling"])
        stages = update_milling_ui(microscope, stages=stages, 
                msg=f"Press Run Milling to mill copper block release. Press Continue when done.", 
                        parent_ui=parent_ui, validate=validate)

        # check for release
        ret = ask_user(parent_ui=parent_ui, 
                    msg="Has the copper block been released?",
                    pos="Yes", neg="No")

    # reference images
    log_status_message_raw(workflow_stage, "REFERENCE_IMAGES")
    settings.image.filename = f"ref_prepare_manipulator_final"
    settings.image.hfw = fcfg.REFERENCE_HFW_HIGH
    settings.image.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, settings.image)
    set_images_ui(parent_ui, eb_image, ib_image)

    # retract manipulator
    microscope.retract_manipulator()

    # logging
    log_status_message_raw(workflow_stage, "FINISHED")
    
    return 

PREPARE_MANIPULATOR_WORKFLOW = {
    "liftout": _prepare_manipulator_autoliftout,
    "serial-liftout": _prepare_manipulator_serial_liftout
}


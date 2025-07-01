import logging
import time
from collections import Counter
from copy import deepcopy
from pprint import pprint
from typing import List

import numpy as np
from fibsem import acquire, alignment, calibration
from fibsem import config as fcfg
from fibsem import utils as fibsem_utils
from fibsem.detection import detection
from fibsem.detection.detection import (
    CopperAdapterBottomEdge,
    CopperAdapterTopEdge,
    ImageCentre,
    LamellaBottomEdge,
    LamellaCentre,
    LamellaLeftEdge,
    LamellaRightEdge,
    LamellaTopEdge,
    LandingGridCentre,
    LandingGridLeftEdge,
    LandingGridRightEdge,
    LandingPost,
    NeedleTip,
    VolumeBlockBottomEdge,
    VolumeBlockBottomLeftCorner,
    VolumeBlockBottomRightCorner,
    VolumeBlockCentre,
    VolumeBlockTopEdge,
    VolumeBlockTopLeftCorner,
    VolumeBlockTopRightCorner,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import (
    FibsemMillingStage,
    get_milling_stages,
    get_protocol_from_stages,
)
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    MicroscopeSettings,
    MicroscopeState,
    ImageSettings,
    Point,
)

from autolamella.structures import (
    AutoLamellaProtocol,
    AutoLamellaStage,
    Experiment,
    Lamella,
    LamellaState,
    create_new_lamella,
)

from autolamella.ui import AutoLamellaUI
from autolamella.workflows import actions
from autolamella.workflows.autoliftout import (
    end_of_stage_update,
    log_status_message,
    mill_lamella,
    setup_lamella,
    start_of_stage_update,
    log_status_message_raw,
)
from autolamella.workflows.core import (
    align_feature_coincident,
    mill_trench,
    mill_undercut,
)
from autolamella.workflows.ui import (
    ask_user,
    set_images_ui,
    update_detection_ui,
    update_experiment_ui,
    update_milling_ui,
    update_status_ui,
)

# serial workflow functions

def liftout_lamella(
    microscope: FibsemMicroscope,
    protocol: AutoLamellaProtocol,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:

    # bookkeeping
    validate = protocol.supervision[lamella.workflow]
    image_settings = protocol.configuration.image
    image_settings.path = lamella.path

    # move to liftout angle...
    log_status_message(lamella, "MOVE_TO_LIFTOUT_POSITION")
    update_status_ui(parent_ui, f"{lamella.info} Moving to Liftout Position...")

    # move the stage flat to ion beam
    microscope.move_flat_to_beam(
        beam_type=BeamType.ION,
    )

    # get alignment feature
    scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)
    feature = VolumeBlockBottomEdge() if np.isclose(scan_rotation, 0) else VolumeBlockTopEdge()

    # align feature so beams are coincident
    lamella = align_feature_coincident(microscope=microscope, 
                             image_settings=image_settings, 
                              lamella=lamella,
                              checkpoint= protocol.options.checkpoint,
                              parent_ui=parent_ui, 
                              validate=validate, 
                              hfw=fcfg.REFERENCE_HFW_MEDIUM,
                              feature=feature)

    # insert the manipulator for liftout
    log_status_message(lamella, "INSERT_MANIPULATOR")
    update_status_ui(parent_ui, f"{lamella.info} Inserting Manipulator...")
    microscope.insert_manipulator("PARK")

    # reference images
    image_settings.save = True
    image_settings.hfw = fcfg.REFERENCE_HFW_MEDIUM
    image_settings.filename = f"ref_{lamella.state.stage.name}_manipualtor_inserted"
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # align manipulator to top of lamella in electron

    log_status_message(lamella, f"NEEDLE_EB_DETECTION_0")
    update_status_ui(parent_ui, f"{lamella.info} Moving Manipulator to Lamella...")

    image_settings.beam_type = BeamType.ELECTRON
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

    # DETECT COPPER ADAPTER, VOLUME TOP
    scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ELECTRON)
    features = [CopperAdapterTopEdge(), VolumeBlockBottomEdge()] if np.isclose(scan_rotation, 0) else [CopperAdapterBottomEdge(), VolumeBlockTopEdge()]
    
    # TODO: FAIL HERE
    det = update_detection_ui(microscope=microscope, 
                              image_settings=image_settings, 
                              checkpoint=protocol.options.checkpoint, 
                              features=features, parent_ui=parent_ui, 
                              validate=validate, msg=lamella.info)

    # TODO: do we offset this? Align the top edges -> move down by half the blcok? or just align the centre of the block

    # MOVE TO VOLUME BLOCK TOP
    detection.move_based_on_detection(
        microscope, det, beam_type=image_settings.beam_type, move_x=True,
         _move_system="manipulator"
    )

    if validate:
        # reference images
        image_settings.save = True
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
        image_settings.filename = f"ref_{lamella.state.stage.name}_manipualtor_validation_electron"
        eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)
        response = ask_user(parent_ui, 
                            msg=f"No more movements in Electron, move manualy if required. Press Continue to proceed for {lamella.petname}.", 
                            pos="Continue")

    # align manipulator to top of lamella in ion x3
    HFWS = [fcfg.REFERENCE_HFW_LOW, fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH, fcfg.REFERENCE_HFW_SUPER]

    for i, hfw in enumerate(HFWS):

        log_status_message(lamella, f"NEEDLE_IB_DETECTION_{i:02d}")
        update_status_ui(parent_ui, f"{lamella.info} Moving Manipulator to Lamella...")

        image_settings.beam_type = BeamType.ION
        image_settings.hfw = hfw

        # DETECT COPPER ADAPTER, LAMELLA TOP
        scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)
        features = [CopperAdapterTopEdge(), VolumeBlockBottomEdge()] if np.isclose(scan_rotation, 0) else [CopperAdapterBottomEdge(), VolumeBlockTopEdge()]
        
        det = update_detection_ui(microscope, image_settings=image_settings,
                                  checkpoint=protocol.options.checkpoint,
                                    features=features, 
                                    parent_ui=parent_ui, 
                                    validate=validate, 
                                    msg=lamella.info)

        # MOVE TO VOLUME BLOCK TOP
        detection.move_based_on_detection(
            microscope, det, beam_type=image_settings.beam_type, move_x=True, 
            _move_system="manipulator"
        )

    # reference images
    image_settings.beam_type = BeamType.ION
    image_settings.hfw = fcfg.REFERENCE_HFW_SUPER
    image_settings.save = True
    image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_landed"
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # WELD
    log_status_message(lamella, "WELD_VOLUME_TO_COPPER")
    update_status_ui(parent_ui, f"{lamella.info} Welding Volume to Copper...")

    features = [VolumeBlockBottomEdge() if np.isclose(scan_rotation, 0) else VolumeBlockTopEdge()] 
    det = update_detection_ui(microscope=microscope, 
                              image_settings=image_settings, 
                              checkpoint=protocol.options.checkpoint, 
                              features=features, 
                              parent_ui=parent_ui, 
                              validate=validate, msg=lamella.info)

    # move the pattern to the top of the volume (i.e up by half the height of the pattern)
    _V_OFFSET = protocol.milling["liftout-weld"][0].pattern.height / 2
    if np.isclose(scan_rotation, 0):
        _V_OFFSET *= -1
    point = det.features[0].feature_m 
    point.y += _V_OFFSET

    stages = protocol.milling["liftout-weld"]
    for stage in stages:
        stage.pattern.point = point
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the weld for {lamella.petname}. Press Continue when done.", 
        validate=validate)
    
    lamella.protocol["liftout-weld"] = deepcopy(get_protocol_from_stages(stages[0]))

    # reference images
    image_settings.beam_type = BeamType.ION
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
    image_settings.save = True
    image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_weld"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # SEVER
    log_status_message(lamella, "SEVER_VOLUME_BLOCK")
    update_status_ui(parent_ui, f"{lamella.info} Sever Manipulator...")

    # repeat severing until the volume is free
    while True:
        image_settings.hfw = fcfg.REFERENCE_HFW_MEDIUM

        features = [VolumeBlockTopEdge() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge()] 
        det = update_detection_ui(microscope=microscope, 
                                  image_settings=image_settings, 
                                  checkpoint=protocol.options.checkpoint,
                                  features=features,
                                  parent_ui=parent_ui, 
                                  validate=validate, 
                                  msg=lamella.info)

        point = det.features[0].feature_m 
        set_images_ui(parent_ui, None, det.fibsem_image)

        stages = protocol.milling["liftout-sever"]
        stages[0].pattern.point = point
        stages = update_milling_ui(microscope, stages, parent_ui, 
            msg=f"Press Run Milling to sever for {lamella.petname}. Press Continue when done.", 
            validate=validate)
        
        lamella.protocol["liftout-sever"] = deepcopy(get_protocol_from_stages(stages))

        # reference images
        image_settings.hfw = fcfg.REFERENCE_HFW_MEDIUM
        image_settings.save = True
        image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_sever"
        acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)

        # RETRACT MANIPULATOR
        log_status_message(lamella, "RETRACT_MANIPULATOR")
        update_status_ui(parent_ui, f"{lamella.info} Retracting Manipulator...")

        # retract small and validate
        microscope.move_manipulator_corrected(dx=0, dy=0.5e-6, beam_type=BeamType.ION)
        image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_removal_initial"
        eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)

        # TODO: implemented automated detection for separation of volume from trench
        if validate:
            response = ask_user(parent_ui, 
                                msg=f"Press Continue to confirm to separation of volume for {lamella.petname}.", 
                                pos="Continue", 
                                neg="Repeat")

        if response:
            logging.info(f"Volume Severed for {lamella.petname}, response: {response}")
            break

    # retract slowly at first
    for i in range(10):
        microscope.move_manipulator_corrected(dx=0, dy=1e-6, beam_type=BeamType.ION)
        if i % 3 == 0:
            image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_removal_slow_{i:02d}"
            eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
            set_images_ui(parent_ui, eb_image, ib_image)
        time.sleep(1)

    # then retract quickly
    for i in range(3):
        microscope.move_manipulator_corrected(dx=0, dy=20e-6, beam_type=BeamType.ION)
        image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_removal_{i:02d}"
        eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)
        time.sleep(1)

    # take reference images
    log_status_message(lamella, "REFERENCE_IMAGES")
    reference_images = acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=image_settings,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_{lamella.state.stage.name}_final",
    )
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    # move needle to park position
    microscope.retract_manipulator()  # retracted needle not supported on tescan


    return lamella

# START_HERE

def land_lamella(
    microscope: FibsemMicroscope,
    protocol: AutoLamellaProtocol,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
) -> Lamella:

    # bookkeeping
    validate = protocol.supervision[lamella.workflow]
    image_settings: ImageSettings = protocol.configuration.image
    image_settings.path = lamella.path

    # # MOVE TO LANDING POSITION
    log_status_message(lamella, "MOVE_TO_LANDING_POSITION")
    update_status_ui(parent_ui, f"{lamella.info} Moving to Landing Position...")

    # reference images
    image_settings.save = True
    image_settings.hfw = fcfg.REFERENCE_HFW_MEDIUM
    image_settings.filename = f"ref_{lamella.state.stage.name}_start"
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    if validate:
        response = ask_user(parent_ui, 
                            msg=f"Press Continue to confirm to landing position for {lamella.petname}.", 
                            pos="Continue")

    # INSER MANIPUALTOR TO PARK
    log_status_message(lamella, "INSERT_MANIPULATOR")
    update_status_ui(parent_ui, f"{lamella.info} Inserting Manipulator...")

    # insert the needle for landing
    actions.move_needle_to_park_position(microscope)

    # reference images
    image_settings.save = True
    image_settings.hfw = fcfg.REFERENCE_HFW_LOW
    image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_inserted"
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)


    # align in electron beam
    log_status_message(lamella, "NEEDLE_EB_DETECTION_INITIAL")
    update_status_ui(parent_ui, f"{lamella.info} Moving Volume to Landing Grid...")

    image_settings.beam_type = BeamType.ELECTRON
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

    # detect volume top right corner, landing grid right edge 
    scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ELECTRON)
    features = [VolumeBlockTopEdge() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge(), LandingGridCentre()]  # TODO: SCAN_ROTATION FOR LANDING_POST
    det = update_detection_ui(microscope=microscope, 
                              image_settings=image_settings, 
                              checkpoint=protocol.options.checkpoint, 
                              features=features, parent_ui=parent_ui, 
                              validate=validate, msg=lamella.info)
    set_images_ui(parent_ui, det.fibsem_image, None)

    detection.move_based_on_detection(
        microscope=microscope, det=det, 
        beam_type=image_settings.beam_type, 
        move_x=True,
         _move_system="manipulator"
    )

    if validate:
        # reference images
        image_settings.save = True
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
        image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_validation_electron"
        eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)

        response = ask_user(parent_ui, msg=f"No more movements in Electron, Press Continue to confirm to manipulator position for {lamella.petname}.", pos="Continue")


    # DETECT LAMELLA BOTTOM EDGE, LandingGridCentre_TOP
    # align manipulator to top of lamella
    log_status_message(lamella, "NEEDLE_IB_DETECTION")
    update_status_ui(parent_ui, f"{lamella.info} Moving Volume to Landing Grid...")

    HFWS = [fcfg.REFERENCE_HFW_LOW, fcfg.REFERENCE_HFW_MEDIUM]#, fcfg.REFERENCE_HFW_HIGH]
    for i, hfw in enumerate(HFWS):

        log_status_message(lamella, f"NEEDLE_IB_DETECTION_{i:02d}")

        image_settings.beam_type = BeamType.ION
        image_settings.hfw = hfw

        # DETECT COPPER ADAPTER, LAMELLA TOP
        scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)
        features = [VolumeBlockTopEdge() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge(), LandingGridCentre()]  # TODO: SCAN_ROTATION FOR LANDING_POST
        det = update_detection_ui(microscope=microscope, 
                                image_settings=image_settings, 
                                checkpoint=protocol.options.checkpoint, 
                                features=features, parent_ui=parent_ui, 
                                validate=validate, msg=lamella.info)
        set_images_ui(parent_ui, None, det.fibsem_image)

        # offset above the grid
        det._offset = Point(0, -20e-6) 

        # MOVE TO LANDING POST
        detection.move_based_on_detection(
            microscope=microscope, det=det, 
            beam_type=image_settings.beam_type, 
            move_x=True,
            _move_system="manipulator"
        )
        log_status_message(lamella, "NEEDLE_IB_DETECTION")


    image_settings.beam_type = BeamType.ION
    image_settings.hfw = fcfg.REFERENCE_HFW_SUPER

    # # DETECT COPPER ADAPTER, LAMELLA TOP
    # scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)
    # features = [VolumeBlockTopEdge() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge(), LandingGridCentre()]  # TODO: SCAN_ROTATION FOR LANDING_POST
    #     det = update_detection_ui(microscope=microscope, 
    #                             image_settings=image_settings, 
    #                             checkpoint=protocol.options.checkpoint, 
    #                             features=features, parent_ui=parent_ui, 
    #                             validate=validate, msg=lamella.info)    
    # set_images_ui(parent_ui, None, det.fibsem_image)
    
    # # MOVE TO LANDING GRID
    # detection.move_based_on_detection(
    #     microscope=microscope, det=det, 
    #     beam_type=image_settings.beam_type, 
    #     move_x=True,
    #     _move_system="manipulator"
    # )

    # TODO: check if the volume is wider than the landing grid
    # mill away excess width

    # detect the left and right edges of the landing grid
    # detect the left and right top corners of the volume block
    # the distance to move is the distance in x between the left grid edge, and left volume corner
    # the grid width is the distance between the left and right grid edges
    # the offset for the milling is the grid width from the left corner of the volume block + distance to move
    # the milling is a rectangle with width of grid width - volume width
    # height is the height of the volume block?

    # scan_rotation 180
    if np.isclose(scan_rotation, 0):
        features = [VolumeBlockTopLeftCorner(), VolumeBlockTopRightCorner(), LandingGridLeftEdge(), LandingGridRightEdge()]
    else:      
        features = [VolumeBlockBottomLeftCorner(), VolumeBlockBottomRightCorner(), LandingGridLeftEdge(), LandingGridRightEdge()]

    det = update_detection_ui(microscope=microscope, 
                            image_settings=image_settings, 
                            checkpoint=protocol.options.checkpoint, 
                            features=features, parent_ui=parent_ui, 
                            validate=validate, msg=lamella.info)    
    set_images_ui(parent_ui, None, det.fibsem_image)

    # get individual features
    vol_block_left, vol_block_right, grid_left_edge, grid_right_edge = det.features
    
    # get properties
    vol_block_width = vol_block_left.px.distance(vol_block_right.px)._to_metres(det.pixelsize).x
    grid_width = grid_left_edge.px.distance(grid_right_edge.px)._to_metres(det.pixelsize).x
    diff = abs(vol_block_width - grid_width)

    logging.info(f"Volume Block Width: {vol_block_width} um")
    logging.info(f"Landing Grid Width: {grid_width} um")
    logging.info(f"Excess Width: {diff} um")

    # TODO: handle the case where the block is smaller than the grid? abort?
 
    # amount to move the manipulator to align left edge / volume left corner
    dx = vol_block_left.px.distance(grid_left_edge.px)._to_metres(det.pixelsize).x
    logging.info(f"Move Manipulator by: {dx} um") 
    # TODO: I dont think this is corrected for scan rotation^^^, but move_based_on detection is
    if not np.isclose(scan_rotation, 0):
        dx *= -1
    microscope.move_manipulator_corrected(dx=dx, dy=0, beam_type=BeamType.ION)

    # TODO: only do this is required?

    ########### THIN VOLUME BLOCK

    # reference images
    image_settings.hfw = fcfg.REFERENCE_HFW_SUPER
    image_settings.save = True
    image_settings.filename = f"ref_{lamella.state.stage.name}_match_volume_and_grid_width_pre"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # draw milling pattern to remove excess material
    pt = vol_block_right.feature_m # poisition of milling pattern
    pt.x += dx - diff / 2
    pt.y -= 20e-6
    
    logging.info(f"Milling Point: {pt}")

    # mill thin
    stages = protocol.milling["landing-thin"]
    stages[0].pattern.width = diff
    stages[0].pattern.point = pt
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the thinning pattern for {lamella.petname}. Press Continue when done.", 
        validate=validate)

    lamella.protocol["landing-thin"] = deepcopy(get_protocol_from_stages(stages))

    # reference images
    image_settings.hfw = fcfg.REFERENCE_HFW_SUPER
    image_settings.save = True
    image_settings.filename = f"ref_{lamella.state.stage.name}_match_volume_and_grid_width_post"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)


    ########### FINAL MOVEMENT

    # align in ion beam
    log_status_message(lamella, "NEEDLE_IB_DETECTION_FINAL")
    update_status_ui(parent_ui, f"{lamella.info} Moving Volume to Landing Grid...")

    image_settings.beam_type = BeamType.ION
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

    # DETECT COPPER ADAPTER, LAMELLA TOP
    scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)
    features = [VolumeBlockTopLeftCorner() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge(), LandingGridLeftEdge()]  # TODO: SCAN_ROTATION FOR LANDING_POST
    det = update_detection_ui(microscope=microscope, 
                            image_settings=image_settings, 
                            checkpoint=protocol.options.checkpoint, 
                            features=features, parent_ui=parent_ui, 
                            validate=validate, msg=lamella.info)    
    set_images_ui(parent_ui, None, det.fibsem_image)
    
    # MOVE TO LANDING GRID
    detection.move_based_on_detection(
        microscope=microscope, det=det, 
        beam_type=image_settings.beam_type, 
        move_x=True,
        _move_system="manipulator"
    )

    if validate:
        response = ask_user(parent_ui, 
                            msg=f"No more movements, move manually if required, Press Continue to continue to landing welds for {lamella.petname}.", 
                            pos="Continue")


    # WELD BOTH SIDES
    log_status_message(lamella, "WELD_LAMELLA_TO_POST")

    image_settings.beam_type = BeamType.ION
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

    features = [VolumeBlockTopLeftCorner(), VolumeBlockTopRightCorner()] if np.isclose(scan_rotation, 0) else [VolumeBlockBottomLeftCorner(), VolumeBlockBottomRightCorner()]
    det = update_detection_ui(microscope=microscope, 
                            image_settings=image_settings, 
                            checkpoint=protocol.options.checkpoint, 
                            features=features, parent_ui=parent_ui, 
                            validate=validate, msg=lamella.info)       
    set_images_ui(parent_ui, None, det.fibsem_image)

    # get the points
    left_corner = det.features[0].feature_m 
    right_corner = det.features[1].feature_m

    # add some offset in y
    v_offset = 2e-6  # half of recommended 4um height
    left_corner.y  +=  v_offset
    right_corner.y +=  v_offset

    # mill welds
    stages = protocol.milling["landing-weld"]
    stages[0].pattern.point = left_corner
    stages[1].pattern.point = right_corner 
    stages = update_milling_ui(microscope, stages, parent_ui, 
        msg=f"Press Run Milling to mill the weld for {lamella.petname}. Press Continue when done.", 
        validate=validate)
    
    lamella.protocol["landing-weld"] = deepcopy(get_protocol_from_stages(stages))

    # reference images
    image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
    image_settings.save = True
    image_settings.filename = f"ref_{lamella.state.stage.name}_weld_volume_to_post"
    acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    ########################
    # move manipulator up 50-100 nm to create strain
    dy = 100e-9
    microscope.move_manipulator_corrected(dx=0, dy=dy, beam_type=BeamType.ION)

    ######################### 

    # SEVER
    lamella = sever_lamella_block(microscope=microscope, 
                                  protocol=protocol, 
                                  image_settings=image_settings, 
                                  lamella=lamella, 
                                  parent_ui=parent_ui, 
                                  validate=validate)

    # RETRACT MANIPULATOR
    log_status_message(lamella, "RETRACT_MANIPULATOR")
    update_status_ui(parent_ui, f"{lamella.info} Retracting Manipulator...")

    # move up slowly at first
    for i in range(10):
        microscope.move_manipulator_corrected(dx=0, dy=1e-6, beam_type=BeamType.ION)
        if i % 5 == 0:
            image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_removal_slow{i:02d}"
            eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
            set_images_ui(parent_ui, eb_image, ib_image)
        time.sleep(1)
    
    # move manipulator up
    microscope.move_manipulator_corrected(dx=0, dy=100e-6, beam_type=BeamType.ION)

    # move needle to park position
    microscope.retract_manipulator()  # retracted needle not supported on tescan

    # take reference images
    log_status_message(lamella, "REFERENCE_IMAGES")
    reference_images = acquire.take_set_of_reference_images(
        microscope=microscope,
        image_settings=image_settings,
        hfws=[fcfg.REFERENCE_HFW_MEDIUM, fcfg.REFERENCE_HFW_HIGH],
        filename=f"ref_{lamella.state.stage.name}_final",
    )
    set_images_ui(parent_ui, reference_images.high_res_eb, reference_images.high_res_ib)

    return lamella

def sever_lamella_block(microscope: FibsemMicroscope,
    protocol: AutoLamellaProtocol,
    image_settings: ImageSettings,
    lamella: Lamella,
    parent_ui: AutoLamellaUI,
    validate: bool = True) -> Lamella:

    # bookkeeping
    scan_rotation = microscope.get("scan_rotation", beam_type=BeamType.ION)

    # SEVER
    confirm_severed = False
    i = 0
    while confirm_severed is False:
        log_status_message(lamella, f"SEVER_LAMELLA_BLOCK_{i:02d}")
        update_status_ui(parent_ui, f"{lamella.info} Severing Lamella Block...")

        image_settings.beam_type = BeamType.ION
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

        features = [VolumeBlockTopEdge() if np.isclose(scan_rotation, 0) else VolumeBlockBottomEdge()]  
        det = update_detection_ui(microscope=microscope, 
                                image_settings=image_settings, 
                                checkpoint=protocol.options.checkpoint, 
                                features=features, parent_ui=parent_ui, 
                                validate=validate, msg=lamella.info) 
        set_images_ui(parent_ui, None, det.fibsem_image)
                
        point = det.features[0].feature_m

        stages = protocol.milling["landing-sever"]
        stages[0].pattern.point = point
        stages = update_milling_ui(microscope, stages, parent_ui, 
            msg=f"Press Run Milling to sever for {lamella.petname}. Press Continue when done.", 
            validate=validate)

        lamella.protocol["landing-sever"] = deepcopy(get_protocol_from_stages(stages))
        
        # reference images
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH
        image_settings.save = True
        image_settings.filename = f"ref_{lamella.state.stage.name}_sever_volume_block_{i:02d}"
        eb_image, ib_image = acquire.take_reference_images(microscope=microscope, image_settings=image_settings)
        set_images_ui(parent_ui, eb_image, ib_image)

        # move up slowly at first
        for j in range(3):
            microscope.move_manipulator_corrected(dx=0, dy=50e-9, beam_type=BeamType.ION)
            image_settings.filename = f"ref_{lamella.state.stage.name}_manipulator_removal_check{j:02d}"
            acquire.take_reference_images(microscope, image_settings)
            time.sleep(1)

        # confirm that the lamella is free     
        log_status_message(lamella, f"CONFIRM_LAMELLA_SEVER_{i:02d}")
        update_status_ui(parent_ui, f"{lamella.info} Confirming Lamella has been Severed...")

        image_settings.beam_type = BeamType.ION
        image_settings.hfw = fcfg.REFERENCE_HFW_HIGH

        features = [VolumeBlockTopEdge(), LamellaBottomEdge()] if np.isclose(scan_rotation, 0) else [VolumeBlockBottomEdge(), LamellaTopEdge()]  
        det = update_detection_ui(microscope=microscope, 
                                image_settings=image_settings, 
                                checkpoint=protocol.options.checkpoint, 
                                features=features, parent_ui=parent_ui, 
                                validate=validate, msg=lamella.info) 
        # if the distance is less than the threshold, then the lamella is not severed
        threshold = protocol.tmp.get("landing-sever-threshold", 0.5e-6)
        if abs(det.distance.y) < threshold:
            logging.info(f"Lamella Not Severed: {det.distance.y} < {threshold}")
            logging.debug({"msg": "check_volume_sever",  "detected_features": det.to_dict(), "threshold": threshold})   
            confirm_severed = False
        
        # check with the user
        if validate:
            response = ask_user(parent_ui, 
                                msg=f"Confirm Lamella has been severed for {lamella.petname}. "
                                 f"Distance measured was {det.distance.y*1e6:.2f} um. (Threshold = {threshold*1e6:.2f}) um", 
                                pos="Confirm", neg="Retry")
            confirm_severed = response

        i += 1

    return lamella


# serial workflow functions
SERIAL_WORKFLOW_STAGES = {
    AutoLamellaStage.MillTrench: mill_trench,
    AutoLamellaStage.MillUndercut: mill_undercut,
    AutoLamellaStage.LiftoutLamella: liftout_lamella,
    AutoLamellaStage.LandLamella: land_lamella,
    AutoLamellaStage.SetupLamella: setup_lamella,
    AutoLamellaStage.MillRough: mill_lamella,
    AutoLamellaStage.MillPolishing: mill_lamella,
}

def run_serial_liftout_workflow(
    microscope: FibsemMicroscope,
    protocol: AutoLamellaProtocol,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
) -> Experiment:
    """Run the serial AutoLiftout workflow for a given experiment. """

    update_status_ui(parent_ui, "Starting AutoLiftout Workflow...")
    logging.info(
        f"Serial Workflow started for {len(experiment.positions)} lamellae."
    )

    image_settings = protocol.configuration.image

    image_settings.save = False
    image_settings.path = experiment.path
    image_settings.filename = f"{fibsem_utils.current_timestamp()}"

    # standard workflow
    lamella: Lamella
    for lamella in experiment.positions:
        if lamella.is_failure:
            logging.info(f"Skipping {lamella.petname} due to failure.")
            continue  # skip failures

        while lamella.state.stage.value < AutoLamellaStage.LiftoutLamella.value:
            next_stage = AutoLamellaStage(lamella.state.stage.value + 1)
            if True:
                msg = (
                    f"""Continue Lamella {(lamella.petname)} from {next_stage.name}?"""
                )
                response = ask_user(parent_ui, msg=msg, pos="Continue", neg="Skip")

            else:
                response = True

            # update image settings (save in correct directory)
            image_settings.path = lamella.path

            if response:
                # reset to the previous state
                lamella = start_of_stage_update(
                    microscope, lamella, next_stage=next_stage, parent_ui=parent_ui
                )

                # run the next workflow stage
                lamella = SERIAL_WORKFLOW_STAGES[next_stage](
                    microscope=microscope,
                    protocol=protocol,
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


def run_serial_liftout_landing(    
    microscope: FibsemMicroscope,
    protocol: AutoLamellaProtocol,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
) -> Experiment:
    """Run the serial AutoLiftout workflow for landing a given experiment. """

    update_status_ui(parent_ui, "Starting Serial Liftout (Landing) Workflow...")
    logging.info(
        f"Serial Landing Workflow started for {len(experiment.positions)} lamellae."
    )

    lamella = experiment.positions[0]
    image_settings = protocol.configuration.image
    image_settings.save = False
    image_settings.path = lamella.path
    image_settings.filename = f"{fibsem_utils.current_timestamp()}"

    # move to landing position
    log_status_message(lamella, "MOVING_TO_LANDING_POSITION")
    update_status_ui(parent_ui, "Moving to Landing Position...")   
    # microscope.set_microscope_state(lamella.landing_state)

    # take images, 
    log_status_message(lamella, "REFERENCE_IMAGES")
    image_settings.hfw = fcfg.REFERENCE_HFW_MEDIUM
    image_settings.filename = f"ref_{lamella.state.stage.name}_start"
    image_settings.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # calculate landing positions
    # log_status_message(lamella, "CALCULATE_LANDING_POSITIONS")
    # update_status_ui(parent_ui, "Generating Landing Positions...")
    # logging.warning("The landing procedure is currently disabled...")
    # TODO: get the user to select landing positions:
    # move to position
    # prepare landing post/grid
    # save on experiment.landing_positions
    # then use these positions to land the lamellae
    # return experiment
    # positions = calculate_landing_positions(microscope=microscope, protocol=protocol)
    positions = experiment.landing_positions # NOTE: we really should pop these off the list once they are used by workflow

    if not positions:
        logging.warning("No landing positions found, please select landing positions.")
        return experiment # early exit for now, but should support selecting more positions here...
        experiment = select_landing_positions(microscope=microscope, 
                                              protocol=protocol, 
                                              experiment=experiment, 
                                              parent_ui=parent_ui)
        positions = experiment.landing_positions

    # see where we are in the workflow
    # _counter = Counter([p.state.stage.name for p in experiment.positions])
    # land_idx = _counter[AutoLamellaStage.LandLamella.name]
    # count how many at finished
    # finished_idx = _counter[AutoLamellaStage.Finished.name]
    land_idx =  len(experiment.at_stage(AutoLamellaStage.LandLamella))
    finished_idx = len(experiment.at_stage(AutoLamellaStage.Finished))

    # start of workflow
    response = ask_user(parent_ui, 
                        msg=f"Land Another Lamella? ({land_idx} Lamella Landed, {finished_idx} Lamella Finished)", 
                        pos="Continue", neg="Finish")

    while response:

        # TODO: this whole procedure needs to be rethought
        # create another lamella
        lamella = deepcopy(create_lamella_at_landing(microscope=microscope, 
                                                    experiment=experiment, 
                                                    protocol=protocol, 
                                                    positions=positions)) # TODO: FIX_IMMEDIATE
        experiment.positions.append(lamella)
        experiment.save()

        # advance workflow
        lamella = start_of_stage_update(microscope, lamella, 
            next_stage=AutoLamellaStage.LandLamella, parent_ui=parent_ui)

        # run the next workflow stage
        lamella = SERIAL_WORKFLOW_STAGES[AutoLamellaStage.LandLamella](
            microscope=microscope,
            protocol=protocol,
            lamella=lamella,
            parent_ui=parent_ui,
        )

        # advance workflow
        experiment = end_of_stage_update(microscope, experiment, lamella, parent_ui=parent_ui)
                
        # update ui
        update_experiment_ui(parent_ui, experiment)

        # land another lamella?
        _counter = Counter([p.state.stage.name for p in experiment.positions])
        land_idx = _counter[AutoLamellaStage.LandLamella.name]
        response = ask_user(parent_ui, msg=f"Land Another Lamella? ({land_idx} Lamella Landed), {finished_idx} Lamella Finished)", 
            pos="Continue", neg="Finish")

    return experiment

def select_landing_positions(microscope: FibsemMicroscope, 
                             protocol: AutoLamellaProtocol, 
                             experiment: Experiment, 
                             parent_ui: AutoLamellaUI) -> Experiment:
    """Select landing positions for the serial liftout workflow."""

    log_status_message_raw("Setup", "SELECT_LANDING_POSITIONS")
    
    # move to landing grid / position

    while True:
        response = ask_user(parent_ui,
                            msg=f"Prepare Landing Position and press Continue ({len(experiment.landing_positions)} selected).",
                            pos="Continue",
                            neg="Exit")

        # prepare the landing position, e.g. mill structures, line etc

        if not response:
            break

        stage_position = microscope.get_stage_position()
        stage_position.name = f"Landing Position {len(experiment.landing_positions):02d}"

        # add the position to the experiment
        experiment.landing_positions.append(deepcopy(stage_position))
        experiment.save()

        update_experiment_ui(parent_ui, experiment)

    return experiment

def create_lamella_at_landing(microscope: FibsemMicroscope, 
                   experiment: Experiment, 
                   protocol: AutoLamellaProtocol,
                   positions: List[FibsemStagePosition], 
                   ) -> Lamella:

    # create a new lamella for landing
    _counter = Counter([p.state.stage.name for p in experiment.positions])
    land_idx = _counter[AutoLamellaStage.LandLamella.name]

    print("COUNTER: ", _counter, land_idx)

    num = max(len(experiment.positions) + 1, 1)

    mprotocol = protocol.milling # TODO: migrate to milling_workflow
    tmp_protocol = deepcopy({k: get_protocol_from_stages(v) for k, v in mprotocol.items()})
    state = LamellaState(stage=AutoLamellaStage.LiftoutLamella, 
                         microscope_state=microscope.get_microscope_state())
    state.microscope_state.stage_position = deepcopy(positions[land_idx])
    lamella = create_new_lamella(experiment_path=experiment.path, 
                                    number=num, 
                                    state=state, 
                                    protocol=tmp_protocol)
    log_status_message(lamella, "CREATION")

    # set state
    # lamella.state.stage = AutoLamellaStage.LiftoutLamella
    # lamella.state.microscope_state = microscope.get_microscope_state()
    # lamella.state.microscope_state.stage_position = deepcopy(positions[land_idx])
    # lamella.landing_state = deepcopy(lamella.state.microscope_state)
    lamella.landing_selected = True

    print("LANDING POSITION")
    pprint(lamella.state.microscope_state.stage_position)
    print("---------")

    return lamella



# WORKFLOW

# SETUP
# MILL_TRENCH
# MILL_UNDERCUT
# LIFTOUT
# LOOP UNTIL NO MORE BULK / LANDING POSTS:
    # LANDING
# MILL_ROUGH
# MILL_REGULAR
# MILL_POLISH



def _prepare_manipulator(    
    microscope: FibsemMicroscope,
    settings: MicroscopeSettings,
    experiment: Experiment,
    parent_ui: AutoLamellaUI,
) -> None:

    # bookeeping
    validate = True
    settings.image.path = experiment.path


    # move to the landing grid

    # insert the manipulator

    # flatten to manipualtor (optional)

    # move manpiulator to the grid

    # weld

    # sever

    # retract manipulator

    return

def calculate_landing_positions(microscope: FibsemMicroscope, 
                                protocol: AutoLamellaProtocol) -> List[FibsemStagePosition]:
    """Calculate the landing positions for a given experiment."""
    # make user set the initial position for the landing grid
    # create a grid of positions, based on the grid sizes x, y
    # loop through each position until block is consumed

    # TODO: this should be done at the flat to sem orientation, calculate the grid -> in minimap show positions

    # base state = top left corner
    base_state = microscope.get_microscope_state()

    # get the landing grid protocol
    landing_grid_protocol = protocol.tmp["landing_grid"]
    grid_square = Point(landing_grid_protocol['x'], landing_grid_protocol['y'])
    n_rows, n_cols = landing_grid_protocol['rows'], landing_grid_protocol['cols']

    positions = []

    for i in range(n_rows):
        for j in range(n_cols):
            new_position = microscope.project_stable_move( 
                dx=grid_square.x*j, 
                dy=-grid_square.y*i, 
                beam_type=BeamType.ION, 
                base_position=base_state.stage_position)            
            
            # position name is number of position in the grid
            new_position.name = f"Landing Position {i*n_cols + j:02d}"
            
            positions.append(new_position)

    return positions
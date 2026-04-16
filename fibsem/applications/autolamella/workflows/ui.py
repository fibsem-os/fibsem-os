from __future__ import annotations
import logging
import time
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Sequence

from fibsem import milling
from fibsem.detection import detection
from fibsem.detection import utils as det_utils
from fibsem.detection.detection import DetectedFeatures, Feature
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemImage,
    FibsemRectangle,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.applications.autolamella.structures import Experiment
if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI

# CORE UI FUNCTIONS -> PROBS SEPARATE FILE
def _check_for_abort(parent_ui: Optional['AutoLamellaUI'], msg: str = "Workflow aborted by user.") -> bool:
    # headless mode
    if parent_ui is None:
        return False

    # Use task_manager's stop event if available, fall back to legacy
    task_manager = getattr(parent_ui, '_task_manager', None)
    if task_manager is not None:
        if task_manager.is_stopped:
            raise InterruptedError(msg)
    elif parent_ui._workflow_stop_event.is_set():
        raise InterruptedError(msg)
    return False

def update_detection_ui(
    microscope: FibsemMicroscope, 
    image_settings: ImageSettings, # TODO: deprecate
    checkpoint: str,
    features: Sequence[Feature], 
    parent_ui: Optional['AutoLamellaUI'] = None, 
    validate: bool = True, 
    msg: str = "Lamella", position: Optional[FibsemStagePosition] = None,
) -> DetectedFeatures:
    feat_str = ", ".join([f.name for f in features])
    if len(feat_str) > 15:
        feat_str = feat_str[:15] + "..."
    update_status_ui(parent_ui, f"{msg}: Detecting Features ({feat_str})...")

    det = detection.take_image_and_detect_features(
        microscope=microscope,
        image_settings=image_settings,
        features=features,
        point=position,
        checkpoint=checkpoint,
    )

    if validate and parent_ui is not None:
        ask_user(
            parent_ui,
            msg="Confirm Feature Detection. Press Continue to proceed.",
            pos="Continue",
            det=det,
        )

        det = parent_ui.det_widget._get_detected_features()

        # I need this to happen in the parent thread for it to work correctly
        parent_ui.detection_confirmed_signal.emit(True)

    else:
        det_utils.save_ml_feature_data(det)

    # TODO: set images in ui here
    return det


def set_images_ui(
    parent_ui: Optional['AutoLamellaUI'],
    eb_image: Optional[FibsemImage] = None,
    ib_image: Optional[FibsemImage] = None,
):
    # headless mode
    if parent_ui is None:
        return
    
    _check_for_abort(parent_ui)

    # TMP: prevent milling images overwriting existing
    if eb_image is not None:
        eb_image.metadata.image_settings.save = False
    if ib_image is not None:
        ib_image.metadata.image_settings.save = False

    INFO = {
        "msg": "Updating Images",
        "sem_image": eb_image,
        "fib_image": ib_image,

    }
    parent_ui.WAITING_FOR_UI_UPDATE = True
    parent_ui.workflow_update_signal.emit(INFO)

    while parent_ui.WAITING_FOR_UI_UPDATE:
        time.sleep(0.5)

def update_status_ui(parent_ui: Optional['AutoLamellaUI'], msg: str, workflow_info: Optional[str] = None) -> None:

    if parent_ui is None:
        logging.info(msg)
        return

    _check_for_abort(parent_ui)

    INFO = {
        "msg": msg,
        "workflow_info": workflow_info,
    }
    parent_ui.workflow_update_signal.emit(INFO)


def ask_user(
    parent_ui: Optional['AutoLamellaUI'],
    msg: str,
    pos: str,
    neg: Optional[str] = None,
    mill: Optional[bool] = None,
    det: Optional[DetectedFeatures] = None,
    spot_burn: Optional[bool] = None,
) -> bool:

    if parent_ui is None:
        logging.warning(f"User input requested in headless mode: {msg}, always returning True.")
        return True

    INFO = {
        "msg": msg,
        "pos": pos,
        "neg": neg,
        "det": det,
        "milling_enabled": mill,
        "spot_burn": spot_burn,
    }
    parent_ui.workflow_update_signal.emit(INFO)

    parent_ui.WAITING_FOR_USER_INTERACTION = True
    logging.info("WAITING_FOR_USER_INTERACTION...")
    while parent_ui.WAITING_FOR_USER_INTERACTION:
        _check_for_abort(parent_ui=parent_ui)
        time.sleep(1)

    INFO = {
        "msg": "",
    }
    parent_ui.workflow_update_signal.emit(INFO) # clear the message

    return parent_ui.USER_RESPONSE

def ask_user_continue_workflow(parent_ui, msg: str = "Continue with the next stage?", validate: bool = True):

    ret = True
    if validate:
        ret = ask_user(parent_ui=parent_ui, msg=msg, pos="Continue", neg="Exit")
    return ret

def update_alignment_area_ui(alignment_area: FibsemRectangle, parent_ui: Optional['AutoLamellaUI'],
        msg: str = "Edit Alignment Area", validate: bool = True) -> FibsemRectangle:
    """ Update the alignment area in the UI and return the updated alignment area."""
    
    _check_for_abort(parent_ui)

    # headless mode, return the alignment area   
    if parent_ui is None or not validate:
        return alignment_area

    INFO = {
        "msg": msg,
        "pos": "Continue",
        "alignment_area": alignment_area,
    }
    parent_ui.workflow_update_signal.emit(INFO)

    parent_ui.WAITING_FOR_USER_INTERACTION = True
    logging.info("WAITING_FOR_USER_INTERACTION...")
    while parent_ui.WAITING_FOR_USER_INTERACTION:
        _check_for_abort(parent_ui=parent_ui)
        time.sleep(1)

    _check_for_abort(parent_ui)

    INFO = {
        "msg": "Clearing Alignment Area",
        "alignment_area": "clear",
    }

    parent_ui.WAITING_FOR_UI_UPDATE = True
    parent_ui.workflow_update_signal.emit(INFO)
    while parent_ui.WAITING_FOR_UI_UPDATE:
        time.sleep(0.5)

    # retrieve the updated alignment area
    alignment_area = deepcopy(parent_ui.image_widget.get_alignment_area())

    return alignment_area

def select_poi_ui(parent_ui: Optional['AutoLamellaUI'],
                  msg: str = "Select Point of Interest",
                  validate: bool = True,
                  initial_poi: Optional[Point] = None) -> Optional[Point]:
    """Display a draggable POI marker on the FIB image; return the selected point in milling coordinates."""
    _check_for_abort(parent_ui)

    if parent_ui is None or not validate:
        return None

    INFO = {"msg": msg, "pos": "Continue", "poi_selection": True, "initial_poi": initial_poi}
    parent_ui.workflow_update_signal.emit(INFO)

    parent_ui.WAITING_FOR_USER_INTERACTION = True
    logging.info("WAITING_FOR_USER_INTERACTION (POI selection)...")
    while parent_ui.WAITING_FOR_USER_INTERACTION:
        _check_for_abort(parent_ui=parent_ui)
        time.sleep(1)

    _check_for_abort(parent_ui)

    # clear the layer and compute POI
    INFO = {"msg": "", "poi_selection": "clear"}
    parent_ui.WAITING_FOR_UI_UPDATE = True
    parent_ui.workflow_update_signal.emit(INFO)
    while parent_ui.WAITING_FOR_UI_UPDATE:
        time.sleep(0.5)

    return deepcopy(parent_ui.SELECTED_POI)

def update_experiment_ui(parent_ui: Optional['AutoLamellaUI'], experiment: Experiment) -> None:

    # headless mode
    if parent_ui is None:
        return

    parent_ui.update_experiment_signal.emit(deepcopy(experiment))

def update_spot_burn_parameters(parent_ui: 'AutoLamellaUI',
                                parameters: dict | None = None,
                                clear_spots: bool = False):
    """Update the spot burn parameters in the UI."""
    _check_for_abort(parent_ui)

    INFO = {
        "msg": "Updating Spot Burn Parameters",
        "spot_burn_parameters": parameters,
        "clear_spot_burn": clear_spots,
    }

    parent_ui.WAITING_FOR_UI_UPDATE = True
    parent_ui.workflow_update_signal.emit(INFO)
    while parent_ui.WAITING_FOR_UI_UPDATE:
        time.sleep(0.5)

def clear_spot_burn_ui(parent_ui: 'AutoLamellaUI'):
    """Clear the spot burn UI."""
    update_spot_burn_parameters(parent_ui=parent_ui, parameters=None, clear_spots=True)

import logging
from typing import TYPE_CHECKING, Optional

from fibsem import acquire
from fibsem import config as fcfg
from fibsem.applications.autolamella.workflows.ui import (
    set_images_ui,
    update_detection_ui,
    update_status_ui,
)
from fibsem.detection.detection import (
    Feature,
    LamellaCentre,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    ImageSettings,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.ui import AutoLamellaUI
    from fibsem.applications.autolamella.structures import Lamella

# CORE WORKFLOW STEPS
def log_status_message(lamella: 'Lamella', step: str):
    logging.debug({"msg": "status", "petname": lamella.name, "stage": lamella.status_info, "step": step})

def log_status_message_raw(stage: str, step: str, petname: str = "null"):
    logging.debug({"msg": "status", "petname": petname, stage: stage, "step": step })   


def align_feature_coincident(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    lamella: 'Lamella',
    checkpoint: str,
    parent_ui: Optional['AutoLamellaUI'],
    validate: bool,
    hfw: float = fcfg.REFERENCE_HFW_MEDIUM,
    feature: Feature = LamellaCentre(),
) -> 'Lamella':
    """Align the feature in the electron and ion beams to be coincident."""

    # bookkeeping
    features = [feature]

    # update status
    log_status_message(lamella, "ALIGN_FEATURE_COINCIDENT")
    update_status_ui(parent_ui, f"{lamella.info} Aligning Feature Coincident ({feature.name})...")
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.hfw = hfw
    image_settings.filename = f"ref_{lamella.task_state.name}_{feature.name}_align_coincident_ml"
    image_settings.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # detect
    det = update_detection_ui(microscope=microscope,
                              image_settings=image_settings,
                              features=features,
                              checkpoint=checkpoint,
                              parent_ui=parent_ui, 
                              validate=validate, 
                              msg=lamella.info, 
                              position=lamella.stage_position)

    microscope.stable_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
        beam_type=image_settings.beam_type
    )

    # Align ion so it is coincident with the electron beam
    image_settings.beam_type = BeamType.ION
    image_settings.hfw = hfw

    det = update_detection_ui(microscope=microscope,
                              image_settings=image_settings,
                              features=features,
                              checkpoint=checkpoint,
                              parent_ui=parent_ui, 
                              validate=validate, 
                              msg=lamella.info, 
                              position=lamella.stage_position)
    
    # align vertical
    microscope.vertical_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
    )

    # reference images
    image_settings.save = True
    image_settings.hfw = hfw
    image_settings.filename = f"ref_{lamella.task_state.name}_{feature.name}_align_coincident_final"
    sem_image, fib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, sem_image, fib_image)

    return lamella

def align_feature_beam_shift(microscope: FibsemMicroscope, 
                            image_settings: ImageSettings, 
                            lamella: 'Lamella', parent_ui: Optional['AutoLamellaUI'], 
                            validate: bool, 
                            beam_type: BeamType = BeamType.ELECTRON,
                            hfw: float = fcfg.REFERENCE_HFW_MEDIUM,
                            feature: Feature = LamellaCentre(), 
                            checkpoint: Optional[str] = None) -> 'Lamella':
    """Align the feature to the centre of the image using the beamshift."""

    # bookkeeping
    features = [feature]

    # update status
    log_status_message(lamella, "ALIGN_FEATURE_BEAM_SHIFT")
    update_status_ui(parent_ui, f"{lamella.info} Aligning Feature with beam shift ({feature.name})...")
    image_settings.beam_type = beam_type
    image_settings.hfw = hfw
    image_settings.filename = f"ref_{lamella.task_state.name}_{feature.name}_align_beam_shift_ml"
    image_settings.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    # detect
    det = update_detection_ui(microscope=microscope, 
                              image_settings=image_settings, 
                              checkpoint=checkpoint, 
                              features=features, 
                              parent_ui=parent_ui, 
                              validate=validate, 
                              msg=lamella.info, 
                              position=None, 
                              )

    # TODO: add movement modes; stable move, vertical move, beam shift

    microscope.beam_shift(
        dx=-det.features[0].feature_m.x,
        dy=-det.features[0].feature_m.y,
        beam_type=image_settings.beam_type
    )

    # reference images
    image_settings.save = True
    image_settings.hfw = hfw
    image_settings.filename = f"ref_{lamella.task_state.name}_{feature.name}_align_beam_shift_final"
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(parent_ui, eb_image, ib_image)

    return lamella

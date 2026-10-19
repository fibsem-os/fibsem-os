import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

from fibsem import acquire
from fibsem import config as fcfg
from fibsem.applications.autolamella.workflows.ui import (
    set_images_ui,
    update_status_ui,
)
from fibsem.detection.detection import (
    DetectedFeatures,
    Feature,
    LamellaCentre,
)
from fibsem.microscope import FibsemMicroscope
from fibsem.structures import (
    BeamType,
    ImageSettings,
)

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Lamella


# CORE WORKFLOW STEPS
def log_status_message(lamella: "Lamella", step: str):
    logging.debug(
        {
            "msg": "status",
            "petname": lamella.name,
            "stage": lamella.status_info,
            "step": step,
        }
    )


def log_status_message_raw(stage: str, step: str, petname: str = "null"):
    logging.debug({"msg": "status", "petname": petname, stage: stage, "step": step})


def align_feature_coincident(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
    lamella: "Lamella",
    checkpoint: str,
    run: Optional[Any],
    detect: Callable[..., DetectedFeatures],
    hfw: float = fcfg.REFERENCE_HFW_MEDIUM,
    feature: Feature = LamellaCentre(),
) -> "Lamella":
    """Align the feature in the electron and ion beams to be coincident.

    ``detect`` is how the two detections are asked: a task passes its own
    ``detect`` so they go on its record. ``run`` is the task's manager: where
    the status and the images go.
    """

    # bookkeeping
    features = [feature]

    # update status
    log_status_message(lamella, "ALIGN_FEATURE_COINCIDENT")
    update_status_ui(
        run, f"{lamella.info} Aligning Feature Coincident ({feature.name})..."
    )
    image_settings.beam_type = BeamType.ELECTRON
    image_settings.hfw = hfw
    image_settings.filename = (
        f"ref_{lamella.task_state.name}_{feature.name}_align_coincident_ml"
    )
    image_settings.save = True
    eb_image, ib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(run, eb_image, ib_image)

    # detect
    det = detect(
        image_settings=image_settings,
        checkpoint=checkpoint,
        features=features,
        position=lamella.stage_position,
        message=lamella.info,
    )

    microscope.stable_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
        beam_type=image_settings.beam_type,
    )

    # Align ion so it is coincident with the electron beam
    image_settings.beam_type = BeamType.ION
    image_settings.hfw = hfw

    det = detect(
        image_settings=image_settings,
        checkpoint=checkpoint,
        features=features,
        position=lamella.stage_position,
        message=lamella.info,
    )

    # align vertical
    microscope.vertical_move(
        dx=det.features[0].feature_m.x,
        dy=det.features[0].feature_m.y,
    )

    # reference images
    image_settings.save = True
    image_settings.hfw = hfw
    image_settings.filename = (
        f"ref_{lamella.task_state.name}_{feature.name}_align_coincident_final"
    )
    sem_image, fib_image = acquire.take_reference_images(microscope, image_settings)
    set_images_ui(run, sem_image, fib_image)

    return lamella

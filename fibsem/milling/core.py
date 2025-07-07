import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional

from fibsem import acquire, config as fcfg
from fibsem.microscope import FibsemMicroscope
from fibsem.milling import FibsemMillingStage
from fibsem.structures import (
    FibsemBitmapSettings,
    FibsemCircleSettings,
    FibsemImage,
    FibsemLineSettings,
    FibsemPatternSettings,
    FibsemRectangleSettings,
    ImageSettings,
    BeamType,
)
from fibsem.utils import current_timestamp_v2
from fibsem.milling.base import FibsemMillingTask, FibsemMillingTaskConfig

########################### SETUP


def setup_milling(
    microscope: FibsemMicroscope,
    milling_stage: FibsemMillingStage,
    config: FibsemMillingTaskConfig,
    reference_image: Optional[FibsemImage] = None,
):
    """Setup Microscope for FIB Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_stage (FibsemMillingStage): Milling Stage
        config (FibsemMillingTaskConfig): Task configuration containing alignment and imaging settings
        reference_image (Optional[FibsemImage], optional): Reference image for alignment. Defaults to None.
    """
    
    # acquire reference image for alignment if not provided
    if config.alignment.enabled and reference_image is None:
        reference_image = acquire_stage_reference_image(microscope, config, milling_stage.name)

    # apply task-level configuration to milling settings
    milling_stage.milling.hfw = config.hfw
    milling_stage.milling.milling_channel = config.milling_channel

    # set up milling settings
    microscope.setup_milling(mill_settings=milling_stage.milling)

    # align at the milling current to correct for shift
    if config.alignment.enabled:
        if reference_image is None:
            raise ValueError("Reference image is required for alignment but was not provided or acquired")
        
        from fibsem import alignment
        logging.info(f"FIB Aligning at Milling Current: {milling_stage.milling.milling_current:.2e}")
        alignment.multi_step_alignment_v2(
            microscope=microscope,
            ref_image=reference_image,
            beam_type=milling_stage.milling.milling_channel,
            steps=3,
            use_autocontrast=True,
        )  # high current -> damaging


# TODO: migrate run milling to take milling_stage argument, rather than current, voltage
def run_milling(
    microscope: FibsemMicroscope,
    milling_current: float,
    milling_voltage: float,
    asynch: bool = False,
) -> None:
    """Run Ion Beam Milling.

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        milling_current (float, optional): ion beam milling current. Defaults to None.
        asynch (bool, optional): flag to run milling asynchronously. Defaults to False.
    """
    microscope.run_milling(milling_current, milling_voltage, asynch)

def finish_milling(
    microscope: FibsemMicroscope, imaging_current: float = 20e-12, imaging_voltage: float = 30e3
) -> None:
    """Clear milling patterns, and restore to the imaging current.

    Args:
        microscope (FIbsemMicroscope): Fibsem microscope instance
        imaging_current (float, optional): Imaging Current. Defaults to 20e-12.
        imaging_voltage: Imaging Voltage. Defaults to 30e3.
    """
    # restore imaging current
    logging.info(f"Changing to Imaging Current: {imaging_current:.2e}")
    microscope.finish_milling(imaging_current=imaging_current, imaging_voltage=imaging_voltage)
    logging.info("Finished Ion Beam Milling.")

def draw_patterns(microscope: FibsemMicroscope, patterns: List[FibsemPatternSettings]) -> None:
    """Draw milling patterns on the microscope from the list of settings
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        patterns (List[FibsemPatternSettings]): List of milling patterns
    """
    for pattern in patterns:
        draw_pattern(microscope, pattern)

        
def draw_pattern(microscope: FibsemMicroscope, pattern: FibsemPatternSettings):
    """Draw a milling pattern from settings

    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        pattern_settings (FibsemPatternSettings): pattern settings
    """
    if isinstance(pattern, FibsemRectangleSettings):
        microscope.draw_rectangle(pattern)

    elif isinstance(pattern, FibsemLineSettings):
        microscope.draw_line(pattern)

    elif isinstance(pattern, FibsemCircleSettings):
        microscope.draw_circle(pattern)

    elif isinstance(pattern, FibsemBitmapSettings):
        microscope.draw_bitmap_pattern(pattern, pattern.path)

def convert_to_bitmap_format(path):
    import os

    from PIL import Image
    img=Image.open(path)
    a=img.convert("RGB", palette=Image.ADAPTIVE, colors=8)
    new_path = os.path.join(os.path.dirname(path), "24bit_img.tif")
    a.save(new_path)
    return new_path


############################# UTILS #############################

def acquire_images_after_milling(
    microscope: FibsemMicroscope,
    image_settings: ImageSettings,
) -> Tuple[FibsemImage, FibsemImage]:
    """Acquire images after milling for reference.
    Args:
        microscope (FibsemMicroscope): Fibsem microscope instance
        image_settings (ImageSettings): Imaging settings for acquiring images
    Returns:
        Tuple[FibsemImage, FibsemImage]: Tuple of acquired images (SEM, FIB).
    """
    # restore imaging conditions
    imaging_current = microscope.system.ion.beam.beam_current or 20e-12
    imaging_voltage = microscope.system.ion.beam.voltage or 30e3
    finish_milling(
        microscope=microscope,
        imaging_current=imaging_current,
        imaging_voltage=imaging_voltage,
    )

    # acquire images
    from fibsem import acquire
    return acquire.take_reference_images(microscope, image_settings)

def acquire_stage_reference_image(microscope: FibsemMicroscope, config: FibsemMillingTaskConfig, name: str) -> FibsemImage:
    """Acquire a reference image for the stage using the task configuration."""
    path = config.imaging.path
    if path is None:
        path = Path(fcfg.DATA_CC_PATH)
    image_settings = ImageSettings(
        hfw=config.hfw,
        dwell_time=1e-6,
        resolution=(1536, 1024),
        beam_type=config.milling_channel,
        reduced_area=config.alignment.rect,
        save=True,
        path=path,
        filename=f"ref_{name}_initial_alignment_{current_timestamp_v2()}",
    )
    return acquire.acquire_image(microscope, image_settings)


# QUERY: should List[FibsemMillingStage] be a class? that has it's own settings?
# E.G. should acquire images be set at that level, rather than at the stage level?

def mill_stages(
    microscope: FibsemMicroscope,
    stages: List[FibsemMillingStage], parent_ui=None) -> None:
    """Backwards compatible wrapper function to mill stages."""

    config = FibsemMillingTaskConfig(
        hfw=stages[0].milling.hfw,
        acquire_after_milling=stages[0].milling.acquire_images,
        alignment=stages[0].alignment,
        milling_channel=stages[0].milling.milling_channel,
        imaging=stages[0].imaging,
    )

    task = FibsemMillingTask(
        name="Milling Task",
        config=config,
        stages=stages,
    )
    task.run(microscope=microscope, parent_ui=parent_ui)

def run_milling_task(
    microscope: FibsemMicroscope,
    task: FibsemMillingTask,
    parent_ui=None,
):
    """Run a milling task (sequence of milling stages), with a progress bar and notifications."""

    # TODO: add task_id, stage_id

    initial_beam_shift = None

    stages = task.stages
    if isinstance(stages, FibsemMillingStage):
        stages = [stages]
    try:
        if hasattr(microscope, "milling_progress_signal"):
            if parent_ui: # TODO: tmp ladder to handle progress indirectly
                def _handle_progress(ddict: dict) -> None:
                    parent_ui.milling_progress_signal.emit(ddict)
            else:
                def _handle_progress(ddict: dict) -> None:
                    logging.info(ddict)
            microscope.milling_progress_signal.connect(_handle_progress)

        if task.config.alignment.enabled:
            task.reference_image = acquire_stage_reference_image(microscope=microscope, config=task.config, name=task.name)
        initial_beam_shift = microscope.get_beam_shift(beam_type=task.config.milling_channel)

        for idx, stage in enumerate(stages):
            start_time = time.time()
            if parent_ui:
                if parent_ui.STOP_MILLING:
                    raise Exception("Milling stopped by user.")

                msgd =  {"msg": f"Preparing: {stage.name}",
                        "progress": {"state": "start", 
                                    "start_time": start_time,
                                    "current_stage": idx, 
                                    "total_stages": len(stages),
                                    }}
                parent_ui.milling_progress_signal.emit(msgd)

            try:
                task.run_stage(microscope=microscope, stage=stage, asynch=False, parent_ui=parent_ui)

                # performance logging
                msgd = {"msg": "mill_stages", "idx": idx, "stage": stage.to_dict(), "start_time": start_time, "end_time": time.time()}
                logging.debug(f"{msgd}")

                # optionally acquire images after milling
                if task.config.acquire_after_milling:
                    filename = f"ref_milling_{stage.name.replace(' ', '-')}_finished_{str(start_time).replace('.', '_')}"
                    image_settings = task.config.imaging
                    image_settings.filename = filename
                    acquire_images_after_milling(microscope=microscope, image_settings=image_settings)

                if parent_ui:
                    parent_ui.milling_progress_signal.emit({"msg": f"Finished: {stage.name}"})
            except Exception as e:
                logging.error(f"Error running milling stage: {stage.name}, {e}")

        if parent_ui:
            parent_ui.milling_progress_signal.emit({"msg": f"Finished {len(stages)} Milling Stages. Restoring Imaging Conditions..."})

    except Exception as e:
        if parent_ui:
            import napari.utils.notifications
            napari.utils.notifications.show_error(f"Error while milling {e}")
        logging.error(e)
    finally:
        imaging_current = microscope.system.ion.beam.beam_current or 20e-12
        imaging_voltage = microscope.system.ion.beam.voltage or 30e3
        finish_milling(
            microscope=microscope,
            imaging_current=imaging_current,
            imaging_voltage=imaging_voltage,
        )
        # restore initial beam shift
        if initial_beam_shift is not None:
            microscope.set_beam_shift(initial_beam_shift, beam_type=task.config.milling_channel)
        if hasattr(microscope, "milling_progress_signal"):
            microscope.milling_progress_signal.disconnect(_handle_progress)
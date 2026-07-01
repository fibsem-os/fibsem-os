from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Literal,
    Type,
    Optional,
)
from copy import deepcopy
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.tasks import FibsemMillingTaskConfig, FibsemMillingStage
from fibsem.milling.base import FibsemMillingSettings
from fibsem.structures import CrossSectionPattern
import fibsem.utils as utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
    AcquireFluorescenceImageConfig,
)
from fibsem.applications.autolamella.workflows.tasks.base import (
    ALIGNMENT_REFERENCE_IMAGE_FILENAME,
    AutoLamellaTask,
)
from fibsem.applications.autolamella.workflows.ui import (
    ask_user,
)
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)

MILL_COINCIDENT_KEY = "mill_coincident"

DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY] = FibsemMillingTaskConfig(
    name="Coincident Milling",
    field_of_view=80e-6,
    stages=[
        FibsemMillingStage(
            name="Coincident Milling 01",
            milling=FibsemMillingSettings(milling_current=60e-12, application_file="Si-ccs"),
            pattern=RectanglePattern(width=9.0e-6, depth=4.0e-7, height=20e-6, 
                                  cross_section=CrossSectionPattern.CleaningCrossSection),
            strategy=CoincidenceMillingStrategy()
        )
    ],
)


@dataclass
class MillCoincidentTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillCoincidentTask."""

    acquire_sem: bool = field(
        default=True,
        metadata={
            "help": "Whether to acquire an SEM reference image",
            "label": "Acquire SEM Image",
        },
    )
    acquire_fib: bool = field(
        default=True,
        metadata={
            "help": "Whether to acquire a FIB reference image",
            "label": "Acquire FIB Image",
        },
    )
    acquire_fluorescence_images: bool = field(
        default=True,
        metadata={
            "label": "Acquire Fluorescence Images",
            "help": "Whether to acquire fluorescence images before and after coincident milling",
        },
    )
    orientation: Literal["SEM", "FIB", "MILLING"] = field(
        default="MILLING",
        metadata={"help": "The orientation to perform coincident milling in"},
    )
    channel_name: str = field(
        default="Red Channel",
        metadata={"help": "The fluorescence channel to use for coincident milling"},
    )
    task_type: ClassVar[str] = "MILL_COINCIDENT"
    display_name: ClassVar[str] = "Coincident Milling"

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy(
                {MILL_COINCIDENT_KEY: DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY]}
            )


class MillCoincidentTask(AutoLamellaTask):
    """Task to mill the coincident trench for a lamella."""

    config: MillCoincidentTaskConfig
    config_cls: ClassVar[Type[MillCoincidentTaskConfig]] = MillCoincidentTaskConfig

    def _run(self) -> None:
        """Run the task to mill the coincident trenches for a lamella."""

        if self.microscope.fm is None:
            raise ValueError(
                "Microscope does not have a fluorescence microscope attached. Cannot run MillCoincidentTask."
            )
        if (
            self.lamella.fluorescence_pose is None
            or self.lamella.fluorescence_pose.objective_position is None
        ):
            raise ValueError(
                f"Lamella {self.lamella.name} does not have an objective position set. Cannot run MillCoincidentTask."
            )

        # bookkeeping
        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # move to lamella milling position
        self.log_status_message("MOVE_TO_LAMELLA", "Moving to Lamella Position...")
        if (
            self.lamella.milling_pose is None
            or self.lamella.milling_pose.stage_position is None
        ):
            raise ValueError(
                f"Milling pose for {self.lamella.name} is not set. Please set the milling pose before milling the lamella."
            )
        milling_position = self.lamella.milling_pose.stage_position

        # TODO: don't allow user to change orientation? only allow MILLING???
        if (
            self.microscope.get_stage_orientation(milling_position)
            != self.config.orientation
        ):
            raise ValueError(
                f"Stage position {milling_position} is not in {self.config.orientation} orientation..."
            )
        self.microscope.set_microscope_state(self.lamella.milling_pose)

        # beam_shift alignment
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

        # reference images
        self._acquire_set_of_reference_images(image_settings)

        # QUERY: we cannot insert the objective when the stage is tilted, we may need to do a work around
        # 1. move to SEM orientation (t=0), insert objective,
        # 2. move back to milling angle
        # 3. move to objective position
        # 4. once task finished, retract obj again...
        if not self.microscope.fm.objective.state == "Inserted":
            logging.warning(
                "Objective is not inserted. Inserting the objective before acquiring fluorescence images."
            )
            self.microscope.fm.objective.insert()

        # Check for cancellation before each position
        if self._stop_event and self._stop_event.is_set():
            logging.info(
                f"{self.task_name}: {self.lamella.name} - Acquisition cancelled"
            )
            return

        # Move stage to the saved stage position and objective position
        self.microscope.fm.objective.move_absolute(self.lamella.fluorescence_pose.objective_position)

        # mill coincident
        self.log_status_message("MILL_COINCIDENT", "Milling Coincident Lamella...")
        milling_task_config = self.config.milling[MILL_COINCIDENT_KEY]
        milling_task_config.alignment.rect = self.lamella.alignment_area
        milling_task_config.acquisition.imaging.path = self.lamella.path

        # ensure the first stage is a coincidence milling strategy
        if not isinstance(
            milling_task_config.stages[0].strategy, CoincidenceMillingStrategy
        ):
            milling_task_config.stages[0].strategy = CoincidenceMillingStrategy(
                config=CoincidenceMillingStrategyConfig()
            )

        # acquire fib image (to show in gui)
        self._acquire_channels(
            image_settings,
            field_of_view=milling_task_config.field_of_view,
            acquire_sem=False,
            acquire_fib=True,
        )

        # apply channel settings to GUI
        fm_config = self.lamella.task_config.get("Acquire Fluorescence Image", None)
        if fm_config is None or not isinstance(
            fm_config, AcquireFluorescenceImageConfig
        ):
            raise ValueError(
                "No previous AcquireFluorescenceImageTask found in lamella task config. Cannot run MillCoincidentTask."
            )

        channel_settings = deepcopy(fm_config.channel_settings)
        self.set_fluorescence_channels_ui(channel_settings)

        # find the selected channel settings
        selected_channel_settings: Optional[ChannelSettings] = None
        for cs in channel_settings:
            if cs.name == self.config.channel_name:
                selected_channel_settings = cs  # only use the selected channel
                break

        if selected_channel_settings is None:
            raise ValueError(
                f"Channel {self.config.channel_name} not found in channel settings."
            )

        # apply channel settings and acquire image (updates GUI)
        self.microscope.fm.set_channel(selected_channel_settings)
        self.microscope.fm.acquire_image()

        # apply the milling config to the milling widget
        self._set_milling_config_ui(milling_task_config)

        # QUERY: can we just use milling ui?
        # wait for user to complete coincidence milling
        ask_user(
            self.parent_ui,
            msg=f"Run coincidence milling for {self.lamella.name}. Press Continue when done.",
            pos="Continue",
            coincidence_milling=True,
        )

        # clear the coincidence milling config from the milling widget
        milling_task_config = self.clear_coincidence_ui()
        if milling_task_config is not None:
            self.config.milling[MILL_COINCIDENT_KEY] = deepcopy(milling_task_config)

        # acquire fluorescence images
        if self.config.acquire_fluorescence_images:
            self.log_status_message(
                "ACQUIRE_FLUORESCENCE_IMAGE", "Acquiring Fluorescence Image..."
            )
            # Generate timestamp-based filename
            timestamp = utils.current_timestamp_v3(timeonly=True)
            basename = f"{self.lamella.name}-coincidence-final-{timestamp}.ome.tiff"
            filename = os.path.join(self.lamella.path, basename)

            self.microscope.fm.acquisition_progress_signal.emit(
                {"state": "acquiring", "task": f"{self.task_name}"}
            )
            image = acquire_image(
                microscope=self.microscope.fm,
                channel_settings=fm_config.channel_settings,
                zparams=fm_config.zparams,
                stop_event=self._stop_event,
                filename=filename,
            )

        # acquire reference images
        self._acquire_set_of_reference_images(image_settings)

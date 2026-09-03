"""Mill Coincident: a queued coincidence mill, driven by the per-site setup.

Runs the coincidence milling strategy itself, at the milling pose with the
objective inserted, using the record ``SetupCoincidenceMillingTask`` left on the
lamella: the objective height, the FM region to monitor, where the pattern sits,
which channel, and the drop fraction. Unsupervised, the strategy stops the beam
when the monitored intensity drops by that fraction; supervised, the drop only
alerts and the operator stops it (the viewer's monitor mode, FIB-912).

Before this rewrite the task was unregistered, called a method that did not
exist, and even fixed would only have pushed the milling config to the widget
and waited for the operator to run the viewer by hand. See FIB-907.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

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
from fibsem.applications.autolamella.workflows.tasks.setup_coincidence_milling import (
    COINCIDENCE_SETUP_REFERENCE_FILENAME,
    SetupCoincidenceMillingTaskConfig,
)
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.base import FibsemMillingSettings
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.tasks import FibsemMillingStage, FibsemMillingTaskConfig
from fibsem.structures import CrossSectionPattern, field_meta

MILL_COINCIDENT_KEY = "mill_coincident"
DEFAULT_SETUP_TASK_NAME = "Setup Coincidence Milling"

DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY] = FibsemMillingTaskConfig(
    name="Coincident Milling",
    field_of_view=80e-6,
    stages=[
        FibsemMillingStage(
            name="Coincident Milling 01",
            milling=FibsemMillingSettings(
                milling_current=60e-12, application_file="Si-ccs"
            ),
            pattern=RectanglePattern(
                width=9.0e-6,
                depth=4.0e-7,
                height=20e-6,
                cross_section=CrossSectionPattern.CleaningCrossSection,
            ),
            strategy=CoincidenceMillingStrategy(),
        )
    ],
)


@dataclass
class MillCoincidentTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the MillCoincidentTask."""

    setup_task: str = field(
        default=DEFAULT_SETUP_TASK_NAME,
        metadata=field_meta(
            label="Setup Task",
            tooltip="The Setup Coincidence Milling task whose per-site record "
            "(objective height, FM region, pattern position, channel, drop "
            "fraction) this mill runs from",
        ),
    )
    acquire_fluorescence_images: bool = field(
        default=True,
        metadata=field_meta(
            label="Acquire Fluorescence Images",
            tooltip="Acquire a fluorescence image after coincident milling, with the "
            "lamella's fluorescence task settings",
        ),
    )
    task_type: ClassVar[str] = "MILL_COINCIDENT"
    display_name: ClassVar[str] = "Coincident Milling"

    @property
    def opens_with_reference_alignment(self) -> bool:
        return True

    def __post_init__(self):
        if self.milling == {}:
            self.milling = deepcopy(
                {MILL_COINCIDENT_KEY: DEFAULT_MILLING_CONFIG[MILL_COINCIDENT_KEY]}
            )


class MillCoincidentTask(AutoLamellaTask):
    """Mill the coincident trench for a lamella, stopping on the intensity drop."""

    config: MillCoincidentTaskConfig
    config_cls: ClassVar[Type[MillCoincidentTaskConfig]] = MillCoincidentTaskConfig

    def _run(self) -> None:
        if self.microscope.fm is None:
            raise ValueError(
                "Microscope does not have a fluorescence microscope attached. "
                "Cannot run MillCoincidentTask."
            )
        # fail before moving anything: the record is what makes the mill possible
        setup = self._setup_config()
        if (
            self.lamella.milling_pose is None
            or self.lamella.milling_pose.stage_position is None
        ):
            raise ValueError(
                f"Milling pose for {self.lamella.name} is not set. Select the milling "
                "position before milling the lamella."
            )

        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # 1. the milling pose, stage only; tilt lands before the objective goes in
        self._move_to_milling_pose()

        try:
            # 2. objective in and at the height the setup left it
            self._move_objective(setup.objective_position)

            self._check_for_abort()

            # 3. realign to the frame the boxes were drawn in, under the same
            # conditions (objective in, same field of view). The generic reference
            # is a different field of view, taken before rough milling.
            self._align_to_setup_reference()

            # reference images
            self._acquire_reference_image(
                image_settings, field_of_view=setup.field_of_view
            )

            # 4. the per-site setup onto the milling config
            milling_task_config = self._apply_setup(setup)

            # 5. the mill. Headless it runs right here with our abort token; with a
            # UI the milling widget runs it (Run Milling gate when supervised).
            self.log_status_message("MILL_COINCIDENT", "Milling Coincident Lamella...")
            msg = (
                f"Press Run Milling to coincidence mill {self.lamella.name}. "
                "Press Continue when done."
            )
            milling_task_config = self.update_milling_config_ui(
                milling_task_config, msg=msg
            )
            self.config.milling[MILL_COINCIDENT_KEY] = deepcopy(milling_task_config)
            self._record_end_reason(milling_task_config)

            # 6. what the mill left: a fluorescence image and the FIB reference set
            if self.config.acquire_fluorescence_images:
                self._acquire_final_fluorescence_image()
            self._acquire_set_of_reference_images(image_settings)
        finally:
            # on every exit: an inserted objective blocks the next stage move
            self._retract_objective()

    # ------------------------------------------------------------------

    def _setup_config(self) -> SetupCoincidenceMillingTaskConfig:
        """The per-site record, by the setup task's name."""
        setup = self.lamella.task_config.get(self.config.setup_task)
        if not isinstance(setup, SetupCoincidenceMillingTaskConfig):
            raise ValueError(
                f"{self.lamella.name} has no '{self.config.setup_task}' task. Run "
                "Setup Coincidence Milling for this lamella first."
            )
        if not setup.is_set_up:
            raise ValueError(
                f"'{self.config.setup_task}' has not been run for {self.lamella.name}: "
                "no objective position is recorded. Run it first."
            )
        return setup

    def _move_objective(self, position: float) -> None:
        objective = self.microscope.fm.objective
        self.log_status_message("MOVE_OBJECTIVE", "Inserting Objective...")
        if objective.state != "Inserted":
            objective.insert()
        self.log_status_message("MOVE_OBJECTIVE", "Moving Objective to position...")
        objective.move_absolute(position)

    def _align_to_setup_reference(self) -> None:
        if os.path.exists(
            os.path.join(self.lamella.path, COINCIDENCE_SETUP_REFERENCE_FILENAME)
        ):
            self._align_reference_image(COINCIDENCE_SETUP_REFERENCE_FILENAME)
            return
        logging.warning(
            f"{self.task_name}: {self.lamella.name} has no coincidence setup reference "
            f"({COINCIDENCE_SETUP_REFERENCE_FILENAME}); aligning to the generic "
            f"reference {ALIGNMENT_REFERENCE_IMAGE_FILENAME} instead."
        )
        self._align_reference_image(ALIGNMENT_REFERENCE_IMAGE_FILENAME)

    def _apply_setup(
        self, setup: SetupCoincidenceMillingTaskConfig
    ) -> FibsemMillingTaskConfig:
        """Put the per-site record onto the milling config.

        The pattern offset goes onto every enabled stage, so a multi-stage mill
        (top-to-bottom, then bottom-to-top) keeps one position. The FM region and
        the drop fraction seed every coincidence strategy: the site's value is the
        one the operator set, and it applies to all stages alike.
        """
        milling_task_config = self.config.milling[MILL_COINCIDENT_KEY]
        milling_task_config.field_of_view = setup.field_of_view
        milling_task_config.alignment.rect = self.lamella.alignment_area
        milling_task_config.acquisition.imaging.path = self.lamella.path

        supervised = self.validate
        for stage in milling_task_config.enabled_stages:
            stage.pattern.point = deepcopy(setup.pattern_offset)
            if not isinstance(stage.strategy, CoincidenceMillingStrategy):
                stage.strategy = CoincidenceMillingStrategy(
                    config=CoincidenceMillingStrategyConfig()
                )
            strategy_config = stage.strategy.config
            strategy_config.bbox = deepcopy(setup.fm_roi)
            strategy_config.intensity_drop_fraction = setup.intensity_drop_fraction
            # unsupervised: the drop stops the beam; supervised: it alerts, the
            # operator stops (or flips the mode in the viewer)
            strategy_config.supervised = supervised

        # the channel the strategy monitors is whatever the FM is set to when the
        # mill starts, so set it now
        channel = self._find_channel(setup.channel_name)
        if channel is not None:
            self.set_fluorescence_channels_ui([channel])
            self.microscope.fm.set_channel(channel)
        return milling_task_config

    def _record_end_reason(self, milling_task_config: FibsemMillingTaskConfig) -> None:
        """Say why each coincidence stage ended, so a batch reads without the folders."""
        reasons = []
        for stage in milling_task_config.enabled_stages:
            strategy = stage.strategy
            if isinstance(strategy, CoincidenceMillingStrategy):
                reasons.append(f"{stage.name}: {strategy.end_reason or 'unknown'}")
        if reasons:
            self.log_status_message(
                "MILL_COINCIDENT_END",
                "Coincidence milling ended: " + "; ".join(reasons),
            )

    def _fluorescence_config(self) -> Optional[AcquireFluorescenceImageConfig]:
        for task_config in self.lamella.task_config.values():
            if isinstance(task_config, AcquireFluorescenceImageConfig):
                return task_config
        return None

    def _find_channel(self, name: str) -> Optional[ChannelSettings]:
        fm_config = self._fluorescence_config()
        if fm_config is None:
            logging.warning(
                f"{self.task_name}: {self.lamella.name} has no fluorescence task; "
                f"cannot look up channel '{name}'. Milling on the current channel."
            )
            return None
        for channel in fm_config.channel_settings:
            if channel.name == name:
                return deepcopy(channel)
        logging.warning(
            f"{self.task_name}: channel '{name}' is not in the fluorescence task for "
            f"{self.lamella.name}. Milling on the current channel."
        )
        return None

    def _acquire_final_fluorescence_image(self) -> None:
        fm_config = self._fluorescence_config()
        if fm_config is None:
            logging.warning(
                f"{self.task_name}: no fluorescence task for {self.lamella.name}; "
                "skipping the final fluorescence image."
            )
            return
        self.log_status_message(
            "ACQUIRE_FLUORESCENCE_IMAGE", "Acquiring Fluorescence Image..."
        )
        timestamp = utils.current_timestamp_v3(timeonly=True)
        basename = f"{self.lamella.name}-coincidence-final-{timestamp}.ome.tiff"
        filename = os.path.join(self.lamella.path, basename)
        image = acquire_image(
            microscope=self.microscope.fm,
            channel_settings=fm_config.channel_settings,
            zparams=fm_config.zparams,
            stop_event=self._stop_event,
            filename=filename,
        )
        # acquire_image swallows save failures; an unwritten image has no filepath
        # and is skipped. See AcquireFluorescenceImageTask for the same pattern.
        self._record_output("fluorescence", image)

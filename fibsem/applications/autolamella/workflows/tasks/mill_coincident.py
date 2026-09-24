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
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

import fibsem.utils as utils
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows._default_milling_config import (
    DEFAULT_MILLING_CONFIG,
)
from fibsem.applications.autolamella.workflows.interaction import (
    ReleaseCoincidenceMilling,
    RunCoincidenceMilling,
    WatchCoincidenceMilling,
    ask,
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
from fibsem.applications.autolamella.workflows.ui import _abort_requested
from fibsem.fm.acquisition import acquire_image
from fibsem.fm.structures import ChannelSettings
from fibsem.milling.base import FibsemMillingSettings
from fibsem.milling.patterning.patterns2 import RectanglePattern
from fibsem.milling.strategy.coincidence import (
    CoincidenceMillingStrategy,
    CoincidenceMillingStrategyConfig,
)
from fibsem.milling.tasks import (
    FibsemMillingStage,
    FibsemMillingTaskConfig,
    run_milling_task,
)
from fibsem.structures import CrossSectionPattern, field_meta

MILL_COINCIDENT_KEY = "mill_coincident"
DEFAULT_SETUP_TASK_NAME = "Setup Coincidence Milling"
# how long an instruction to the viewer may hold the mill; it never fails it
WATCH_TIMEOUT_S = 30.0


class _StopEither:
    """A stop for the headless mill that either the workflow's abort or the
    viewer's Stop can set. ``FibsemMillingTask`` and the strategy only ever ask
    ``is_set``, so this stands in for the ``threading.Event`` they expect.
    """

    def __init__(self, abort: Optional[threading.Event]) -> None:
        self._abort = abort
        self._own = threading.Event()

    def set(self) -> None:
        self._own.set()

    def is_set(self) -> bool:
        return self._own.is_set() or (self._abort is not None and self._abort.is_set())


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
    # The channel the strategy monitors while milling. Its own settings, not a
    # name looked up in the fluorescence task: monitoring wants a short exposure
    # at low power for many minutes of continuous frames, where a z-stack wants a
    # long exposure at full power once. Protocol-level, shared by every site; the
    # setup step tunes it against the live frame.
    monitoring_channel: ChannelSettings = field(
        default_factory=lambda: ChannelSettings(
            name="Monitoring",
            excitation_wavelength=550,
            emission_wavelength="Fluorescence",
            power=0.1,
            exposure_time=0.1,
        ),
        metadata=field_meta(hidden=True),  # its own controls, not a form row
    )
    task_type: ClassVar[str] = "MILL_COINCIDENT"
    display_name: ClassVar[str] = "Coincident Milling"

    @property
    def parameters(self) -> tuple[str, ...]:
        # serialised under its own key below, as the fluorescence task does
        return tuple(p for p in super().parameters if p != "monitoring_channel")

    def to_dict(self) -> dict:
        ddict = super().to_dict()
        ddict["monitoring_channel"] = self.monitoring_channel.to_dict()
        return ddict

    @classmethod
    def from_dict(cls, ddict: dict) -> "MillCoincidentTaskConfig":
        # the base loader reads milling + reference imaging; it warns about any
        # parameter it does not know, and it knows none of ours, so hand it none
        cfg = AutoLamellaTaskConfig.from_dict({**ddict, "parameters": {}})
        params = ddict.get("parameters", {}) or {}
        channel = ddict.get("monitoring_channel")
        kwargs = dict(
            task_name=cfg.task_name,
            milling=cfg.milling,
            reference_imaging=cfg.reference_imaging,
            setup_task=str(params.get("setup_task", DEFAULT_SETUP_TASK_NAME)),
            acquire_fluorescence_images=bool(
                params.get("acquire_fluorescence_images", True)
            ),
        )
        if channel is not None:
            kwargs["monitoring_channel"] = ChannelSettings.from_dict(channel)
        return cls(**kwargs)

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

    sessions = ("coincidence milling",)

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

            # 5. the mill. Supervised, the coincidence viewer runs it: the operator
            # checks the boxes, starts, watches, stops, continues. Otherwise it
            # runs right here with the abort token, and the viewer only watches.
            self.log_status_message("MILL_COINCIDENT", "Milling Coincident Lamella...")
            if self.parent_ui is not None and self.validate:
                milling_task_config = self._mill_supervised(milling_task_config)
            else:
                milling_task_config = self._mill_automated(milling_task_config)
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

    def _mill_supervised(
        self, milling_task_config: FibsemMillingTaskConfig
    ) -> FibsemMillingTaskConfig:
        """The viewer's run: one question, answered with the config as run."""
        result = ask(
            self.parent_ui.ui_responder,
            RunCoincidenceMilling(
                lamella=self.lamella,
                milling_config=deepcopy(milling_task_config),
                fib_image=self._last_fib_image,
                monitoring_channel=self.config.monitoring_channel,
                message=(
                    f"Check the boxes for {self.lamella.name}, Start Milling, "
                    "then Continue."
                ),
            ),
            abort=lambda: _abort_requested(self.parent_ui),
        )
        # The operator pressed Continue in the viewer, milled or not.
        self._decided_in_the_workflow()
        if result is None:
            self.log_status_message(
                "MILL_COINCIDENT_SKIPPED",
                f"Continued without coincidence milling {self.lamella.name}.",
            )
            return milling_task_config
        return result

    def _mill_automated(
        self, milling_task_config: FibsemMillingTaskConfig
    ) -> FibsemMillingTaskConfig:
        """Run the mill here; the viewer, if there is one, watches and can stop it."""
        stop = _StopEither(self._stop_event)
        self._tell(
            WatchCoincidenceMilling(
                milling_config=milling_task_config,
                stop=stop.set,
                title=f"{milling_task_config.name} · {self.lamella.name}",
            )
        )
        try:
            task = run_milling_task(
                self.microscope, milling_task_config, None, stop_event=stop
            )
        finally:
            self._tell(ReleaseCoincidenceMilling())
        return task.config

    def _tell(self, request) -> None:
        """An instruction to the viewer that must never hold or fail the mill."""
        if self.parent_ui is None:
            return
        try:
            ask(self.parent_ui.ui_responder, request, timeout=WATCH_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 - the viewer is not the mill
            logging.warning(
                f"{self.task_name}: the coincidence viewer did not take "
                f"{type(request).__name__}: {exc}"
            )

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
        channel = deepcopy(self.config.monitoring_channel)
        self.set_fluorescence_channels_ui([channel])
        self.microscope.fm.set_channel(channel)
        return milling_task_config

    def _record_end_reason(self, milling_task_config: FibsemMillingTaskConfig) -> None:
        """Say why each coincidence stage ended, so a batch reads without the folders.

        Logged now, and kept for the task history's status message: the record
        of a monitored mill should say "drop" or "completed", not "Finished".
        """
        reasons = []
        for stage in milling_task_config.enabled_stages:
            strategy = stage.strategy
            if isinstance(strategy, CoincidenceMillingStrategy):
                # no end reason: the strategy never ran -- a Stop on an earlier
                # stage cancels the rest, and a Continue without milling runs none
                reasons.append(f"{stage.name}: {strategy.end_reason or 'not run'}")
        self._end_reasons = reasons
        if reasons:
            self.log_status_message(
                "MILL_COINCIDENT_END",
                "Coincidence milling ended: " + "; ".join(reasons),
            )

    @property
    def finished_message(self) -> str:
        reasons = getattr(self, "_end_reasons", None)
        if not reasons:
            return "Finished"
        return "Finished · " + "; ".join(reasons)

    def _fluorescence_config(self) -> Optional[AcquireFluorescenceImageConfig]:
        for task_config in self.lamella.task_config.values():
            if isinstance(task_config, AcquireFluorescenceImageConfig):
                return task_config
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

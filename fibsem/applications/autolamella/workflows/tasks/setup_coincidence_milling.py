"""Setup Coincidence Milling: the per-site setup step for a queued coincidence mill.

Coincidence milling happens at the lamella's milling pose with the fluorescence
objective inserted, monitoring an FM region while a FIB pattern mills. Which
region, which pattern position, which objective height and which channel are all
per site, and until this task existed they lived only in the operator's hands for
the duration of one manual run in the coincidence viewer.

This task puts the microscope where that mill will happen, hands the operator the
viewer to place the boxes (FIB-911), and records what they leave behind on the
lamella. ``MillCoincidentTask`` reads that record back by task name. Headless --
no ``parent_ui`` -- it records the defaults and the objective position it used,
without waiting.

Always supervised: like Select Milling Position, the point of the task is the
hand-off, so the protocol's ``supervise`` flag does not gate it.

The objective position is a *task parameter*, not a pose. The coincidence pose is
the milling pose with the objective inserted, and storing an objective position on
``milling_pose`` would make every other mill task drive the objective, because
``set_microscope_state`` moves it whenever the field is set. See FIB-907.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Type

from fibsem import timing
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.structures import FibsemRectangle, Point, field_meta

SETUP_COINCIDENCE_MILLING_KEY = "SETUP_COINCIDENCE_MILLING"

# The FIB image the boxes are drawn against. The mill task beam-shift aligns to
# this file before applying them, so they land where they were placed. Written by
# ``_acquire_channels`` under this stem with the beam suffix appended.
COINCIDENCE_SETUP_REFERENCE_STEM = "ref_coincidence_setup"
COINCIDENCE_SETUP_REFERENCE_FILENAME = f"{COINCIDENCE_SETUP_REFERENCE_STEM}_ib.tif"


@dataclass
class SetupCoincidenceMillingTaskConfig(AutoLamellaTaskConfig):
    """Per-site coincidence milling setup.

    Everything here except ``channel_name``, ``field_of_view`` and
    ``align_coincidence`` is written by the task from what the operator left in the
    viewer. The stage position is deliberately absent: it is always the lamella's
    milling pose at run time.
    """

    task_type: ClassVar[str] = "SETUP_COINCIDENCE_MILLING"
    display_name: ClassVar[str] = "Setup Coincidence Milling"

    channel_name: str = field(
        default="Red Channel",
        metadata=field_meta(
            label="Channel",
            tooltip="The fluorescence channel whose intensity is monitored while milling",
        ),
    )
    field_of_view: float = field(
        default=80e-6,
        metadata=field_meta(
            label="Field of View",
            unit="m",
            scale=1e6,
            tooltip="FIB field of view for the setup image; the mill task reuses it",
        ),
    )
    intensity_drop_fraction: float = field(
        default=0.4,
        metadata=field_meta(
            label="Drop Fraction",
            minimum=0.05,
            maximum=0.95,
            tooltip="Stop when the rolling mean drops by this fraction below its peak "
            "(e.g. 0.4 = 40% drop). Seeds every coincidence stage of the mill.",
        ),
    )
    align_coincidence: bool = field(
        default=False,
        metadata=field_meta(
            label="Align Coincidence",
            tooltip="Run the SEM/FIB coincidence alignment at the milling pose before "
            "inserting the objective. Off until the bench data is in (FIB-872).",
        ),
    )
    # --- written by the task -------------------------------------------------
    # metres; None until the task or the operator has set it
    objective_position: Optional[float] = field(
        default=None,
        metadata=field_meta(
            label="Objective Position",
            unit="m",
            scale=1e6,
            tooltip="Objective height that focuses the sample at the milling tilt",
        ),
    )
    # fraction of the FM camera frame (left, top, width, height); None = whole frame
    fm_roi: Optional[FibsemRectangle] = field(
        default=None, metadata=field_meta(hidden=True)
    )
    # metres from the FIB image centre, applied to every enabled milling stage
    pattern_offset: Point = field(
        default_factory=lambda: Point(0.0, 0.0), metadata=field_meta(hidden=True)
    )

    @property
    def parameters(self) -> tuple[str, ...]:
        # the two non-scalar fields serialise under their own keys below
        return tuple(
            p for p in super().parameters if p not in ("fm_roi", "pattern_offset")
        )

    @property
    def is_set_up(self) -> bool:
        """Whether the task has recorded an objective position for this site."""
        return self.objective_position is not None

    @property
    def estimated_duration(self) -> float:
        # the objective insert and move, plus two acquisitions; the hand-off itself
        # is the operator's time and is not estimated
        total = super().estimated_duration
        total += timing.stage_move_cost(1)  # objective insert stands in for a move
        return total

    def to_dict(self) -> dict:
        ddict = super().to_dict()
        ddict["fm_roi"] = self.fm_roi.to_dict() if self.fm_roi is not None else None
        ddict["pattern_offset"] = self.pattern_offset.to_dict()
        return ddict

    @classmethod
    def from_dict(cls, ddict: dict) -> "SetupCoincidenceMillingTaskConfig":
        cfg = AutoLamellaTaskConfig.from_dict(ddict)
        params = ddict.get("parameters", {}) or {}
        objective_position = params.get("objective_position")
        fm_roi = ddict.get("fm_roi")
        pattern_offset = ddict.get("pattern_offset")
        return cls(
            task_name=cfg.task_name,
            milling=cfg.milling,
            reference_imaging=cfg.reference_imaging,
            channel_name=str(params.get("channel_name", "Red Channel")),
            field_of_view=float(params.get("field_of_view", 80e-6)),
            intensity_drop_fraction=float(params.get("intensity_drop_fraction", 0.4)),
            align_coincidence=bool(params.get("align_coincidence", False)),
            objective_position=(
                float(objective_position) if objective_position is not None else None
            ),
            fm_roi=FibsemRectangle.from_dict(fm_roi) if fm_roi is not None else None,
            pattern_offset=(
                Point.from_dict(pattern_offset)
                if pattern_offset is not None
                else Point(0.0, 0.0)
            ),
        )


class SetupCoincidenceMillingTask(AutoLamellaTask):
    """Put the microscope where the coincidence mill will run and record the setup."""

    config: SetupCoincidenceMillingTaskConfig
    config_cls: ClassVar[Type[SetupCoincidenceMillingTaskConfig]] = (
        SetupCoincidenceMillingTaskConfig
    )

    def _run(self) -> None:
        if self.microscope.fm is None:
            raise ValueError(
                "Fluorescence microscope not initialized in the FibsemMicroscope "
                "instance. Cannot set up coincidence milling."
            )
        if (
            self.lamella.milling_pose is None
            or self.lamella.milling_pose.stage_position is None
        ):
            raise ValueError(
                f"Milling pose for {self.lamella.name} is not set. Select the milling "
                "position before setting up coincidence milling."
            )

        image_settings = self.config.imaging
        image_settings.path = self.lamella.path

        # 1. the milling pose, stage only. Tilt first: once the objective is in, z and
        # t are unavailable, so the order here is not negotiable.
        self.log_status_message("MOVE_TO_POSITION", "Moving to Milling Position...")
        self.microscope.set_microscope_state(self.lamella.milling_pose)

        if self.config.align_coincidence:
            self._align_coincidence()

        self._check_for_abort()

        # 2. objective in, then to the stored height -- or the FM pose's as a first
        # guess, which is the only other place an objective height for this site lives.
        objective_position = self._initial_objective_position()
        try:
            self._move_objective(objective_position)

            self._check_for_abort()

            # 3. the FIB frame the boxes are drawn against, and one FM frame so the
            # viewer opens with something to draw on. Recorded under its own name so
            # the conventional reference sets are not polluted (see _acquire_channels).
            self._acquire_channels(
                image_settings,
                filename=COINCIDENCE_SETUP_REFERENCE_STEM,
                field_of_view=self.config.field_of_view,
                acquire_sem=False,
                acquire_fib=True,
            )
            self._acquire_fm_frame()

            # 4. hand-off. Without a UI there is nobody to hand to: record what we
            # used and move on. The viewer's setup mode (FIB-911) plugs in here.
            if self.parent_ui is not None:
                self._hand_off()

            # 5. the record. Whatever the operator left is now this site's setup.
            self.config.objective_position = self._current_objective_position(
                objective_position
            )
            self.log_status_message(
                "RECORD_SETUP",
                f"Recorded coincidence setup for {self.lamella.name}: "
                f"objective {self.config.objective_position * 1e6:.1f} µm, "
                f"channel {self.config.channel_name}.",
            )
        finally:
            # on every exit, including abort: an inserted objective blocks the next
            # stage move, and the end-of-queue retract is too late for that (FIB-376)
            self._retract_objective()

    # ------------------------------------------------------------------

    def _initial_objective_position(self) -> Optional[float]:
        if self.config.objective_position is not None:
            return self.config.objective_position
        fm_pose = self.lamella.fluorescence_pose
        if fm_pose is not None and fm_pose.objective_position is not None:
            logging.info(
                f"{self.task_name}: no coincidence objective position stored for "
                f"{self.lamella.name}; starting from the fluorescence pose's."
            )
            return fm_pose.objective_position
        return None

    def _move_objective(self, position: Optional[float]) -> None:
        objective = self.microscope.fm.objective
        self.log_status_message("MOVE_OBJECTIVE", "Inserting Objective...")
        if objective.state != "Inserted":
            objective.insert()
        if position is None:
            logging.warning(
                f"{self.task_name}: no objective position known for "
                f"{self.lamella.name}; leaving the objective where insertion put it."
            )
            return
        self.log_status_message("MOVE_OBJECTIVE", "Moving Objective to position...")
        objective.move_absolute(position)

    def _current_objective_position(self, fallback: Optional[float]) -> float:
        """Where the objective is now: what the operator left, or what we set."""
        objective = self.microscope.fm.objective
        position = objective.position if objective.state == "Inserted" else None
        if position is None:
            position = fallback
        if position is None:
            raise ValueError(
                f"No objective position could be determined for {self.lamella.name}. "
                "Set one in the viewer, or set the fluorescence pose's objective "
                "position first."
            )
        return float(position)

    def _acquire_fm_frame(self) -> None:
        """One frame on the configured channel, for the viewer to open on.

        The frame is not recorded: it is a preview, and the FM tasks own the
        experiment's fluorescence records.
        """
        fm = self.microscope.fm
        channel = self._find_channel()
        self.log_status_message(
            "ACQUIRE_FLUORESCENCE_IMAGE", "Acquiring Fluorescence Image..."
        )
        try:
            image = fm.acquire_image(channel)
            fm.acquisition_signal.emit(image)
        except Exception as e:
            # a missing frame costs the operator a click on Acquire, not the task
            logging.warning(
                f"{self.task_name}: could not acquire a fluorescence frame: {e}"
            )

    def _find_channel(self):
        """The configured channel, from the lamella's fluorescence task if it has one."""
        from fibsem.applications.autolamella.workflows.tasks.acquire_fluorescence import (
            AcquireFluorescenceImageConfig,
        )

        for task_config in self.lamella.task_config.values():
            if not isinstance(task_config, AcquireFluorescenceImageConfig):
                continue
            for channel in task_config.channel_settings:
                if channel.name == self.config.channel_name:
                    return channel
        logging.info(
            f"{self.task_name}: channel '{self.config.channel_name}' is not in any "
            "fluorescence task for this lamella; acquiring on the current channel."
        )
        return None

    def _align_coincidence(self) -> None:
        """Opt-in SEM/FIB coincidence alignment at the milling pose, keeping the FIB view."""
        from fibsem.alignment.coincidence import ensure_coincident
        from fibsem.structures import BeamType

        self.log_status_message("ALIGN_COINCIDENCE", "Aligning SEM/FIB coincidence...")
        result = ensure_coincident(self.microscope, reference=BeamType.ION)
        if not result.converged:
            logging.warning(
                f"{self.task_name}: coincidence alignment did not converge for "
                f"{self.lamella.name}: {result.reason}. Continuing at the milling pose."
            )

    def _hand_off(self) -> None:
        """Give the operator the viewer to place the boxes. Wired in FIB-911.

        Until then a supervised run is the same as a headless one: the defaults and
        the objective position are recorded, and the operator adjusts nothing.
        """
        logging.info(
            f"{self.task_name}: viewer hand-off not yet available; recording defaults "
            f"for {self.lamella.name}."
        )

    @property
    def setup_reference_path(self) -> str:
        return os.path.join(self.lamella.path, COINCIDENCE_SETUP_REFERENCE_FILENAME)

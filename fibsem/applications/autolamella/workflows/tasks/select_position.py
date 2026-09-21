######## SELECT MILLING POSITION TASK DEFINITIONS ########

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

import numpy as np

from fibsem import constants
from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_POSE,
    MILLING_POSE,
    derive_pose,
    move_pose,
)
from fibsem.applications.autolamella.proposals import (
    POINT_OF_INTEREST,
    Decision,
    DecisionOutcome,
    Proposal,
    human_author,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.ui import ask_user, select_poi_ui
from fibsem.structures import BeamType, ImageSettings, Point, field_meta

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import Lamella
    from fibsem.structures import FibsemImage


@dataclass
class SelectMillingPositionTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SelectMillingPositionTask."""

    milling_angle: float = field(
        default=15,
        metadata=field_meta(
            tooltip="The angle between the FIB and sample used for milling",
            unit=constants.DEGREE_SYMBOL,
        ),
    )
    auto_milling_alignment: bool = field(
        default=False,
        metadata=field_meta(
            label="Auto Coincidence Alignment",
            tooltip="Align SEM/FIB coincidence at the current pose, then tilt "
            "to the milling angle while keeping it (from the SEM orientation "
            "or the milling angle; other start poses are not validated)",
        ),
    )
    use_autofocus: bool = field(
        default=True,
        metadata=field_meta(
            label="Use Autofocus",
            tooltip="Whether to autofocus before moving to the milling position",
        ),
    )

    select_poi: bool = field(
        default=True,
        metadata=field_meta(
            label="Select Point of Interest",
            tooltip="Whether to ask the user to select a point of interest in the FIB image",
        ),
    )
    sync_fluorescence_pose: bool = field(
        default=False,
        metadata=field_meta(
            label="Sync Fluorescence Pose",
            tooltip="Move the lamella's fluorescence pose to follow the milling "
            "position recorded here, so a fluorescence task images the site as "
            "set up rather than where the lamella was first marked. Off (the "
            "default) leaves the fluorescence pose as it was.",
        ),
    )
    task_type: ClassVar[str] = "SELECT_MILLING_POSITION"
    display_name: ClassVar[str] = "Select Milling Position"


def consumed_values(lamella: "Lamella") -> List[str]:
    """The value names a point-of-interest proposal for this lamella may
    carry: a value exists because a later task consumes it. ``poi`` is
    consumed by any milling task whose patterns follow the point."""
    values = []
    for task_config in lamella.task_config.values():
        if getattr(task_config, "sync_to_poi", False) and task_config.milling:
            values.append("poi")
            break
    return values


class CurrentPoiProposer:
    """The v1 point-of-interest proposer: the point the lamella already has,
    as it stood when the task started. For a lamella nobody has touched that
    is the origin of the milling frame, the centre of the image; for one that
    correlation, a script or an earlier decision positioned, it is that point,
    so proposing does not undo it. No confidence, no alternatives. A real
    proposer -- a segmentation model, say -- is a swap for this class.

    Declines when nothing after Setup consumes a point, so no empty proposal
    is recorded. Proposes the point as it stood when the task started, before
    the operator was asked, so a supervised answer leaves a delta.
    """

    kind = POINT_OF_INTEREST
    name = "current-poi"
    version = 2

    def propose(self, task: "SelectMillingPositionTask") -> Optional[Proposal]:
        if not consumed_values(task.lamella):
            return None
        poi = task._prior_poi
        return Proposal(kind=self.kind, values={"poi": Point(poi.x, poi.y)})


class SelectMillingPositionTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""

    # the milling position: a value someone may change, so a kind of its own
    proposer = CurrentPoiProposer()

    config: SelectMillingPositionTaskConfig
    config_cls: ClassVar[Type[SelectMillingPositionTaskConfig]] = (
        SelectMillingPositionTaskConfig
    )

    def _run(self) -> None:
        """Run the task to select the milling position for the lamella for milling."""

        # bookkeeping
        self.image_settings: ImageSettings = self.config.imaging
        self.image_settings.path = self.lamella.path

        # move to lamella milling position
        self._move_to_milling_pose()

        self.log_status_message("SELECT_POSITION", "Selecting Position...")
        milling_angle = self.config.milling_angle
        is_close = self.microscope.is_close_to_milling_angle(
            milling_angle=milling_angle
        )

        # acquire an image at the milling position
        if self.config.use_autofocus:
            self._run_autofocus(beam_type=BeamType.ELECTRON)
            self._run_autofocus(beam_type=BeamType.ION)
        self._acquire_reference_image(
            image_settings=self.image_settings,
            filename=f"ref_{self.task_name}_start",
            field_of_view=self.config.reference_imaging.field_of_view1,
        )

        if self.config.auto_milling_alignment:
            self._align_coincident_for_milling(milling_angle, is_close)

        if not is_close:
            if self.config.auto_milling_alignment:
                pass  # tilted coincidently above
            elif self.validate:
                current_milling_angle = self.microscope.get_current_milling_angle()
                ret = ask_user(
                    parent_ui=self.parent_ui,
                    msg=f"Tilt to specified milling angle ({milling_angle:.1f}{constants.DEGREE_SYMBOL})? "
                    f"Current milling angle is {current_milling_angle:.1f}{constants.DEGREE_SYMBOL}.",
                    pos="Tilt",
                    neg="Skip",
                )
                if ret:
                    self.microscope.move_to_milling_angle(
                        milling_angle=np.radians(milling_angle)
                    )
            else:
                self.microscope.move_to_milling_angle(
                    milling_angle=np.radians(milling_angle)
                )

            if self.config.use_autofocus:
                self._run_autofocus(beam_type=BeamType.ION)

            # reacquire image at milling angle
            self._acquire_reference_image(
                image_settings=self.image_settings,
                filename=f"ref_{self.task_name}_post_tilt",
                field_of_view=self.config.reference_imaging.field_of_view1,
            )

        # confirm with user to move to milling position
        if self.validate:
            ask_user(
                parent_ui=self.parent_ui,
                msg=f"Double click the image to move to the milling position for {self.lamella.name}. "
                f"Press Continue when done.",
                pos="Continue",
            )

        # select point of interest -- under review it is proposed at the end of
        # the task instead, on the final reference image; otherwise the answer
        # given here (supervised) is the decision on that proposal
        # The point as it stands before anyone is asked: what the proposal
        # says, so the delta against the answer is what the operator moved.
        self._prior_poi = Point(self.lamella.poi.x, self.lamella.poi.y)
        if self.config.select_poi and not self.review:
            poi = select_poi_ui(
                parent_ui=self.parent_ui,
                # the FIB image the reference acquisition above displayed — the
                # marker's coordinates only mean something against it
                image=self._last_fib_image,
                msg=f"Move the marker to the point of interest for {self.lamella.name}. Press Continue when done.",
                validate=self.validate,
                initial_poi=self.lamella.poi,
            )
            if poi is not None:
                self.lamella.poi = poi
                synced = self.lamella.sync_tasks_to_poi()
                if synced:
                    logging.info(f"Synced tasks to POI: {synced}")
                # The answer is the decision on the proposal the base records
                # after the run; already applied, so it is recorded without a
                # second write-through, and the delta between it and the
                # proposer's point is what supervised runs used to throw away.
                experiment = getattr(self.task_manager, "experiment", None)
                self.inline_decision = Decision(
                    outcome=DecisionOutcome.Confirmed,
                    author=experiment.author()
                    if experiment is not None
                    else human_author(""),
                    values={"poi": poi},
                    via="workflow",
                )

        # validate alignment area
        self._validate_alignment_area()

        # acquire alignment reference image
        self._acquire_alignment_reference_image(
            image_settings=self.image_settings,
            reduced_area=self.lamella.alignment_area,
            field_of_view=self.config.reference_imaging.field_of_view1,
        )

        # reference images
        self._acquire_set_of_reference_images(self.image_settings)

        # store milling pose and angle
        # the task moved the lamella (the coincidence walk, the operator's own
        # centring), so this is a move and not only a record: a fluorescence pose
        # that is still a guess is worked out again from the new milling pose, as
        # it is wherever a person moves a lamella. Left behind, a fluorescence
        # stack taken from it lands tens of microns off the marks it is meant to
        # show (FIB-954). One somebody centred under the objective is theirs, and
        # stays -- unless the task is configured to overwrite it.
        move_pose(
            self.microscope,
            self.lamella,
            MILLING_POSE,
            state=self.microscope.get_microscope_state(),
        )
        if (
            self.config.sync_fluorescence_pose
            and self.lamella.fluorescence_pose is not None
        ):
            derive_pose(self.microscope, self.lamella, FLUORESCENCE_POSE)

    def _align_coincident_for_milling(
        self, milling_angle: float, is_close: bool
    ) -> None:
        """Make the SEM and FIB coincident at the milling angle.

        Initial scope: the task starts either AT the milling angle (align
        only) or at the SEM orientation (align there, then tilt to the
        milling angle keeping coincidence, and undo the surface walk the
        tilt produced so the site the operator chose is still centred).
        Aligning BEFORE the tilt is what makes the tilt's height-offset
        estimate - and so the walk undo - valid. Any other start pose is
        allowed but unvalidated, and says so in the log.

        A refusal never stops the task: the stage is left where the last
        reliable correction put it and the refusal is logged (policy for
        escalation - ask, spot burn - is deliberately not decided here).
        """
        import os

        from fibsem.alignment import ALIGNMENT_SUBDIR
        from fibsem.alignment.coincidence import ensure_coincident, tilt_coincident
        from fibsem.alignment.plotting import save_coincidence_diagnostics
        from fibsem.transformations import get_stage_tilt_from_milling_angle

        diagnostics_path = os.path.join(self.lamella.path, ALIGNMENT_SUBDIR)

        def on_progress(progress) -> None:
            self.update_status_ui(progress.describe())

        self.log_status_message("ALIGN_COINCIDENCE", "Aligning SEM/FIB coincidence...")
        start = ensure_coincident(
            self.microscope, reference=BeamType.ION, on_progress=on_progress
        )
        save_coincidence_diagnostics(start, diagnostics_path, prefix="start_")
        if not start.converged:
            logging.warning(
                "SEM/FIB coincidence not reached before the tilt (%s); the tilt's "
                "height-offset estimate will be off",
                start.reason,
            )
        if is_close:
            return

        orientation = self.microscope.get_stage_orientation()
        if orientation != "SEM":
            logging.warning(
                "Coincident tilt to the milling angle starting from the %s "
                "orientation is not validated (expected SEM or MILLING)",
                orientation,
            )
        target_stage_tilt = get_stage_tilt_from_milling_angle(
            self.microscope, np.radians(milling_angle)
        )
        self.log_status_message(
            "TILT_COINCIDENT",
            f"Tilting to the milling angle ({milling_angle:.1f}"
            f"{constants.DEGREE_SYMBOL}) keeping coincidence...",
        )
        tilt = tilt_coincident(
            self.microscope,
            target_stage_tilt,
            reference=BeamType.ION,
            on_progress=on_progress,
        )
        for i, alignment in enumerate(tilt.alignments, start=1):
            save_coincidence_diagnostics(
                alignment, diagnostics_path, prefix=f"tilt{i:02d}_"
            )
        if tilt.converged:
            logging.info(
                {
                    "msg": "milling_tilt_coincident",
                    "tilt_axis_offset": tilt.tilt_axis_offset,
                    "walk": tilt.walk,
                    "walk_undone": tilt.walk_undone,
                    "moves_applied": tilt.moves_applied,
                }
            )
        else:
            logging.warning(
                "Coincidence not restored at the milling angle (%s); continuing "
                "at the target tilt",
                tilt.reason,
            )

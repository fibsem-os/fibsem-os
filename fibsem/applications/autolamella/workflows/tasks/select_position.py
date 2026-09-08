######## SELECT MILLING POSITION TASK DEFINITIONS ########

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Type

import numpy as np

from fibsem import constants
from fibsem.applications.autolamella.proposals import (
    MILLING_SETUP,
    Proposal,
    supersede,
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
    task_type: ClassVar[str] = "SELECT_MILLING_POSITION"
    display_name: ClassVar[str] = "Select Milling Position"


class SelectMillingPositionTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""

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
        # the task instead, on the final reference image
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
        self.lamella.milling_pose = self.microscope.get_microscope_state()
        self.lamella.update_milling_angle(self.microscope)

        # propose the point of interest for review, on the final reference
        # image -- the last thing acquired, at the stored pose, and the one the
        # Review tab shows
        if self.config.select_poi and self.review:
            self._propose_poi()

    def _propose_poi(self) -> None:
        """Leave the point of interest as a proposal instead of asking for it.

        The task still completes -- everything after this step is independent
        of the point (only the rough and polishing patterns follow it, and they
        follow it when a decision writes it through). ``lamella.poi`` is not
        written here and the patterns are not synced: both happen in
        Experiment.decide on confirm, so the lamella is never in a state
        nobody sanctioned and the proposed point survives beside the confirmed
        one for the delta.

        Re-running the task is a deliberate act: a proposal that already has
        decisions is superseded, not kept and not overwritten. It moves, with
        its decisions and delta, onto the new proposal's record, and the new
        one is pending on the new image. The old value is not carried over as
        the default. (A stalled run resumes without re-running completed
        tasks, so that case never reaches here.)
        """
        existing = self.lamella.proposals.get(self.task_name)
        # The first of the final set is the tightest field of view; the
        # values are in the milling frame, so any image at the stored pose
        # would do, but the renderer shows this one.
        image_name = f"ref_{self.task_name}_final_res_01_ib.tif"
        if not os.path.exists(os.path.join(str(self.lamella.path), image_name)):
            settings = getattr(
                getattr(self._last_fib_image, "metadata", None), "image_settings", None
            )
            image_name = f"{settings.filename}_ib.tif" if settings else ""
        proposal = propose_milling_setup(self.lamella, image_name)
        if proposal is None:
            logging.info(
                f"{self.lamella.name}: nothing after {self.task_name} consumes a "
                "point of interest; no proposal to make."
            )
            return
        self.log_status_message("PROPOSE_POI", "Proposing Point of Interest...")
        if existing is not None and not existing.pending:
            logging.info(
                f"{self.lamella.name}: {self.task_name} re-run; the decided "
                "proposal is superseded and a new one is pending."
            )
        self.lamella.proposals[self.task_name] = supersede(
            existing if existing is not None and not existing.pending else None,
            proposal,
        )
        logging.info(
            {
                "msg": "proposal_recorded",
                "lamella": self.lamella.name,
                "task_name": self.task_name,
                "kind": proposal.kind,
                "values": {k: v.to_dict() for k, v in proposal.values.items()},
                "provenance": proposal.provenance,
            }
        )

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


def consumed_values(lamella: "Lamella") -> List[str]:
    """The value names a milling-setup proposal for this lamella may carry: a
    value exists because a later task consumes it. ``poi`` is consumed by any
    milling task whose patterns follow the point; a fiducial value would be
    consumed by the fiducial task, but has no writer yet, so it is not
    proposed."""
    values = []
    for task_config in lamella.task_config.values():
        if getattr(task_config, "sync_to_poi", False) and task_config.milling:
            values.append("poi")
            break
    return values


def propose_milling_setup(
    lamella: "Lamella", reference_image: str = ""
) -> Optional[Proposal]:
    """The v1 proposer: the centre of the image, which is today's default
    position (a lamella's point of interest starts at the origin of the milling
    frame). It exists to get the machinery running, not to be right -- no
    confidence, no alternatives. A real proposer is a swap for this function
    with the same return type.

    None when nothing consumes a point, so no empty proposals are recorded.
    """
    values = consumed_values(lamella)
    if not values:
        return None
    provenance: Dict[str, Any] = {
        "proposer": "centre-of-image",
        "version": 1,
        "values": values,
    }
    if reference_image:
        # A file name relative to the lamella's folder (<name>_ib.tif); readers
        # join it onto lamella.path, which also survives a moved experiment.
        provenance["reference_image"] = reference_image
    return Proposal(
        kind=MILLING_SETUP,
        values={"poi": Point(0.0, 0.0)},
        confidence=None,
        provenance=provenance,
    )

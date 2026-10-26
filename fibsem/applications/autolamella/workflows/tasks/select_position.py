######## SELECT MILLING POSITION TASK DEFINITIONS ########

import logging
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Tuple, Type

import numpy as np

from fibsem import constants
from fibsem.applications.autolamella.poses import sync_fluorescence_pose
from fibsem.applications.autolamella.structures import AutoLamellaTaskConfig
from fibsem.applications.autolamella.workflows.tasks.base import AutoLamellaTask
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    sync_fiducial_to_alignment_area,
)
from fibsem.structures.params import Display, param
from fibsem.structures.settings import (
    BeamType,
    FibsemStagePosition,
    ImageSettings,
    Point,
)
from fibsem.workflows.proposals import (
    ALIGNMENT_AREA,
    DETECTION,
    POINT_OF_INTEREST,
    STATE,
    DecisionOutcome,
)

# How far the confirmed position may be from the one imaged before the view
# is imaged again, in metres.
MOVED_TOL = 0.5e-6


@dataclass
class SelectMillingPositionTaskConfig(AutoLamellaTaskConfig):
    """Configuration for the SelectMillingPositionTask."""

    milling_angle: float = param(
        15,
        unit=constants.DEGREE_SYMBOL,
        doc="The angle between the FIB and sample used for milling",
    )
    auto_milling_alignment: bool = param(
        False,
        doc="Align SEM/FIB coincidence at the current pose, then tilt "
        "to the milling angle while keeping it (from the SEM orientation "
        "or the milling angle; other start poses are not validated)",
        display=Display("Auto Coincidence Alignment"),
    )
    use_autofocus: bool = param(
        True,
        doc="Whether to autofocus before moving to the milling position",
        display=Display("Use Autofocus"),
    )

    select_poi: bool = param(
        True,
        doc="Ask you to place the point of interest on the FIB image "
        "after the position is confirmed. Off, the point is used as it "
        "stands.",
        display=Display("Select Point of Interest"),
    )
    confirm_position: bool = param(
        True,
        doc="Ask you to confirm the milling position before the point "
        "of interest is picked. Off, the position is used as arrived at, "
        "so it is only as safe as the automatic positioning "
        "(Auto Coincidence Alignment) that put the stage there.",
        display=Display("Confirm Position"),
    )
    confirm_tilt: bool = param(
        True,
        doc="Ask you before tilting to the milling angle when the "
        "stage is not already there. Off, the task tilts on its own.",
        display=Display("Confirm Tilt"),
    )
    confirm_alignment_area: bool = param(
        True,
        doc="Ask you to check the alignment area on the last FIB image "
        "before the alignment reference is taken in it. Off, the area is "
        "used as it stands.",
        display=Display("Confirm Alignment Area"),
    )
    sync_fluorescence_pose: bool = param(
        False,
        doc="Move the lamella's fluorescence pose to follow the milling "
        "position recorded here, so a fluorescence task images the site as "
        "set up rather than where the lamella was first marked. Off (the "
        "default) leaves the fluorescence pose as it was.",
        display=Display("Sync Fluorescence Pose"),
    )
    sync_fiducial_position: bool = param(
        True,
        doc="Start the fiducial at the centre of the alignment area "
        "chosen here, so it clears the milling site. When the fiducial "
        "does not fit in the area, it keeps its configured position. You "
        "can still move it when the fiducial is milled.",
        display=Display("Sync Fiducial Position"),
    )
    task_type: ClassVar[str] = "SELECT_MILLING_POSITION"
    display_name: ClassVar[str] = "Select Milling Position"


class SelectMillingPositionTask(AutoLamellaTask):
    """Task to setup the lamella for milling."""

    # What needs a person while it runs, when supervised. The position is
    # confirmed before the point is picked, and the tilt to the milling angle
    # before it is made (both ``state``, each with a switch of its own); the
    # coincidence walk asks a detection when it is on; the point of interest
    # is placed once the position is confirmed; the alignment area is checked
    # before the reference is taken in it.
    asks = (STATE, DETECTION, POINT_OF_INTEREST, ALIGNMENT_AREA)

    @classmethod
    def asks_for(cls, config) -> Tuple[str, ...]:
        asked = {
            STATE: getattr(config, "confirm_position", True)
            or getattr(config, "confirm_tilt", True),
            DETECTION: getattr(config, "auto_milling_alignment", False),
            POINT_OF_INTEREST: getattr(config, "select_poi", True),
            ALIGNMENT_AREA: getattr(config, "confirm_alignment_area", True),
        }
        return tuple(k for k in cls.asks if asked.get(k, True))

    config: SelectMillingPositionTaskConfig
    config_cls: ClassVar[Type[SelectMillingPositionTaskConfig]] = (
        SelectMillingPositionTaskConfig
    )

    def _run(self) -> None:
        """Run the task to select the milling position for the lamella for milling."""

        # bookkeeping
        self.image_settings: ImageSettings = self.config.imaging

        # move to lamella milling position
        self._move_to_pose()

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
            else:
                # On the record: the position before the tilt is the proposal,
                # Tilt confirms it and the tilt follows. Skip rejects it: the
                # task carries on at the angle it is at.
                current_milling_angle = self.microscope.get_current_milling_angle()
                decision = self.ask(
                    STATE,
                    {"stage_position": self.microscope.get_stage_position()},
                    image=self._last_fib_image_file(),
                    message=f"Tilt to the milling angle ({milling_angle:.1f}"
                    f"{constants.DEGREE_SYMBOL})? The current milling angle is "
                    f"{current_milling_angle:.1f}{constants.DEGREE_SYMBOL}.",
                    provenance={
                        "positive": "Tilt",
                        "negative": "Skip",
                        "label": "Tilt",
                    },
                    decided=self._stage_position_now,
                    enabled=self.config.confirm_tilt,
                )
                if decision.outcome is not DecisionOutcome.Rejected:
                    self.microscope.move_to_milling_angle(milling_angle=milling_angle)

            if self.config.use_autofocus:
                self._run_autofocus(beam_type=BeamType.ION)

            # reacquire image at milling angle
            self._acquire_reference_image(
                image_settings=self.image_settings,
                filename=f"ref_{self.task_name}_post_tilt",
                field_of_view=self.config.reference_imaging.field_of_view1,
            )

        # confirm with user to move to milling position
        msg = (
            f"Double click the image to move to the milling position for "
            f"{self.lamella.name}. Press Continue when done."
        )
        # The position as arrived at is the proposal; the position as it stands
        # when Continue is pressed is the decision, so the move the operator
        # made is the delta.
        asked_at = self.microscope.get_stage_position()
        decision = self.ask(
            STATE,
            {"stage_position": asked_at},
            image=self._last_fib_image_file(),
            message=msg,
            decided=self._stage_position_now,
            enabled=self.config.confirm_position,
        )
        confirmed_at = decision.values.get("stage_position")
        if isinstance(confirmed_at, FibsemStagePosition) and not confirmed_at.is_close2(
            asked_at, tol=MOVED_TOL, axes=["x", "y", "z"]
        ):
            # Moved since the image was taken: the point of interest, the
            # alignment area and the fiducial are placed on this view, not that
            # one.
            self._acquire_reference_image(
                image_settings=self.image_settings,
                filename=f"ref_{self.task_name}_position",
                field_of_view=self.config.reference_imaging.field_of_view1,
            )

        self._select_poi()

        # validate alignment area
        self._validate_alignment_area(enabled=self.config.confirm_alignment_area)
        self.sync_fiducial_milling_task_position()

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
        # the task moved the lamella (the coincidence walk, the operator's own
        # centring), so the fluorescence pose derived when it was marked now
        # describes where it was. Every UI path that moves a lamella
        # syncs it; a fluorescence stack taken from the stale pose lands tens
        # of microns off the marks it is meant to show. Opt-in: the
        # default leaves the pose alone, as the task always has.
        if self.config.sync_fluorescence_pose:
            sync_fluorescence_pose(self.microscope, self.lamella, force=True)

    def _select_poi(self) -> None:
        """Ask for the point of interest on the FIB image the reference
        acquisition above displayed: a marker on the FIB view, where the
        lamella's point is now, dragged and then Continue. The point as it
        stands is the proposal, so the delta against the decision is what the
        operator moved. The decision writes the point through (the ``poi``
        writer), and every pattern that follows it moves with it."""
        poi = self.lamella.poi
        self.ask(
            POINT_OF_INTEREST,
            {"poi": Point(poi.x, poi.y)},
            image=self._last_fib_image_file(),
            message=f"Move the marker to the point of interest for "
            f"{self.lamella.name}. Press Continue when done.",
            enabled=self.config.select_poi,
        )

    def sync_fiducial_milling_task_position(self) -> None:
        """Move the fiducial to the centre of the alignment area, on the FIB
        image the area was chosen on, if a fiducial task exists."""
        if not self.config.sync_fiducial_position:
            return
        image = self._last_fib_image
        if image is None:
            return
        sync_fiducial_to_alignment_area(self.lamella, image)

    def _stage_position_now(self) -> Dict[str, Any]:
        return {"stage_position": self.microscope.get_stage_position()}

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

        from fibsem.alignment import ALIGNMENT_SUBDIR
        from fibsem.alignment.coincidence import ensure_coincident, tilt_coincident
        from fibsem.alignment.plotting import save_coincidence_diagnostics
        from fibsem.transformations import get_stage_tilt_from_milling_angle

        # the run's, beside its other alignments
        diagnostics_path = os.path.join(self.output_dir, ALIGNMENT_SUBDIR)

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
            logging.debug(
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

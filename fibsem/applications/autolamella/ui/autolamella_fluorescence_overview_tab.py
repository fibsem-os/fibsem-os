"""The FM Overview tab: the fluorescence overview, plus what an experiment adds to it.

`FMOverviewWidget` knows nothing about lamellae. That is deliberate and worth keeping --
it is what lets the same widget open standalone against a simulator, and be built in a
test from a microscope alone. It says *where* a user pointed and leaves the meaning to
whoever is listening.

This is whoever is listening, and it is the fluorescence twin of
`AutoLamellaOverviewTab`. Everything the two do identically lives in
`AutoLamellaOverviewTabBase`; what is left here is what makes this the fluorescence side
-- **fluorescence poses**, and the objective the sample was focused through.

A widget rather than a plain controller, because part of what it adds is *visible* --
`LamellaNameListWidget` needs real `Lamella` objects (it subscribes to
`lamella.events.description` and reads the defect and task state), so it cannot live
inside the fluorescence widget. Something has to lay the two out together, and a QWidget
that owns both is simpler than a controller reaching into another widget's layout.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.poses import (
    FLUORESCENCE_ORIENTATION,
    build_lamella_poses,
)
from fibsem.applications.autolamella.ui.overview_tab_base import (
    AutoLamellaOverviewTabBase,
)
from fibsem.ui import notification_service
from fibsem.ui.fm.widgets.fm_overview_widget import FMOverviewWidget
from fibsem.ui.utils import message_box_ui

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.structures import FibsemStagePosition

logger = logging.getLogger(__name__)


class AutoLamellaFluorescenceOverviewTab(AutoLamellaOverviewTabBase):
    """Drives `FMOverviewWidget` on behalf of an experiment."""

    POSE_NOUN = "fluorescence pose"
    OVERVIEW_NOUN = "FM overview"

    # ── what makes this the fluorescence side ────────────────────────────

    def _build_overview(self, microscope) -> QWidget:
        return FMOverviewWidget(microscope)

    def _can_build(self, microscope) -> bool:
        """Whether there is a fluorescence detector to drive.

        A capability rather than a flag, but the same build-or-drop answer: on a system
        without an FM the tab stays in place but dead, so the tab bar does not change
        shape between systems. Hiding it is the window's job.
        """
        return microscope.fm is not None

    def _pose_of(self, lamella) -> Optional["FibsemStagePosition"]:
        """The **fluorescence** pose, not the primary stage position.

        A lamella carries one of each and they are different places on this canvas -- on
        a compustage the stage flips 180 degrees between them -- so passing the milling
        pose would mark every lamella somewhere the sample is not.

        None is the normal state on a system with no FM, and possible on one loaded from
        an older experiment. The base reports those rather than dropping them in silence:
        a canvas showing three of five positions and saying nothing is worse than one
        saying which two it cannot show.
        """
        pose = lamella.fluorescence_pose
        if pose is None:
            return None
        return pose.stage_position

    def _add_lamella_kwargs(self) -> Dict[str, Any]:
        """The orientation is *declared* rather than left to be derived.

        On a compustage deriving it would give the same answer; on an offset mount it
        would not, and the wrong answer there is a lamella with a milling pose 48 mm off
        the beam axis that nothing rejects until something tries to mill it (FIB-93).
        Declaring it turns that into a refusal at the point of marking, where there is a
        user to tell.
        """
        return {
            "objective_position": self._objective_position(),
            "marked_at": FLUORESCENCE_ORIENTATION,
        }

    def _objective_position(self) -> Optional[float]:
        """Where the objective is right now, for a pose marked on the overview.

        The focus a user is actually looking through beats the objective's *configured*
        focus position, which `build_lamella_poses` falls back to: that one is a property
        of the instrument and says nothing about this sample. Anyone marking a position
        here has the feature in focus, and that focus is part of what they marked.

        Unless the objective is retracted, in which case its position is a parking spot
        ~10 mm from anything and nobody focused on anything. None hands the decision back
        to the fallback -- which matters because `fluorescence_selected` only asks
        whether an objective position exists, so a parked one would read as a focused
        lamella.
        """
        microscope = self.microscope
        if microscope is None or microscope.fm is None:
            return None
        try:
            objective = microscope.fm.objective
            if objective.state != "Inserted":
                logger.debug(
                    f"Objective is {objective.state}; falling back to its focus position."
                )
                return None
            return objective.position
        except Exception as e:
            logger.debug(f"Could not read the objective position: {e}")
            return None

    # ── turning a request into a lamella ─────────────────────────────────

    def _on_move_requested(self, name: str, position) -> None:
        """A user asked to move a marked lamella to a point on the overview.

        Confirmed first, because moving one pose moves both: the milling pose is derived
        from this one, so a lamella dragged across the fluorescence view also stops being
        where the beam was going to mill it. That is usually the point -- the two describe
        one piece of sample -- but it is not visible from this canvas, which shows only
        the fluorescence side, so it gets said rather than assumed.

        Not shared with the beam tab, which writes the milling pose directly and syncs
        this one after it. See `overview_tab_base`.
        """
        experiment = self.experiment
        if experiment is None:
            return
        lamella = next((p for p in experiment.positions if p.name == name), None)
        if lamella is None:
            logger.debug(f"Cannot move {name!r}: no such lamella in the experiment.")
            return
        if lamella.milling_pose is None:
            notification_service.show_toast(
                f"{name} has no milling pose to move.", "warning"
            )
            return

        try:
            poses = build_lamella_poses(
                microscope=self.microscope,
                position=position,
                objective_position=self._objective_position(),
                # Only reaches the result in one case, and it is worth it for that one:
                # a lamella with no fluorescence pose yet gets a whole new one below, and
                # it should be built on what this lamella recorded rather than on
                # whatever the microscope happens to be set to now. Everywhere else only
                # the stage positions are read, and those come from `position`.
                state=lamella.milling_pose,
                marked_at=FLUORESCENCE_ORIENTATION,
            )
        except Exception as e:
            logger.error(f"Could not move {name} from the FM overview: {e}")
            notification_service.show_toast(str(e), "error")
            return

        history = (
            f"\n\n{name} has already completed "
            f"{', '.join(lamella.completed_tasks)}."
            if lamella.completed_tasks
            else ""
        )
        if not message_box_ui(
            title=f"Move {name}?",
            text=(
                f"Move {name} to {poses.fluorescence.stage_position.pretty_string}?"
                f"\n\nThis moves the milling pose with it, to "
                f"{poses.milling.stage_position.pretty_string}."
                f"{history}"
            ),
            parent=self,
        ):
            return

        # Only the stage positions are replaced, so anything else the poses carry --
        # notably the objective position on a lamella that was focused by hand -- is
        # kept. `stage_position` is the milling pose's, via the property.
        lamella.stage_position = poses.milling.stage_position
        lamella.update_milling_angle(self.microscope)
        if lamella.fluorescence_pose is None:
            lamella.fluorescence_pose = poses.fluorescence
        else:
            lamella.fluorescence_pose.stage_position = poses.fluorescence.stage_position

        experiment.save()
        self.refresh_positions()
        self.autolamella_ui.update_ui()
        notification_service.show_toast(f"Moved {name}.", "info")

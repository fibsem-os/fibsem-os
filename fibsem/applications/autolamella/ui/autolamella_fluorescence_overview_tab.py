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
    FLUORESCENCE_POSE,
    Followed,
    followed_note,
    move_consequence,
    move_pose,
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
        """What this tab knows that the position does not.

        Where the objective is, and that the position was marked through it. Which
        side the position is on is read off the geometry by `build_lamella_poses`;
        `observed` only settles a position both instruments can use, where the
        geometry cannot say which one the person was looking through.
        """
        return {
            "objective_position": self._objective_position(),
            "observed": FLUORESCENCE_POSE,
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

        Moves the fluorescence pose. The milling pose does what the rule in
        `poses.move_pose` says: worked out again while it is still a guess -- so a
        lamella dragged here also stops being where the beam was going to mill it --
        and left alone once it has been set at the beams. Neither is visible from this
        canvas, which shows only the fluorescence side, so the move is confirmed first
        and the confirmation says which.

        Not shared with the beam tab, which does not confirm. See `overview_tab_base`.
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

        history = (
            f"\n\n{name} has already completed {', '.join(lamella.completed_tasks)}."
            if lamella.completed_tasks
            else ""
        )
        # Said before the move, because it is not visible from this canvas: whether
        # the milling pose comes along is decided by where that pose came from.
        consequence = move_consequence(lamella, FLUORESCENCE_POSE)
        if not message_box_ui(
            title=f"Move {name}?",
            text=(
                f"Move {name} to {position.pretty_string}?\n\n{consequence}{history}"
            ),
            parent=self,
        ):
            return

        # Only the stage position is replaced, so anything else the pose carries --
        # notably the objective position on a lamella that was focused by hand -- is
        # kept. A lamella with no fluorescence pose yet gets one built on what it
        # recorded, with the objective where it is now.
        followed = move_pose(
            self.microscope,
            lamella,
            FLUORESCENCE_POSE,
            position=position,
            objective_position=self._objective_position(),
        )

        experiment.save()
        # Writing a pose emits nothing, so the other canvas and the lamella cards
        # would go on showing the old place; the window re-marks both tabs from this.
        experiment.positions.events.changed.emit()
        self.refresh_positions()
        self.autolamella_ui.update_ui()
        level = "warning" if followed is Followed.FAILED else "info"
        notification_service.show_toast(
            f"Moved {name}. {followed_note(FLUORESCENCE_POSE, followed)}".strip(), level
        )

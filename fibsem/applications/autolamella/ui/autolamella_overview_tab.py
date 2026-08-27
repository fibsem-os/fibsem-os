"""The Overview tab: the FIB/SEM overview, plus what an experiment adds to it.

`FibsemOverviewWidget` knows nothing about lamellae. That is deliberate and worth
keeping -- it is what lets the same widget open standalone against a simulator, and be
built in a test from a microscope alone. It says *where* a user pointed and leaves the
meaning to whoever is listening.

This is whoever is listening, and it is the beam-side twin of
`AutoLamellaFluorescenceOverviewTab`. Everything the two do identically lives in
`AutoLamellaOverviewTabBase`; what is left here is what makes this the beam side --
**milling poses** and defects.

**Milling poses, not fluorescence ones** -- the opposite of its twin. This canvas looks
at the sample from the beam side, so a lamella is marked where the beam sees it. On a
compustage the two are 180 degrees apart, so passing the wrong one would put every
marker on the far side of the grid.

Why this exists at all
----------------------
It replaced `FibsemMinimapWidget`, which was the last module under `fibsem/ui/` that
imported AutoLamella (FIB-560): it reached through `parent_widget.experiment` in nineteen
places, typed its parent as `AutoLamellaUI`, and knew about `DefectType`,
`sync_fluorescence_pose` and milling protocol keys -- while also being the widget the
generic `FibsemUI` hosted. Splitting it along the line the parent contract already
suggested is what the pair does: `FibsemOverviewWidget` for anyone with a microscope,
this for anyone with an experiment. That widget is now deleted, and nothing under
`fibsem/ui/` imports AutoLamella.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.poses import sync_fluorescence_pose
from fibsem.applications.autolamella.structures import DefectType
from fibsem.applications.autolamella.ui.overview_tab_base import (
    AutoLamellaOverviewTabBase,
)
from fibsem.ui import notification_service
from fibsem.ui.widgets.overview_widget import FibsemOverviewWidget

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from fibsem.structures import FibsemStagePosition

logger = logging.getLogger(__name__)

# Defect states that make a lamella worth flagging on the canvas. Both mean "do not
# just re-target this": one has failed, the other needs work. The tab this replaces
# drew them red and orange; there is one warning colour here, because the list beside
# the canvas already distinguishes them and a second colour on a crosshair does not.
FLAGGED_DEFECT_STATES = (DefectType.FAILURE, DefectType.REWORK)


class AutoLamellaOverviewTab(AutoLamellaOverviewTabBase):
    """Drives `FibsemOverviewWidget` on behalf of an experiment."""

    POSE_NOUN = "position"
    OVERVIEW_NOUN = "overview"

    def __init__(self, autolamella_ui, parent: Optional[QWidget] = None):
        super().__init__(autolamella_ui, parent)
        # Whether the host wants a live widget here at all. True by default so that
        # building this tab and calling `refresh_microscope` is enough on its own, which
        # is how it is used standalone and in tests.
        #
        # **Nothing in production sets this any more.** It carried
        # `features.overview_canvas_tab` while this tab was staged in beside the napari
        # one; that flag is retired (FIB-780) and the tab ships to everyone, so the only
        # callers left are the tests below. Kept rather than deleted because it is the
        # beam-side half of the build-or-drop pair the fluorescence tab answers with
        # `_can_build`, and a modality that can be switched off is a plausible thing to
        # want back -- but it is dead code today and should be read as such.
        self._enabled = True

    # ── what makes this the beam side ────────────────────────────────────

    def _build_overview(self, microscope) -> QWidget:
        return FibsemOverviewWidget(microscope)

    def _can_build(self, microscope) -> bool:
        return self._enabled

    def _pose_of(self, lamella) -> Optional["FibsemStagePosition"]:
        """The **milling** pose, which is what `lamella.stage_position` is.

        This canvas looks at the sample from the beam side; marking the fluorescence
        poses here would put every lamella where the FM sees it, which on a compustage
        is the other side of the grid.
        """
        return lamella.stage_position

    def _configure_list(self) -> None:
        self.lamella_list.enable_defect_button(True)
        self.lamella_list.defect_changed.connect(self._on_defect_changed)

    def _show_positions(self, positions, lamellae) -> None:
        """Mark them, saying which are flagged.

        The defect is a judgement about a lamella rather than about a view of it, but
        only this canvas draws it -- the fluorescence tab has no defect concept, and the
        list beside both canvases shows the state either way.
        """
        flagged = [
            lamella.name
            for lamella in lamellae
            if lamella.defect.state in FLAGGED_DEFECT_STATES
        ]
        self.overview.set_positions(positions, flagged=flagged)

    # ── what the window asks ─────────────────────────────────────────────

    def set_enabled(self, enabled: bool) -> None:
        """Whether to hold a live widget at all.

        The feature flag, arriving from the window. Separate from hiding the tab because
        hiding it is not enough: the widget subscribes to the microscope for its
        lifetime, so one left behind goes on doing work on every stage move and every
        tile of every acquisition, for a tab nobody can open -- and goes on holding
        psygnal references to tear down later. Same reasoning, and the same build-or-drop
        answer, as the fluorescence tab's capability check.
        """
        enabled = bool(enabled)
        if enabled == self._enabled:
            return
        self._enabled = enabled
        self.refresh_microscope()

    # ── the list ─────────────────────────────────────────────────────────

    def _on_defect_changed(self, lamella) -> None:
        """Persist a defect set from the list, and re-flag the canvas.

        The row writes `lamella.defect` and emits; nothing was listening, so the change
        lived in memory and was gone on reload (FIB-564). Redrawing as well as saving,
        because the marker's colour is one of the things the defect decides.
        """
        experiment = self.experiment
        if experiment is None:
            return
        experiment.save()
        self.refresh_positions()

    # ── turning a request into a lamella ─────────────────────────────────

    def _on_move_requested(self, name: str, position) -> None:
        """A user asked to move a marked lamella to a point on the overview.

        Moves the milling pose, and takes the fluorescence one with it. They describe
        one piece of sample from two sides, so leaving the other behind would have it go
        on naming where this lamella *used to be* -- and nothing about a stale pose
        looks wrong.

        Not shared with the fluorescence tab, which derives both poses through
        `build_lamella_poses` and confirms first. See `overview_tab_base`.
        """
        experiment = self.experiment
        if experiment is None:
            return
        lamella = next((p for p in experiment.positions if p.name == name), None)
        if lamella is None:
            logger.debug(f"Cannot move {name!r}: no such lamella in the experiment.")
            return

        lamella.stage_position = position
        lamella.update_milling_angle(self.microscope)
        sync_fluorescence_pose(self.microscope, lamella)

        experiment.save()
        self.refresh_positions()
        notification_service.show_toast(f"Moved {name}.", "info")

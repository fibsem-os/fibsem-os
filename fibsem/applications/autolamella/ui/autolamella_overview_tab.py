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
import os
import shutil
from typing import TYPE_CHECKING, Optional

from PyQt5.QtWidgets import QWidget

from fibsem.applications.autolamella.poses import sync_fluorescence_pose
from fibsem.applications.autolamella.structures import DefectType, OverlayRecord
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
        overview = FibsemOverviewWidget(microscope)
        overview.gridbar_placement_changed.connect(self._save_gridbar_placement)
        overview.image_placement_changed.connect(self._save_image_placement)
        overview.image_display_changed.connect(self._save_image_display)
        overview.image_removed.connect(self._forget_image)
        return overview

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

    # ── the grid under the stage, and what was placed over it ────────────

    @property
    def current_grid(self):
        """The experiment's record for the grid the stage is over, or None.

        The holder says which grid is in the slot the stage is at, from its cached
        position -- no hardware read, this is asked from paint paths. A holder with a
        single grid loaded answers that grid even before the stage has been read, which
        is every Arctis. None when nothing is loaded, or the experiment has no record
        of the grid that is.
        """
        experiment = self.experiment
        stage = getattr(self.microscope, "_stage", None)
        if experiment is None or stage is None:
            return None
        try:
            grid = stage.current_grid
            if grid is None:
                loaded = stage.loaded_grids
                grid = loaded[0] if len(loaded) == 1 else None
        except Exception as e:  # noqa: BLE001 - a stage that cannot say has no grid
            logger.debug(f"Could not read the grid under the stage: {e}")
            return None
        return experiment.get_grid_by_name(grid.name) if grid is not None else None

    def _save_gridbar_placement(self) -> None:
        """The user placed the grid bars: keep it on the grid they are over."""
        grid = self.current_grid
        if grid is None or self.overview is None:
            logger.debug("No grid under the stage to keep the grid bar placement on.")
            return
        dx, dy, rotation = self.overview.gridbar_placement
        spacing, bar_width = self.overview.gridbar_pitch
        grid.set_overlay(
            OverlayRecord(
                kind="gridbar",
                dx=dx,
                dy=dy,
                rotation=rotation,
                pitch=spacing,
                bar_width=bar_width,
            )
        )
        try:
            self.experiment.save()
        except Exception as e:  # noqa: BLE001 - the placement is on screen either way
            logger.error(f"Could not save the grid bar placement: {e}")

    # ── images aligned over the grid ─────────────────────────────────────

    ALIGNED_IMAGES_DIR = "Aligned Images"

    def _image_source(self, grid, aligned) -> Optional[str]:
        """The image's file, relative to the grid's folder -- copied in if it is
        elsewhere, so the experiment stays self-contained."""
        if not aligned.path:
            return None
        root = str(self.experiment.grid_path(grid))
        path = os.path.abspath(aligned.path)
        if os.path.commonpath([root, path]) == os.path.abspath(root):
            return os.path.relpath(path, root)
        folder = os.path.join(root, self.ALIGNED_IMAGES_DIR)
        os.makedirs(folder, exist_ok=True)
        copied = os.path.join(folder, os.path.basename(path))
        if not os.path.exists(copied):
            shutil.copy2(path, copied)
        aligned.path = copied
        return os.path.relpath(copied, root)

    def _keep_image(self, key: str):
        """Write an aligned image's record on the grid under the stage -- where it
        is and how it is shown -- and save. The grid and the record, or None."""
        grid = self.current_grid
        aligned = self.overview.aligned_images.get(key) if self.overview else None
        if grid is None or aligned is None:
            logger.debug("No grid under the stage to keep the image on.")
            return None
        try:
            source = self._image_source(grid, aligned)
        except Exception as e:  # noqa: BLE001 - a copy that failed is said
            logger.error(f"Could not copy {aligned.label} into the grid folder: {e}")
            return None
        if source is None:
            logger.debug(f"{aligned.label} has no file to keep; not recorded.")
            return None
        dx, dy, rotation, scale = aligned.placement
        record = OverlayRecord(
            kind="image",
            source=source,
            dx=dx,
            dy=dy,
            rotation=rotation,
            scale=scale,
            fit=dict(aligned.fit),
            display=self.overview.aligned_images.display_state(key),
        )
        if aligned.record_id:
            record.id = aligned.record_id
        aligned.record_id = record.id
        grid.set_overlay(record)
        try:
            self.experiment.save()
        except Exception as e:  # noqa: BLE001 - the image is on screen either way
            logger.error(f"Could not save {aligned.label} on grid {grid.name}: {e}")
            return None
        return grid, record

    def _save_image_placement(self, key: str) -> None:
        """The user placed an aligned image: keep it on the grid they are over."""
        kept = self._keep_image(key)
        if kept is None:
            return
        grid, record = kept
        aligned = self.overview.aligned_images.get(key)
        logger.info(
            f"Kept {aligned.label} on grid {grid.name}: dx={record.dx:.3e} m"
            f" dy={record.dy:.3e} m rotation={record.rotation:.2f} deg"
            f" scale={record.scale:.4f}{' (from a fit)' if record.fit else ''};"
            f" source {record.source}"
        )

    def _save_image_display(self, key: str) -> None:
        """The user changed how an aligned image is shown: keep that too."""
        kept = self._keep_image(key)
        if kept is None:
            return
        grid, record = kept
        display = record.display
        shown = [
            f"{c['name']} {c['color']}{'' if c['visible'] else ' (hidden)'}"
            for c in display.get("channels", [])
        ]
        logger.info(
            f"Kept how {self.overview.aligned_images.get(key).label} is shown on grid"
            f" {grid.name}: opacity {display.get('opacity', 0):.2f},"
            f" {'signal only' if display.get('signal_only') else 'whole frame'},"
            f" channels {', '.join(shown) or 'none'}"
        )

    def _forget_image(self, _key: str, record_id: str) -> None:
        grid = self.current_grid
        if not record_id or grid is None:
            return
        grid.overlays = [o for o in grid.overlays if o.id != record_id]
        try:
            self.experiment.save()
        except Exception as e:  # noqa: BLE001
            logger.error(f"Could not save the experiment: {e}")

    def _restore_images(self, grid) -> None:
        """Lay the grid's recorded images over the canvas, where they were left."""
        self.overview.clear_aligned_images()
        if grid is None:
            return
        root = str(self.experiment.grid_path(grid))
        for record in grid.overlays:
            if record.kind != "image" or not record.source:
                continue
            path = os.path.join(root, record.source)
            if not os.path.isfile(path):
                logger.warning(f"Aligned image {record.source} is missing; not shown.")
                continue
            key = self.overview.load_aligned_image(path)
            if key is None:
                continue
            aligned = self.overview.aligned_images.get(key)
            aligned.record_id = record.id
            self.overview.aligned_images.set_placement(
                key, record.dx, record.dy, record.rotation, record.scale
            )
            if record.display:
                self.overview.set_aligned_image_display(key, record.display)

    def _restore_overlays(self) -> None:
        """Put the grid bars and the aligned images where this grid's records say.

        Through the widget's setters, which do not announce, so restoring never
        writes back what was just read.
        """
        if self.overview is None:
            return
        grid = self.current_grid
        grid_id = grid.id if grid is not None else None
        if grid_id == self._overlays_grid_id:
            return
        self._overlays_grid_id = grid_id
        self._restore_images(grid)
        record = grid.overlay_of("gridbar") if grid is not None else None
        if record is None:
            self.overview.set_gridbar_placement(0.0, 0.0, 0.0)
            return
        if record.pitch and record.bar_width:
            self.overview.set_gridbar_pitch(record.pitch, record.bar_width)
        self.overview.set_gridbar_placement(record.dx, record.dy, record.rotation)
        self.overview.overlay_controls.set_visible("gridbars", True)

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

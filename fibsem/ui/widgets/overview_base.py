"""What the FIB/SEM and fluorescence overview widgets share.

Both draw an `OverviewSession` on a real-space canvas: the marked positions, a click
to select one, a double-click to drive the stage there, a right-click to ask for a
position, and the gates on all of those. That part lives here once. What each one
images, how its canvas maps onto the stage and what it says about a position stay in
the widget, behind the hooks at the bottom of this class.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, List, Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QLabel, QWidget

from fibsem.imaging.overview_session import POSITIONS, SELECTION, OverviewSession
from fibsem.structures.settings import FibsemStagePosition
from fibsem.ui import notification_service
from fibsem.ui.control import control_of
from fibsem.ui.qt.connections import connect_on_gui_thread
from fibsem.ui.qt.threading import FunctionWorker
from fibsem.ui.widgets.canvas.overlays import stage_context
from fibsem.ui.widgets.custom_widgets import (
    ContextMenu,
    ContextMenuConfig,
    TitledPanel,
)

logger = logging.getLogger(__name__)


class OverviewWidgetBase(QWidget):
    """An overview session on a canvas: positions, picking, and moving the stage.

    A host (the AutoLamella and Volume overview tabs) drives either widget through
    this class: `set_positions`, `set_selected_position`, `move_to`,
    `set_interactive`, `set_save_directory`, `add_settings_section`, `is_acquiring`,
    and the signals below.
    """

    # Whether a run is in progress here. A *state*, re-emitted on every change
    # rather than an edge, so a host that connects late or recomputes from
    # several facts cannot end up holding a stale answer. The host that matters
    # is the window, which must stop the other overview driving the stage while
    # this one is mid-tileset.
    acquiring_changed = pyqtSignal(bool)
    # Requests, not commands: the widget knows nothing about lamellae, so a host
    # that owns an experiment decides what a position means and whether to ask
    # first. Each widget declares `position_add_requested` with what it carries.
    position_move_requested = pyqtSignal(str, object)  # name, FibsemStagePosition
    # A marked position was clicked, by the name it was marked under.
    position_selected = pyqtSignal(str)

    # Screen-space hit radius for picking a marker, matching `PointOverlay`'s. In
    # pixels rather than stage microns so how close you have to click does not change
    # with the zoom.
    PICK_RADIUS_PX = 12
    # The right-click entry that asks for a new position.
    ADD_POSITION_LABEL = "Add Position Here"

    def __init__(
        self, session: OverviewSession, parent: Optional[QWidget] = None
    ) -> None:
        super().__init__(parent)
        # The positions, selection, lock, run state, mosaic and cached stage
        # position: the overview's, which a host writes.
        self.session = session
        self.microscope = session.microscope
        # Set by `_on_move_errored` so `_on_move_finished`, which always runs, knows
        # not to clear a message the failure has just put up.
        self._move_failed: bool = False
        self._unfollow: List[Callable[[], None]] = []

    def _follow_session(self) -> None:
        """Hear the session, the stage and the holder, on the GUI thread.

        All three are psygnals that fire on whichever thread changed them -- a run's
        worker, for the stage. They outlive this widget, so `closeEvent` lets go.
        """
        follow = [(self.session.changed, self._on_session_changed)]
        if self.microscope.stage is not None:
            follow.append(
                (self.microscope.stage.position.changed, self._on_stage_moved)
            )
        if self.microscope.sample_handling is not None:
            follow.append(
                (
                    self.microscope.sample_handling.holder_changed,
                    self._on_holder_changed,
                )
            )
        self._unfollow = [
            connect_on_gui_thread(self, signal, slot) for signal, slot in follow
        ]

    # ── state ────────────────────────────────────────────────────────────

    @property
    def is_acquiring(self) -> bool:
        """Whether an overview acquisition is running here.

        `_running` as well as the worker, and the order is why: `acquire` calls
        `_set_running(True)` *before* it builds the worker, so for the width of that
        gap a worker-only answer says no while a run is starting. That gap is exactly
        when `acquiring_changed` is emitted, so a host locking the other overview off
        this property would get False and lock nothing.

        Keeping the worker check as well as the flag, rather than replacing it: the
        union is true over a superset of the interval either is, and every caller is a
        guard, so being early and late is the safe direction to be wrong in.
        """
        return self.session.running or (
            self._worker is not None and self._worker.is_alive()
        )

    def set_interactive(self, enabled: bool, reason: str = "") -> None:
        """Allow or forbid starting work, without touching a run already in progress.

        For a host that owns the instrument for a while -- an AutoLamella workflow --
        and needs the tab to stop competing for it. Cancel stays live regardless, so a
        lock arriving mid-run cannot strand an acquisition with no way to stop it.

        *reason* completes the sentence "Cannot move the stage while ___" when a move is
        refused. The widget cannot know why it was locked -- a workflow owning the
        instrument and the other overview being mid-tileset are the same `False` here --
        and a refusal naming the wrong one is barely better than one naming nothing.
        """
        self.session.set_interactive(enabled, reason)

    def cancel(self) -> None:
        if not self.is_acquiring:
            return
        logger.info("Cancelling overview acquisition")
        self._cancelling = True
        self.microscope.imaging.stop()
        self._status_line().setText("Cancelling…")
        # A run stops at the next tile boundary, so the button stays there for a while
        # after it is pressed. Disabled once asked, or a second press reads as the first
        # one not having worked.
        self._apply_enabled_state()

    # ── positions ────────────────────────────────────────────────────────

    def set_positions(
        self,
        positions: List[FibsemStagePosition],
        flagged: Optional[Iterable[str]] = None,
    ) -> None:
        """Mark these positions on the canvas, replacing whatever was marked.

        Names are carried onto the markers. *flagged* names a subset to draw in the
        warning colour; the widget does not know or ask what that means.
        """
        self.session.set_positions(positions, flagged)

    def set_selected_position(self, name: Optional[str]) -> None:
        """Highlight one marked position by name, or None to highlight nothing.

        By name rather than index: the host's list and this one are rebuilt
        independently from the same experiment, and an index would silently point at a
        different lamella the moment one was removed.
        """
        self.session.select(name or None)

    @property
    def selected_position(self) -> Optional[str]:
        return self.session.selected

    def _on_session_changed(self, part: str) -> None:
        if part in (POSITIONS, SELECTION):
            self._refresh_positions()
        else:
            self._apply_enabled_state()

    # ── the holder and the overlays ──────────────────────────────────────

    def reset_context_overlay_defaults(self) -> None:
        """Re-resolve the holder-dependent overlay defaults.

        The switches were resolved when the tab was built, against the holder as it
        was then. Calibrating a slot mid-session changes the answer, and the
        boundary and slot markers should come on without a reconnect. Only when
        the answer changed: a rename fires the same holder signal, and must not
        undo a toggle the user made.
        """
        calibrated = stage_context.holder_is_calibrated(self.microscope)
        if calibrated == self._context_defaults_for_calibrated:
            return
        self._context_defaults_for_calibrated = calibrated
        for key, _label, shown in stage_context.context_overlay_entries(
            self.microscope
        ):
            self.overlay_controls.set_visible(key, shown)

    def _on_holder_changed(self, _holder) -> None:
        """Slots calibrated, or another holder fitted: redraw where they are.

        Always, not only when the overlay defaults flip -- recalibrating a slot on a
        holder that was already calibrated changes nothing `reset_context_overlay_defaults`
        looks at, and moves the slot all the same.
        """
        self.reset_context_overlay_defaults()
        self._redraw_stage_context()

    def _toggle_overlays(self) -> None:
        """Show or hide the overlays popover, anchored under its button."""
        self.overlay_popover.set_open(self.btn_overlays.isChecked(), self.btn_overlays)

    def _section(self, title: str, widget: QWidget) -> QWidget:
        return TitledPanel(title, content=widget)

    # ── moving the stage ─────────────────────────────────────────────────

    def move_to(self, position: FibsemStagePosition) -> None:
        """Drive the stage to *position*, off the GUI thread.

        Public because a host drives it too -- picking a lamella out of a list is the
        same act as double-clicking where it is drawn, so it goes through the same gate
        and is reported the same way.

        The limits are checked here rather than trusted from the caller: a stored
        position can be outside them on a system it was not recorded on.
        """
        if not self._may_move() or not self._within_limits(position):
            return
        # Said on the status line: a double-click that starts a multi-second stage
        # move is otherwise indistinguishable from one that did nothing. A
        # move has no fraction to put on a bar.
        self._status_line().setText(f"Moving to {self._describe(position)}…")
        worker = FunctionWorker(self.session.move_to, position)
        worker.errored.connect(self._on_move_errored)
        worker.finished.connect(self._on_move_finished)
        worker.start(control=control_of(self.microscope), action="move stage")

    def _on_move_errored(self, error: object) -> None:
        """The stage did not get there. Say that, rather than leaving "Moving to …" up."""
        self._move_failed = True
        self._status_line().setText(f"Could not move the stage: {error}")
        notification_service.show_toast("Could not move the stage.", "error")

    def _on_move_finished(self) -> None:
        """Always runs, after `errored` when there was one.

        A failure has already put its own message up and that should stand; only a
        success replaces the "Moving to …" line, with where the stage actually got to,
        read back by the worker rather than assumed from the target.
        """
        if self._move_failed:
            self._move_failed = False
            return
        position = self.session.stage_position or self.session.read_stage_position()
        self._status_line().setText(
            f"At {self._describe(position)}" if position is not None else "Moved."
        )

    def _may_move(self) -> bool:
        """Whether driving the stage from this tab is allowed right now, and say if not.

        The lock covers the canvas as well as the buttons.
        """
        refusal = (
            "Cannot move the stage during an acquisition."
            if self.is_acquiring
            else self.session.move_refusal() or self._orientation_refusal()
        )
        if refusal is not None:
            notification_service.show_toast(refusal, "warning")
            return False
        return True

    def _within_limits(self, target: FibsemStagePosition) -> bool:
        """Whether the stage can reach *target* in x and y, and say if not."""
        limits = self.session.stage_limits()
        if limits and not target.is_within_limits(limits, axes=["x", "y"]):
            notification_service.show_toast(
                "That position is outside the stage limits.", "warning"
            )
            return False
        return True

    # ── canvas interaction ───────────────────────────────────────────────

    def _on_canvas_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Select a marked position if the click landed on one.

        A miss leaves the selection alone rather than clearing it: a host syncs its own
        lists from the selection and their handlers ignore None, so an empty click would
        blank this canvas and nothing else.
        """
        name = self._position_at(x, y)
        if name is None:
            return
        self.set_selected_position(name)
        self.position_selected.emit(name)

    def _on_canvas_double_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Move the stage to the double-clicked point.

        Double-click rather than a single click, matching the coincidence viewer: a
        single click is how the canvas is explored, and a stage that moved on every
        stray click would be unusable.
        """
        target = self._stage_position_at(x, y) if self._may_move() else None
        if target is None:
            return
        self.move_to(target)

    def _on_canvas_right_clicked(self, x: float, y: float, modifiers=None) -> None:
        """Offer to put a position at the right-clicked point."""
        config = self._position_menu(x, y)
        if config is None:
            return
        ContextMenu(config, parent=self).show_at_cursor()

    def _position_menu(self, x: float, y: float) -> Optional[ContextMenuConfig]:
        """What right-clicking at a canvas point offers, or None to offer nothing.

        Separate from showing it because `show_at_cursor` runs a modal event loop, and
        everything worth checking -- whether the menu appears at all, what it offers,
        and which position each entry would request -- is decided here. A test that had
        to open the menu could only hang.

        The widget creates nothing itself: each entry emits a request and lets a host
        that owns an experiment decide. So the menu appears whenever the point is
        somewhere the stage could go, even with nobody listening.
        """
        if self.session.running or self.is_acquiring or not self.session.interactive:
            # Gated on what `_apply_enabled_state` gates the controls on: a host that
            # has taken the instrument is usually a workflow iterating the positions,
            # and adding one underneath it is not a thing to do quietly.
            notification_service.show_toast(
                "Cannot mark positions while an acquisition is running.", "warning"
            )
            return None
        refusal = self._marking_refusal()
        if refusal is not None:
            notification_service.show_toast(refusal, "warning")
            return None

        target = self._stage_position_at(x, y)
        if target is None:
            return None

        config = ContextMenuConfig()
        config.add_action(
            self.ADD_POSITION_LABEL,
            callback=lambda: self._request_add_at(x, y, target),
            tooltip=f"Add a position at {self._describe(target)}",
        )
        selected = self.session.selected
        if selected:
            config.add_action(
                f"Move Selected Position Here ({selected})",
                callback=lambda: self.position_move_requested.emit(selected, target),
                tooltip=f"Move {selected} to {self._describe(target)}",
            )
        return config

    def _position_at(
        self, x: float, y: float, crosshair_only: bool = False
    ) -> Optional[str]:
        """The marked position under a canvas point, or None.

        A click hits a position if it lands inside that position's field-of-view box
        **or** within `PICK_RADIUS_PX` of its crosshair. The union rather than either
        alone, because neither is reliably the bigger target: the box wins once you are
        zoomed into a region, the fixed radius wins at whole-grid zoom where the box
        shrinks below it -- see `FieldOfViewOverlay.covers`.

        The radius is measured on screen, not in data units: at a wide zoom every marker
        would be within any sensible micron radius of the click, and at a tight one none
        would be. Nearest crosshair wins among the hits, which also settles overlapping
        boxes -- and lamellae closer together than the field of view do overlap.

        *crosshair_only* drops the box and leaves the radius, for a caller that has to
        share the canvas with something else. The tile grid stands aside wherever this
        answers a name, and a field-of-view box is a large thing to reserve:
        it is a whole tile on the fluorescence tab by construction, and a whole tile on
        the beam one at any HFW of 100 um or under. Reserving it would leave tiles that
        cannot be toggled at all, with nothing on screen to say why -- where reserving
        only the crosshair costs at worst a click that toggles a tile you meant to
        select, which greys out visibly and undoes with one more click.
        """
        if not self._positions_pickable():
            return None
        frame = self._frame()
        ax = getattr(self._real_space_canvas(), "_ax", None)
        if frame is None or ax is None or not self.session.positions:
            return None
        try:
            transform = ax.transData
            click = transform.transform((x, y))
        except Exception as e:
            logger.debug(f"Could not resolve the click for picking: {e}")
            return None

        best_name, best_distance = None, float("inf")
        for position in self.session.positions:
            if not position.name:
                continue
            try:
                centre = frame.to_canvas(position)
                point = transform.transform(centre)
            except Exception:
                continue
            distance = ((click[0] - point[0]) ** 2 + (click[1] - point[1]) ** 2) ** 0.5
            # `centre` is in canvas units and `point` in screen pixels: the box is a
            # fixed piece of sample, the radius a fixed piece of screen.
            if distance >= self.PICK_RADIUS_PX and (
                crosshair_only or not self.position_overlay.covers(centre, x, y)
            ):
                continue
            if distance < best_distance:
                best_name, best_distance = position.name, distance
        return best_name

    # ── lifecycle ────────────────────────────────────────────────────────

    def hideEvent(self, event) -> None:
        # The channel controls and the tile grid panel are top-level windows, so they
        # do not go with this widget on their own, and would float over the next tab.
        # The tile grid button is unchecked with its panel, so one click opens it
        # again. Reached from C++ during teardown too, where raising is fatal under
        # PyQt5, so looked up tolerantly and never raises.
        super().hideEvent(event)
        channels = self.__dict__.get("_channels_panel")
        tile_grid = self.__dict__.get("tile_grid_panel")
        button = self.__dict__.get("btn_tile_grid")
        try:
            if channels is not None:
                channels.hide()
            if tile_grid is not None:
                tile_grid.hide()
            if button is not None:
                button.setChecked(False)
        except RuntimeError:  # wrapped C/C++ object already deleted
            pass

    def closeEvent(self, event) -> None:
        if self.is_acquiring:
            self.microscope.imaging.stop()
        # Every psygnal this widget subscribed to. They outlive the widget -- they
        # belong to the microscope -- and an emit into a widget Qt has torn down
        # writes into freed memory: closing the tab and then moving the stage from
        # anywhere else was a hard segfault, not an exception.
        for unfollow in self._unfollow:
            unfollow()
        for signal, slot in self._subscriptions():
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError, ValueError, KeyError):
                pass
        super().closeEvent(event)

    # ── what each widget supplies ────────────────────────────────────────

    def set_save_directory(self, path: Optional[str]) -> None:
        """Where acquired overviews are written."""
        raise NotImplementedError

    def add_settings_section(self, title: str, widget: QWidget) -> None:
        """Add a host's panel to the settings column."""
        raise NotImplementedError

    def _status_line(self) -> QLabel:
        """The label under the settings that reports moves and outcomes."""
        raise NotImplementedError

    def _real_space_canvas(self):
        """The `FibsemRealSpaceCanvas` positions are drawn on."""
        raise NotImplementedError

    def _frame(self, fresh: bool = False):
        """The `StageFrame` between canvas and stage, or None when there is none yet."""
        raise NotImplementedError

    def _stage_position_at(self, x: float, y: float) -> Optional[FibsemStagePosition]:
        """The stage position a canvas point names, or None if it is not usable."""
        raise NotImplementedError

    def _describe(self, position: FibsemStagePosition) -> str:
        """A position in the words the status line uses."""
        raise NotImplementedError

    def _request_add_at(self, x: float, y: float, target: FibsemStagePosition) -> None:
        """Emit `position_add_requested` for *target*."""
        raise NotImplementedError

    def _on_stage_moved(self, position: FibsemStagePosition) -> None:
        raise NotImplementedError

    def _refresh_positions(self) -> None:
        """Redraw the marked positions and the selection."""
        raise NotImplementedError

    def _redraw_stage_context(self) -> None:
        """Redraw what is drawn from the holder: slots, boundary, limits."""
        raise NotImplementedError

    def _apply_enabled_state(self) -> None:
        raise NotImplementedError

    def _subscriptions(self) -> list:
        """The (psygnal, slot) pairs this widget connected itself, for `closeEvent`."""
        return []

    def _positions_pickable(self) -> bool:
        """Whether a click may pick a marked position."""
        return True

    def _orientation_refusal(self) -> Optional[str]:
        """Why the stage may not be moved from this pose, or None."""
        return None

    def _marking_refusal(self) -> Optional[str]:
        """Why a position may not be marked from this pose, or None."""
        return None

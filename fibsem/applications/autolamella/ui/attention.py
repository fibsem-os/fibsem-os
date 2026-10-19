"""Where a proposal a task asks mid-run goes up, and the way back to it.

A proposal is put up on the widget its kind is decided on as it is recorded,
and taken down however it is decided. While the run holds, Attention Required
shows on the status bar, the line says what releases the hold, the timeline's
countdown stops, and the window beeps once.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QApplication

from fibsem.applications.autolamella.ui.loop_proposal import LOOP_KINDS
from fibsem.workflows.proposals import ALIGNMENT_AREA, DETECTION, POINT_OF_INTEREST

# The kinds a task asks whose value is edited on a widget of its own.
WIDGET_KINDS = (DETECTION, ALIGNMENT_AREA, POINT_OF_INTEREST)


def play_notification_sound():
    """Play a notification sound to alert the user."""
    QApplication.beep()


class AttentionRouter(QObject):
    def __init__(
        self,
        *,
        prompts,
        tab_widget,
        controls,
        status_bar,
        timeline,
        sound_enabled: Callable[[], bool],
        parent: Optional[QObject] = None,
    ) -> None:
        """*prompts* puts proposals up and takes them down (``AutoLamellaUI``);
        *controls* holds Attention Required."""
        super().__init__(parent)
        self._prompts = prompts
        self._tabs = tab_widget
        self._controls = controls
        self._status_bar = status_bar
        self._timeline = timeline
        self._sound_enabled = sound_enabled
        self._sounded = False
        self._experiment = None
        controls.attention_clicked.connect(self.attention_clicked)

    def listen(self, experiment) -> None:
        """One subscription per experiment; the previous experiment's is
        dropped. A proposal already standing when the window starts listening
        (a run that began without it) is put up at once."""
        previous = self._experiment
        if previous is experiment:
            return
        if previous is not None:
            try:
                previous.asked.disconnect(self._on_asked)
                previous.decided.disconnect(self._on_decided)
            except Exception:
                pass
        self._experiment = experiment
        if experiment is not None:
            experiment.asked.connect(self._on_asked)
            experiment.decided.connect(self._on_decided)
            for item, task_name, _proposal in experiment.pending_proposals():
                self._on_asked(item.id, task_name)

    def _on_asked(self, item_id: str, task_name: str) -> None:
        """Put a proposal a task just asked in front of the operator, where
        its kind is decided: a mill's or a spot burn's pass on the prompt bar
        over its widget, where the operator edits what runs; a detection, an
        alignment area or a point of interest on the widget it is edited on;
        anything else (a state) on the prompt bar of the Microscope tab, where
        Continue confirms it as it stands."""
        experiment = self._experiment
        if experiment is None:
            return
        item = experiment.get_item_by_id(item_id)
        proposal = item.proposal(task_name) if item is not None else None
        if proposal is None or not proposal.asking:
            return
        if proposal.kind in LOOP_KINDS:
            self._prompts.show_loop_proposal(item_id, task_name, proposal)
        elif proposal.kind in WIDGET_KINDS:
            self._prompts.show_widget_proposal(item_id, task_name, proposal)
        else:
            self._prompts.show_state_proposal(item_id, task_name, proposal)

    def _on_decided(self, item_id: str, task_name: str) -> None:
        """However a proposal was decided -- the prompt bar, an agent, or
        withdrawn by Stop -- the prompt comes down."""
        self._prompts.clear_state_proposal(item_id, task_name)
        self._prompts.clear_loop_proposal(item_id, task_name)
        self._prompts.clear_widget_proposal(item_id, task_name)

    def attention_clicked(self) -> None:
        """Attention Required: go to the proposal.

        A proposal asked on a tab of its own -- a detection, a mill -- is not on
        the Microscope tab, so landing there leaves the operator looking for a
        prompt that is somewhere else. The destination comes from the proposal,
        never from the hold's kind: the kind says only that something holds the
        run.
        """
        host = self._prompts.proposal_host()
        if host is not None and self._tabs.indexOf(host) != -1:
            # A proposal decided on one of the window's own tabs.
            self._tabs.setCurrentWidget(host)
            return
        self._tabs.setCurrentIndex(0)  # Microscope tab is index 0
        self._prompts.front_proposal()

    def show_hold(self, hold) -> None:
        """The run holds on *hold*, or nothing holds it (None)."""
        # The timeline freezes its countdown on a hold: a wait for an answer or
        # a decision is not machine time whoever is giving it.
        self._timeline.set_waiting_for_user(hold is not None)
        # A proposal is a wait for the operator, whether or not an agent is
        # connected and may answer first. The line says what releases a hold
        # the operator can release, and counts the wait.
        self._status_bar.set_waiting(hold.releases if hold is not None else None)
        if hold is None:
            self._controls.set_attention(None)
            self._sounded = False
            return
        self._controls.set_attention(
            "Attention Required", f"The run is waiting on you: {hold.releases}."
        )
        if not self._sounded and self._sound_enabled():
            play_notification_sound()
            self._sounded = True

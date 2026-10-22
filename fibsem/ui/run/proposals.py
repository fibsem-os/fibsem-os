"""Proposals put to the operator in the panel: a state question on the prompt
bar, a loop's pass over its widget, a value edited on its own widget.

What `AttentionRouter` puts proposals up and takes them down through. Each
button records its decision through ``Experiment.decide``, the one write path.
"""

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

from fibsem.ui.widgets.canvas.overlays.poi_marker import PoiMarker

if TYPE_CHECKING:
    from fibsem.ui.qt.run_bridge import RunRelay
    from fibsem.ui.widgets.microscope_panel import MicroscopePanel
    from fibsem.workflows.deciding import ProposalDecisions
    from fibsem.workflows.session import AppSession

__all__ = ["ProposalPresenter"]


class ProposalPresenter:
    def __init__(
        self,
        session: "AppSession",
        panel: "MicroscopePanel",
        relay: "RunRelay",
        *,
        view_controller: Callable[[], Any],
    ) -> None:
        self._session = session
        self._panel = panel
        self._relay = relay
        # The point-of-interest marker on the FIB view, for the POI proposal.
        self.poi_marker = PoiMarker(
            image_widget=lambda: panel.image_widget,
            view_controller=view_controller,
        )
        # A state question a task asked (``BaseTask.ask`` with the
        # ``state`` kind), shown on the prompt bar: (item_id, task_name,
        # proposal_id). Continue decides it; the decision takes it down.
        self._state_proposal: Optional[Tuple[str, str, str]] = None
        # A pass of a loop a task asked (``milling``, ``spot_burn``), shown on
        # the prompt bar over its widget: (item_id, task_name, proposal). A
        # button decides it with the widget's value; the decision takes it down.
        self._loop_proposal: Optional[Tuple[str, str, Any]] = None
        # A value a task asked for that is edited on a widget of its own (a
        # ``detection`` on the Detection tab, an ``alignment_area`` or a
        # ``point_of_interest`` on the FIB view): (item_id, task_name,
        # proposal). Continue decides it with the widget's value.
        self._widget_proposal: Optional[Tuple[str, str, Any]] = None
        self._poi_frames_followed: Optional[object] = None

        panel.prompt_bar.answered.connect(self._on_prompt_answered)

    @property
    def experiment(self) -> Optional["ProposalDecisions"]:
        return self._session.experiment

    def _agent_deciding(self) -> bool:
        """The agent decides proposals now (Drive, listening): the panel shows
        them without taking the operator's view to them."""
        from fibsem.acting import AGENT

        return AGENT in self._session.deciders()

    def set_agent_deciding(self, deciding: bool) -> None:
        """Who decides changed while a proposal is up: the prompt bar says so."""
        self._panel.prompt_bar.set_agent_deciding(deciding)

    def _ask(self, message: str, positive: str, negative=None) -> None:
        bar = self._panel.prompt_bar
        bar.ask(message, positive, negative)
        bar.set_agent_deciding(self._agent_deciding())

    def _on_prompt_answered(self, clicked_yes: bool) -> None:
        # A click with no proposal up (a stray double-click after the
        # decision landed) means nothing.
        if self._loop_proposal is not None:
            self._decide_loop_proposal(clicked_yes)
            return
        if self._widget_proposal is not None:
            if clicked_yes:
                self._decide_widget_proposal()
            return
        self._decide_state_proposal(clicked_yes)

    def show_state_proposal(self, item_id: str, task_name: str, proposal) -> None:
        """The prompt bar is the renderer for the ``state`` kind: the task's
        message and its button, Continue unless the task names another
        (``positive`` in the provenance). Continue records the operator's
        confirmation through ``Experiment.decide``, the one write path; the
        task then reads the instrument for the position as confirmed. A task
        that can do without it names a second button (``negative``: the
        tilt's Skip), which rejects it: "don't", and the task carries on."""
        from fibsem.workflows.status import (
            WorkflowStatusEvent,
        )

        self._state_proposal = (item_id, task_name, proposal.id)
        message = str(
            proposal.provenance.get("message") or "Press Continue when ready."
        )
        self._ask(
            message,
            str(proposal.provenance.get("positive") or "Continue"),
            proposal.provenance.get("negative") or None,
        )
        self._relay.status.emit(WorkflowStatusEvent())

    def clear_state_proposal(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the question up,
        the prompt comes down (answered here, by an agent, or withdrawn)."""
        shown = self._state_proposal
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.workflows.status import (
            WorkflowStatusEvent,
        )

        self._state_proposal = None
        self._panel.prompt_bar.clear()
        self._relay.status.emit(WorkflowStatusEvent(message=""))

    def _decide_state_proposal(self, clicked_yes: bool = True) -> None:
        shown = self._state_proposal
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal_id = shown
        from fibsem.workflows.proposals import Decision, DecisionOutcome

        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed
                if clicked_yes
                else DecisionOutcome.Rejected,
                actor=experiment.actor(),
                via="workflow",
                proposal_id=proposal_id,
                reason=""
                if clicked_yes
                else f"{self._panel.prompt_bar.no.text()} pressed",
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(
                f"{task_name}: could not confirm the state: {result.reason}"
            )
            self._panel.prompt_bar.reopen()

    def _loop_widget(self, kind: str):
        """The widget a pass of ``kind`` is decided in, or None without one."""
        from fibsem.workflows.proposals import MILLING, SPOT_BURN

        if kind == MILLING:
            return self._panel.milling_task_config_widget
        if kind == SPOT_BURN:
            return self._panel.spot_burn_widget
        return None

    def show_loop_proposal(self, item_id: str, task_name: str, proposal) -> None:
        """The prompt bar over the milling editor (or the spot-burn widget) is
        the renderer for a loop's pass: the task's message and the pass's two
        buttons. The task has already put what the pass would run in the
        widget; the operator edits it there, and a button records the decision
        through ``Experiment.decide`` with the widget's value. Without the
        widget (no spot-burn hardware) the buttons decide the pass as
        proposed."""
        from fibsem.workflows.proposals import SPOT_BURN, loop_prompt
        from fibsem.workflows.status import WorkflowStatusEvent

        widget = self._loop_widget(proposal.kind)
        if widget is None:
            pass
        elif proposal.kind == SPOT_BURN:
            widget.set_settings(proposal.values["spot_burn"])
            self._panel.set_spot_burn_widget_active(True)
            widget.set_workflow_mode(True)
        else:
            widget.setEnabled(True)
            if not self._agent_deciding():
                self._panel.front_tab(widget)
            # The task owns the mill while it asks: the editor's own Run stands down.
            widget.milling_widget.pushButton_run_milling.setVisible(False)
        self._loop_proposal = (item_id, task_name, proposal)
        prompt = loop_prompt(proposal)
        self._ask(prompt.message, prompt.positive, prompt.negative)
        self._relay.status.emit(WorkflowStatusEvent())

    def loop_proposal_host(self):
        """The widget the pass now up is decided in, for the attention button
        to go back to; None when no pass is up."""
        if self._loop_proposal is None:
            return None
        return self._loop_widget(self._loop_proposal[2].kind)

    def clear_loop_proposal(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the pass up, the
        prompt comes down (decided here, by an agent, or withdrawn). The widget keeps its value: the task asks again, or clears
        it when the loop ends."""
        shown = self._loop_proposal
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.workflows.status import WorkflowStatusEvent

        self._loop_proposal = None
        self._panel.prompt_bar.clear()
        self._relay.status.emit(WorkflowStatusEvent(message=""))

    def _decide_loop_proposal(self, clicked_yes: bool) -> None:
        shown = self._loop_proposal
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal = shown
        from fibsem.workflows.proposals import (
            ACTION_EDIT,
            ACTION_RUN,
            SPOT_BURN,
            Decision,
            DecisionOutcome,
            loop_prompt,
        )

        prompt = loop_prompt(proposal)
        action = prompt.yes if clicked_yes else prompt.no
        if action is None:
            return
        values = {}
        widget = self._loop_widget(proposal.kind)
        if action in (ACTION_RUN, ACTION_EDIT) and widget is not None:
            # What runs is what the widget holds: the operator's edits. Skip
            # and accept carry nothing: the task goes on with what it had.
            if proposal.kind == SPOT_BURN:
                values = {"spot_burn": deepcopy(widget.get_settings())}
            else:
                values = {"milling_config": deepcopy(widget.get_config())}
        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor=experiment.actor(),
                values=values,
                via="workflow",
                proposal_id=proposal.id,
                action=action,
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(f"{task_name}: could not decide the pass: {result.reason}")
            self._panel.prompt_bar.reopen()

    def show_widget_proposal(self, item_id: str, task_name: str, proposal) -> None:
        """The widget a value is edited on is the renderer for its kind: a
        detection's features on the Detection tab (the task loaded them there
        first), an alignment area as the rectangle on the FIB view, a point of
        interest as the marker on it. The prompt bar carries the task's
        message and Continue, which records the decision through
        ``Experiment.decide`` with the widget's value. Without the widget the
        Continue confirms the value as proposed."""
        from fibsem.workflows.proposals import (
            ALIGNMENT_AREA,
            DETECTION,
            POINT_OF_INTEREST,
        )
        from fibsem.workflows.status import WorkflowStatusEvent

        message = str(proposal.provenance.get("message") or "")
        # The agent deciding, the values are shown where they are edited, but
        # the operator's view stays where it is.
        front = not self._agent_deciding()
        if proposal.kind == DETECTION and self._detection_shown():
            if front:
                self._panel.front_tab(self._panel.det_widget)
            message = "Confirm Feature Detection. Press Continue to proceed."
        elif proposal.kind == ALIGNMENT_AREA and self._panel.image_widget is not None:
            if front:
                self._panel.show_experiment_tab()
            self._panel.image_widget.toggle_alignment_area(
                proposal.values.get("alignment_area")
            )
            message = f"{message} Press Continue when done.".strip()
        elif proposal.kind == POINT_OF_INTEREST and self.poi_marker.show(
            proposal.values.get("poi"),
            fallback=self._reference_image(item_id, proposal),
        ):
            if front:
                self._panel.show_experiment_tab()
            self._follow_poi_frames()
        self._widget_proposal = (item_id, task_name, proposal)
        self._ask(message or "Press Continue when ready.", "Continue")
        self._relay.status.emit(WorkflowStatusEvent())

    def _reference_image(self, item_id: str, proposal):
        """The image the proposal's values sit on, for a view showing none."""
        from fibsem.workflows.reference_images import load_reference_image

        item = self.experiment.get_item_by_id(item_id)
        return load_reference_image(self.experiment, item, proposal)

    def _detection_shown(self) -> bool:
        """The Detection tab holds the features the task sent before asking;
        a window that started listening mid-question has none to correct."""
        return (
            self._panel.det_widget is not None
            and self._panel.det_widget.det is not None
        )

    def widget_proposal_host(self):
        """The tab the value now up is edited on, when it has one of its own
        (a detection), for the attention button to go back to."""
        from fibsem.workflows.proposals import DETECTION

        shown = self._widget_proposal
        if shown is None or shown[2].kind != DETECTION:
            return None
        return self._panel.det_widget

    def _follow_poi_frames(self) -> None:
        """Keep the marker on the frame the FIB view shows, once per widget."""
        signal = getattr(self._panel.image_widget, "viewer_update_signal", None)
        if signal is None or self._poi_frames_followed is self._panel.image_widget:
            return
        signal.connect(self._sync_poi_proposal)
        self._poi_frames_followed = self._panel.image_widget

    def _sync_poi_proposal(self, *_args) -> None:
        from fibsem.workflows.proposals import POINT_OF_INTEREST

        shown = self._widget_proposal
        if shown is not None and shown[2].kind == POINT_OF_INTEREST:
            self.poi_marker.sync()

    def clear_widget_proposal(self, item_id: str, task_name: str) -> None:
        """A decision landed on the item and task; if it is the value up, the
        prompt and the widget's editing come down (decided here, by an agent,
        or withdrawn)."""
        shown = self._widget_proposal
        if shown is None or shown[:2] != (item_id, task_name):
            return
        from fibsem.workflows.proposals import (
            ALIGNMENT_AREA,
            DETECTION,
            POINT_OF_INTEREST,
        )
        from fibsem.workflows.status import WorkflowStatusEvent

        self._widget_proposal = None
        kind = shown[2].kind
        if kind == DETECTION and self._panel.det_widget is not None:
            self._panel.det_widget.clear_layers()
        elif kind == ALIGNMENT_AREA and self._panel.image_widget is not None:
            self._panel.image_widget.clear_alignment_area()
        elif kind == POINT_OF_INTEREST:
            self.poi_marker.remove()
        self._panel.prompt_bar.clear()
        self._relay.status.emit(WorkflowStatusEvent(message=""))

    def _widget_value(self, proposal) -> Dict[str, Any]:
        """What the widget holds for ``proposal``'s kind; empty (as proposed)
        when the widget was not there to edit it on."""
        from fibsem.detection.decided import detection_values
        from fibsem.workflows.proposals import (
            ALIGNMENT_AREA,
            DETECTION,
            POINT_OF_INTEREST,
        )

        if proposal.kind == DETECTION and self._detection_shown():
            return detection_values(self._panel.det_widget._get_detected_features())
        if proposal.kind == ALIGNMENT_AREA and self._panel.image_widget is not None:
            area = self._panel.image_widget.get_alignment_area()
            return {"alignment_area": deepcopy(area)} if area is not None else {}
        if proposal.kind == POINT_OF_INTEREST and self.poi_marker.shown:
            poi = self.poi_marker.read()
            return {"poi": poi} if poi is not None else {}
        return {}

    def _decide_widget_proposal(self) -> None:
        shown = self._widget_proposal
        experiment = self.experiment
        if shown is None or experiment is None:
            return
        item_id, task_name, proposal = shown
        from fibsem.workflows.proposals import Decision, DecisionOutcome

        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor=experiment.actor(),
                values=self._widget_value(proposal),
                via="workflow",
                proposal_id=proposal.id,
            ),
        )
        if not result.applied:
            # Left up: the question is still open, and the reason is logged.
            logging.warning(f"{task_name}: could not decide it: {result.reason}")
            self._panel.prompt_bar.reopen()

    def front_proposal(self) -> None:
        """Bring forward the tab the proposal now up is decided on, if it has
        one. The attention button is how the operator gets back to it. Does
        nothing when no proposal is up, or it is on the prompt bar."""
        widget = self.proposal_host()
        if widget is not None:
            self._panel.front_tab(widget)

    def proposal_host(self):
        """The tab the proposal now up is decided on, when it has one of its
        own (a mill's pass, a detection); None otherwise."""
        return self.loop_proposal_host() or self.widget_proposal_host()

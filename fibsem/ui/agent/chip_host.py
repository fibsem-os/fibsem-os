"""The agent on a window's status bar: its level chip and card, what it is
doing on the microscope, and the previews of what its waiting request would do.

The window builds one and tells it the session's agent; the chip's clicks are
handled here.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ui import notification_service
from fibsem.ui.agent.activity import AgentActivity
from fibsem.ui.agent.level_chip import AgentLevelChip, request_outcome
from fibsem.ui.agent.previews import AgentPreviews

# How often a waiting request the operator is looking at is read again.
AGENT_REQUEST_REFRESH_S = 15.0


class AgentChipHost(QObject):
    """The chip, shown whenever the App Server runs, at the level its auth
    holds. Polled a second rather than signalled: the server starts and stops
    with the microscope connection, from the other window."""

    # The agent's request changed: emitted from the request's
    # worker thread, delivered on the GUI thread.
    request_changed = pyqtSignal()

    def __init__(
        self,
        status_bar,
        *,
        session: Callable[[], Any],
        view_controller: Callable[[], Any],
        sources: Any,
        on_driving: Callable[[bool], None],
        parent: Optional[QObject] = None,
    ) -> None:
        """*sources* reads what the previews draw (``PreviewSources``);
        *on_driving* paints the agent's purple while it drives."""
        super().__init__(parent)
        self._session = session
        self._on_driving = on_driving

        self.chip = AgentLevelChip(
            last_heard=lambda: self._agent.seconds_since_seen(),
            parent=status_bar,
            agent_client=lambda: self._agent.client(),
        )
        self.chip.hide()
        self.chip.level_chosen.connect(self._on_level_chosen)
        self.chip.connect_clicked.connect(self._on_connect_agent)
        self.chip.event_log_clicked.connect(self._on_open_event_log)
        self.chip.chat_clicked.connect(self._on_open_chat)
        self.chip.stop_clicked.connect(self._on_stop)
        self.chip.approve_clicked.connect(self._on_approve)
        self.chip.reject_clicked.connect(self._on_reject)
        self.chip.message_sent.connect(self._on_operator_message)
        self.chip.suggestion_approved.connect(self._on_approve_suggestion)
        self.chip.suggestion_declined.connect(self._on_decline_suggestion)
        status_bar.add_action(self.chip)
        self.request_changed.connect(self.refresh)

        self.activity = AgentActivity(
            status_bar, view_controller=view_controller, parent=self
        )
        self.activity.now_changed.connect(self.chip.set_now)
        self.activity.recent_changed.connect(self.chip.set_recent)

        self.previews = AgentPreviews(
            view_controller,
            reference_image=sources.proposal_image,
            task_config=sources.task_config,
            show_picture=self._show_picture,
            item_pose=sources.item_pose,
            item_image=sources.item_image,
            overview_image=sources.overview_image,
            item_overview=sources.item_overview,
            pending_proposal=sources.pending_proposal,
        )
        # A waiting request's lines, read again while the operator looks at it.
        self._request_described: Tuple[Optional[str], float] = (None, 0.0)
        self._experiment = None
        # The pending proposal the card's suggestion is on: (item id, task, id).
        self._suggested: Optional[Tuple[str, str, str]] = None

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

    @property
    def _agent(self):
        """The agent as the operator meets it (``fibsem.ai.agent.session``)."""
        return self._session().agent

    def refresh(self) -> None:
        """Show the chip while the App Server runs, at the level its auth holds."""
        recorder = self._session().recorder
        self.activity.follow_events(getattr(recorder, "buffer", None))
        agent = self._agent
        level = agent.level
        self._follow_requests()
        if level is None:
            self.chip.set_request(None)
            self.previews.show_request(None)
            self.chip.hide()
            self._on_driving(False)
            return
        if self.chip.level is not level:
            self.chip.set_level(level)
        self._on_driving(level is AgentLevel.DRIVE)
        self._show_request()
        self.chip.show()

    def listen(self, experiment) -> None:
        """Show the agent's suggestions on *experiment*'s pending proposals;
        the previous experiment's are dropped."""
        previous = self._experiment
        if previous is experiment:
            return
        if previous is not None:
            try:
                previous.suggested.disconnect(self._on_suggestion_changed)
                previous.decided.disconnect(self._on_suggestion_changed)
            except Exception:
                pass
        self._experiment = experiment
        if experiment is not None:
            experiment.suggested.connect(self._on_suggestion_changed)
            experiment.decided.connect(self._on_suggestion_changed)
        self._show_suggestion()

    def _on_suggestion_changed(self, item_id: str, task_name: str) -> None:
        self._show_suggestion()

    def _show_suggestion(self) -> None:
        """The card shows the agent's suggestion on a pending proposal, until
        it is approved, declined or the proposal is decided otherwise."""
        from fibsem.ai.agent.suggestions import (
            front_suggestion,
            suggestion_card,
            suggestion_request,
        )

        experiment = self._experiment
        front = front_suggestion(experiment) if experiment is not None else None
        if front is None:
            self._suggested = None
            self.chip.set_suggestion(None)
            self.previews.show_suggestion(None)
            return
        item, task_name, proposal = front
        self._suggested = (item.id, task_name, proposal.id)
        self.chip.set_suggestion(
            suggestion_card(item, task_name, proposal, str(experiment.path))
        )
        self.previews.show_suggestion(
            suggestion_request(item, task_name, proposal, str(experiment.path))
        )

    def _show_picture(
        self, image, caption: str = "", suggestion: bool = False, legend=()
    ) -> None:
        """The card's picture, and the same picture for the chat page."""
        from fibsem.ui.agent.card import picture_png

        self.chip.card.set_image(image, caption, suggestion=suggestion, legend=legend)
        drawn = image is not None and not image.isNull()
        self._agent.set_picture(
            "suggestion" if suggestion else "request",
            picture_png(image, legend) if drawn else None,
            caption,
        )

    def _on_approve_suggestion(self) -> None:
        """Approve on the card: the proposal is decided as suggested, as the
        operator's decision, naming the suggestion."""
        if self._suggested is None:
            return
        result = self._agent.approve_suggestion(self._suggested[2])
        if not result.applied:
            notification_service.show(
                f"Could not approve the agent's suggestion: {result.reason}", "info"
            )
        self._show_suggestion()

    def _on_decline_suggestion(self, reason: str) -> None:
        """Decline on the card: the suggestion is dropped, with the reason
        for the agent; the proposal stays pending."""
        if self._suggested is None:
            return
        self._agent.decline_suggestion(self._suggested[2], reason)
        self._show_suggestion()

    def _follow_requests(self) -> None:
        """Move the chip as a request is filed and applied rather than on the
        next poll."""

        def changed(signal=self.request_changed):
            try:
                signal.emit()
            except RuntimeError:  # the window has gone
                pass

        self._agent.follow(changed)

    def _show_request(self) -> None:
        """The chip shows a request waiting or being applied; how one ended is
        said once, as a toast, so the status bar keeps what is running now."""
        agent = self._agent
        for request in agent.outcomes():
            outcome = request_outcome(request.to_dict())
            if outcome is not None:
                message, kind = outcome
                # Kept in the bell's history: the toast goes, the outcome stays.
                notification_service.show(
                    message,
                    kind,
                    action=("Show in event log", self._on_open_event_log)
                    if kind == "error"
                    else None,
                )
        current = agent.current()
        if current is not None and current.status == "pending":
            self._describe_request_again(agent.requests, current.id)
        shown = current.to_dict() if current is not None else None
        if shown != self.chip.request:
            self.chip.set_request(shown)
        self.previews.show_request(shown)
        self.chip.set_now(self.activity.now)

    def _describe_request_again(self, store, request_id: str) -> None:
        """While the card shows a waiting request, its values now are read
        again: on opening the card, and every
        `AGENT_REQUEST_REFRESH_S` while it stays open. Off the GUI thread: it
        reads the routes the request describes."""
        if not (self.chip.popup.isVisible() or self.chip.floating):
            self._request_described = (None, 0.0)
            return
        last_id, last_at = self._request_described
        now = time.monotonic()
        if last_id == request_id and now - last_at < AGENT_REQUEST_REFRESH_S:
            return
        self._request_described = (request_id, now)
        from fibsem.ui.qt.threading import FunctionWorker

        FunctionWorker(store.refresh, request_id).start()

    def _on_connect_agent(self) -> None:
        """From the card: the agent token and how to register the sidecar."""
        from fibsem.ui.agent.connect_agent_dialog import ConnectAgentDialog

        dialog = ConnectAgentDialog(
            lambda: self._session().app_server, self.chip.window()
        )
        dialog.exec_()

    def _on_open_event_log(self) -> None:
        """From the card: the App Server's event log in the browser."""
        from fibsem.server.hosting import event_log_url
        from fibsem.ui.agent.app_server_dialog import open_in_browser

        host = self._agent.host
        if host is None:
            return
        open_in_browser(event_log_url(host))

    def _on_open_chat(self) -> None:
        """From the card: the chat with the agent in the browser."""
        from fibsem.server.hosting import chat_url
        from fibsem.ui.agent.app_server_dialog import open_in_browser

        host = self._agent.host
        if host is None:
            return
        open_in_browser(chat_url(host))

    def _on_approve(self, request_id: str) -> None:
        """Run the request. Its actions run on their own worker thread, so this
        returns at once."""
        self._decide(request_id, approve=True)

    def _on_reject(self, request_id: str, reason: str) -> None:
        """Run none of it; the agent gets the reason."""
        self._decide(request_id, approve=False, reason=reason)

    def _decide(self, request_id: str, approve: bool, reason: str = "") -> None:
        from fibsem.server.requests import RequestRefused

        try:
            if approve:
                self._agent.approve(request_id)
            else:
                self._agent.reject(request_id, reason=reason)
        except RequestRefused as refused:
            notification_service.show_toast(f"Agent's request: {refused}", "error")
        self.refresh()

    def _on_level_chosen(self, level: AgentLevel) -> None:
        """Set the agent's level, effective for its next request."""
        if self._agent.host is not None:
            refused = self._agent.set_level(level)
            if refused:
                notification_service.show_toast(refused, "error")
        self.refresh()

    def _on_stop(self) -> None:
        """The brake (``AgentSession.stop``)."""
        self._agent.stop()
        self.refresh()

    def _on_operator_message(self, text: str) -> None:
        """The operator's message to the agent (``AgentSession.message``)."""
        if self._agent.message(text) is None:
            notification_service.show_toast(
                "No experiment is recording: the agent can't hear messages", "warning"
            )

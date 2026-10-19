"""The agent on a window's status bar (FIB-1035): its level chip and card,
what it is doing on the microscope (FIB-1375), and the previews of what its
waiting request or suggestion would do (FIB-1404).

The window builds one and tells it the session's agent, the question now up
and where the operator answers it; the chip's clicks are handled here.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fibsem.acting import OPERATOR
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

    # The agent's request changed (FIB-1283): emitted from the request's
    # worker thread, delivered on the GUI thread.
    request_changed = pyqtSignal()

    def __init__(
        self,
        status_bar,
        *,
        session: Callable[[], Any],
        view_controller: Callable[[], Any],
        sources: Any,
        responder: Callable[[], Any],
        take_suggestion: Callable[[], bool],
        dismiss_suggestion: Callable[[str], Any],
        on_driving: Callable[[bool], None],
        parent: Optional[QObject] = None,
    ) -> None:
        """*sources* reads what the previews draw (``PreviewSources``);
        *responder* is the one putting the question up; *on_driving* paints
        the agent's purple while it drives."""
        super().__init__(parent)
        self._session = session
        self._responder = responder
        self._take_suggestion = take_suggestion
        self._dismiss_suggestion = dismiss_suggestion
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
        self.chip.stop_clicked.connect(self._on_stop)
        self.chip.approve_clicked.connect(self._on_approve)
        self.chip.reject_clicked.connect(self._on_reject)
        self.chip.accept_clicked.connect(self._on_accept_suggestion)
        self.chip.dismiss_clicked.connect(self._on_dismiss_suggestion)
        self.chip.message_sent.connect(self._on_operator_message)
        status_bar.add_action(self.chip)
        self.request_changed.connect(self.refresh)

        self.activity = AgentActivity(
            status_bar, view_controller=view_controller, parent=self
        )
        self.activity.now_changed.connect(self.chip.set_now)
        self.activity.recent_changed.connect(self.chip.set_recent)

        self.previews = AgentPreviews(
            view_controller,
            reference_image=sources.review_image,
            task_config=sources.task_config,
            show_picture=self.chip.card.set_image,
            item_pose=sources.item_pose,
            item_image=sources.item_image,
            overview_image=sources.overview_image,
            item_overview=sources.item_overview,
            review_proposal=sources.review_proposal,
        )
        # The suggestion the operator dropped by answering otherwise, until the
        # answer that dropped it is heard (the responder takes the suggestion
        # down before it says the question was answered).
        self._dropped_suggestion: Optional[dict] = None
        # A waiting request's lines, read again while the operator looks at it.
        self._request_described: Tuple[Optional[str], float] = (None, 0.0)

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
        again (FIB-1412): on opening the card, and every
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
        from fibsem.ui.agent.app_server_dialog import AppServerDialog, open_in_browser

        host = self._agent.host
        if host is None:
            return
        open_in_browser(AppServerDialog.event_log_url(host))

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

    def on_question_event(self, kind: str, payload: dict, actor: str) -> None:
        """A question was raised, answered or cancelled: say so when the
        operator's own answer dropped the agent's suggestion."""
        dropped, self._dropped_suggestion = self._dropped_suggestion, None
        if (
            dropped is not None
            and kind == "prompt_answered"
            and actor == OPERATOR
            and not payload.get("suggested_by")
        ):
            notification_service.show(
                "Agent's suggestion dropped: you answered the question yourself",
                "info",
            )

    def on_suggestion(self, suggestion: Optional[dict]) -> None:
        """The agent's suggested answer went up or came down. It shows on the
        agent's card, not on the prompt (FIB-1407)."""
        shown = self.chip.suggestion
        self._dropped_suggestion = shown if suggestion is None else None
        self.chip.set_suggestion(suggestion)
        self.previews.show_suggested_point(
            suggestion.get("point") if suggestion is not None else None
        )
        responder = self._responder()
        standing = responder.suggestion if suggestion is not None else None
        self.previews.show_suggestion(
            responder.pending_question() if standing else None,
            standing[2] if standing else None,
            image=standing[4] if standing else None,
        )

    def _on_accept_suggestion(self) -> None:
        """The operator answers with the agent's suggestion, as theirs."""
        self._dropped_suggestion = None
        if not self._take_suggestion():
            notification_service.show(
                "The agent's suggestion had gone: the question was answered or "
                "taken down",
                "info",
            )

    def _on_dismiss_suggestion(self, reason: str) -> None:
        """The operator declines the suggestion, recorded with their reason for
        the agent; the question stays up."""
        self._dismiss_suggestion(reason)
        # Declined, not dropped: no "you answered it yourself" later.
        self._dropped_suggestion = None

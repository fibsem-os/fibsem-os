"""The agent on a window's status bar: its level chip and card, and what it is
doing on the microscope.

The window builds one and tells it the session's agent; the chip's clicks are
handled here. The operator answers the agent on the web Agent page.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ui import notification_service
from fibsem.ui.agent.activity import AgentActivity
from fibsem.ui.agent.level_chip import AgentLevelChip


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
        on_driving: Callable[[bool], None],
        parent: Optional[QObject] = None,
    ) -> None:
        """*on_driving* paints the agent's purple while it drives."""
        super().__init__(parent)
        self._session = session
        self._on_driving = on_driving

        self.chip = AgentLevelChip(
            last_heard=lambda: self._agent.seconds_since_seen(),
            parent=status_bar,
            agent_client=lambda: self._agent.client(),
            agent_session=lambda: self._agent.session_text(),
            agent_budget_low=lambda: self._agent.budget_low(),
        )
        self.chip.hide()
        self.chip.level_chosen.connect(self._on_level_chosen)
        self.chip.event_log_clicked.connect(self._on_open_event_log)
        self.chip.chat_clicked.connect(self._on_open_chat)
        self.chip.stop_clicked.connect(self._on_stop)
        status_bar.add_action(self.chip)
        self.request_changed.connect(self.refresh)

        self.activity = AgentActivity(
            status_bar, view_controller=view_controller, parent=self
        )
        self.activity.now_changed.connect(self.chip.set_now)

        self._experiment = None
        # The asks to connect followed, and how to stop following them.
        self._connections = None
        self._unfollow_connections: Optional[Callable[[], None]] = None

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
        agent = self._agent
        level = agent.level
        self._follow_requests()
        self._follow_connections()
        if level is None:
            self.chip.set_request(None)
            self.chip.set_asking(None)
            self.chip.hide()
            self._on_driving(False)
            return
        if self.chip.level is not level:
            self.chip.set_level(level)
        self._on_driving(level is AgentLevel.DRIVE)
        self._show_request()
        self.chip.set_asking(agent.asking())
        self.chip.show()

    def listen(self, experiment) -> None:
        """Show that the agent suggests a decision on *experiment*'s pending
        proposals; the previous experiment's are dropped."""
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
        """The chip shows the agent's suggestion on a pending proposal, until
        it is approved, declined or the proposal is decided otherwise."""
        from fibsem.ai.agent.suggestions import front_suggestion, suggestion_card

        experiment = self._experiment
        front = front_suggestion(experiment) if experiment is not None else None
        if front is None:
            self.chip.set_suggestion(None)
            return
        item, task_name, proposal = front
        self.chip.set_suggestion(
            suggestion_card(item, task_name, proposal, str(experiment.path))
        )

    def _follow_requests(self) -> None:
        """Move the chip as a request is filed and applied rather than on the
        next poll."""

        def changed(signal=self.request_changed):
            try:
                signal.emit()
            except RuntimeError:  # the window has gone
                pass

        self._agent.follow(changed)

    def _follow_connections(self) -> None:
        """Show an agent asking to connect as it asks, not on the next poll."""
        connections = self._agent.connections
        if connections is self._connections:
            return
        if self._unfollow_connections is not None:
            self._unfollow_connections()
            self._unfollow_connections = None
        self._connections = connections
        if connections is None:
            return

        def changed(signal=self.request_changed):
            try:
                signal.emit()
            except RuntimeError:  # the window has gone
                pass

        self._unfollow_connections = connections.on_change(changed)

    def _show_request(self) -> None:
        """The chip shows a request waiting or being applied."""
        agent = self._agent
        current = agent.current()
        shown = current.to_dict() if current is not None else None
        if shown != self.chip.request:
            self.chip.set_request(shown)
        self.chip.set_now(self.activity.now)

    def _on_open_event_log(self) -> None:
        """From the card: the App Server's event log in the browser."""
        from fibsem.server.hosting import event_log_url
        from fibsem.ui.agent.app_server_dialog import open_in_browser

        host = self._agent.host
        if host is None:
            return
        open_in_browser(event_log_url(host))

    def _on_open_chat(self) -> None:
        """From the card: the web Agent page in the browser."""
        from fibsem.server.hosting import chat_url
        from fibsem.ui.agent.app_server_dialog import open_in_browser

        host = self._agent.host
        if host is None:
            return
        open_in_browser(chat_url(host))

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

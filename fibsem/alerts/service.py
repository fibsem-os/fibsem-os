"""Telling someone away from the instrument, on Slack.

One per session, started by the host: it reads the session's record and what
waits in it every few seconds (`AlertRules`), and posts what is due. Settings
are read again on every pass, so a change in Settings counts at once; the token
is read from ``FIBSEM_SLACK_TOKEN`` when it starts.

Off unless set up: nothing is sent while notifications are off, no token is
set, no channel is named, or the settings can't be read. What happened
meanwhile is not sent later.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from fibsem.alerts.rules import (
    AGENT_MESSAGE,
    ANSWER,
    FAILED,
    FINISHED,
    Alert,
    AlertRules,
)
from fibsem.alerts.slack import SlackChannel
from fibsem.config import NotificationPreferences, load_user_preferences

TOKEN_ENV = "FIBSEM_SLACK_TOKEN"
POLL_S = 3.0

# Which setting turns each kind on.
_SETTING = {
    ANSWER: "answers",
    AGENT_MESSAGE: "messages",
    FAILED: "failures",
    FINISHED: "run_finished",
}


def slack_token() -> str:
    return os.environ.get(TOKEN_ENV, "").strip()


def notification_settings() -> NotificationPreferences:
    return load_user_preferences().install.notifications


def wanted(settings: NotificationPreferences, alert: Alert) -> bool:
    return bool(getattr(settings, _SETTING.get(alert.kind, ""), False))


def _channel_name(settings: NotificationPreferences) -> str:
    return (settings.channel_id or settings.channel).strip()


def send_test(
    settings: NotificationPreferences,
    token: Optional[str] = None,
    channel: Optional[Callable[..., Any]] = None,
) -> str:
    """Post a test message to the channel the settings name, whether or not
    notifications are on. Returns the channel's ID; raises ``ValueError`` when
    there is no token or channel, and ``SlackError`` when Slack refuses."""
    token = slack_token() if token is None else token
    if not token:
        raise ValueError(
            f"No Slack token: set {TOKEN_ENV} on this PC and start fibsemOS again."
        )
    name = _channel_name(settings)
    if not name:
        raise ValueError("No Slack channel: name one first.")
    poster = (channel or SlackChannel)(token, name, settings.link)
    try:
        return poster.send(
            Alert(
                "test",
                "test",
                "Test from fibsemOS",
                ["Notifications from this instrument come here."],
            )
        )
    finally:
        poster.close()


class Alerts:
    """Reads the session and posts what is due; ``tick`` is one pass."""

    def __init__(
        self,
        session: Any,
        settings: Callable[[], NotificationPreferences] = notification_settings,
        token: Optional[str] = None,
        channel: Callable[..., Any] = SlackChannel,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], float] = time.time,
        poll_s: float = POLL_S,
    ):
        self._session = session
        self._settings = settings
        self._token = slack_token() if token is None else token
        self._make_channel = channel
        self._rules = AlertRules(session, clock=clock)
        self._now = now
        self._poll_s = poll_s
        # The last record read from each stream (the idle one and each
        # microscope's), so a reconnect reads on from where it was.
        self._seen: Dict[str, int] = {}
        # What happened before this started is history, not news: the stream
        # there is now is read as history here, and a stream first seen
        # later only from this moment on.
        self._started = now()
        self._read(news=False)
        self._poster: Optional[Any] = None
        self._poster_for: Optional[tuple] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --- running -------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        if not self._token:
            logging.info("Slack notifications: no %s, so none are sent", TOKEN_ENV)
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="fibsem-alerts", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._poster is not None:
            self._poster.close()
            self._poster = None

    def _run(self) -> None:
        while not self._stop.wait(self._poll_s):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - one bad pass must not end the next
                logging.warning("Slack notifications: a pass failed", exc_info=True)

    # --- one pass ------------------------------------------------------------

    def _read(self, news: bool = True) -> None:
        buffer = self._session.event_buffer
        stream = buffer.stream
        known = stream in self._seen
        page = buffer.events_since(self._seen.get(stream, 0))
        for record in page["events"]:
            new = news and (known or record.get("timestamp", 0) >= self._started)
            self._rules.feed(record, news=new)
        self._seen[stream] = page["latest_seq"]

    def tick(self) -> List[Alert]:
        """Read what is new, and send what is due; returns what was sent."""
        self._read()
        try:
            asks = self._rules.asks()
        except Exception:  # noqa: BLE001 - keep what was last seen
            logging.debug(
                "Slack notifications: could not read what waits", exc_info=True
            )
            asks = None
        try:
            settings = self._settings()
        except Exception:  # noqa: BLE001 - unreadable settings send nothing
            logging.warning(
                "Slack notifications: could not read the settings", exc_info=True
            )
            settings = None
        wait_s = max(0.0, float(settings.wait_minutes)) * 60 if settings else 0.0
        due = self._rules.due(wait_s, asks)
        if settings is None or not settings.enabled or not self._token:
            return []
        poster = self._poster_of(settings)
        if poster is None:
            return []
        sent = []
        for alert in due:
            if wanted(settings, alert) and self._send(poster, alert, settings.images):
                sent.append(alert)
        return sent

    def _poster_of(self, settings: NotificationPreferences) -> Optional[Any]:
        name = _channel_name(settings)
        if not name:
            return None
        key = (name, settings.link)
        if self._poster_for != key:
            if self._poster is not None:
                self._poster.close()
            self._poster = self._make_channel(self._token, name, settings.link)
            self._poster_for = key
        return self._poster

    def _send(self, poster: Any, alert: Alert, images: bool) -> bool:
        image = None
        if images and alert.image is not None:
            try:
                image = alert.image()
            except Exception:  # noqa: BLE001 - send it without the picture
                logging.debug(
                    "Slack notifications: no picture for %s", alert.key, exc_info=True
                )
        try:
            poster.send(alert, image)
            return True
        except Exception as e:  # noqa: BLE001 - the next one may go
            # The message names Slack's error or the network's, never the token.
            logging.warning(
                "Slack notifications: could not send %r: %s",
                alert.headline,
                type(e).__name__ + f": {e}",
            )
            return False

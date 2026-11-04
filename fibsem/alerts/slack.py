"""Posting an alert to a Slack channel, as the fibsemOS bot.

A bot token (``chat:write``, ``files:write``) can post and upload only in the
channels the bot was added to, and read nothing. It is sent in a header and
never logged.

An image is uploaded privately and the message points at it with an image
block. Slack refuses the block until it has processed the file, so the post is
tried a few times, then sent without the image and the file shared under it.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import httpx

from fibsem.alerts.rules import Alert

API = "https://slack.com/api/"
TIMEOUT_S = 15.0
# Slack refuses a section longer than this.
SECTION_LIMIT = 3000
# Tries of a post with an image block before it goes without one.
IMAGE_TRIES = 4


class SlackError(RuntimeError):
    """Slack answered ``ok: false``; ``error`` is its code, such as
    ``channel_not_found`` or ``not_in_channel``."""

    def __init__(self, method: str, error: str):
        super().__init__(f"Slack {method}: {error}")
        self.error = error


def escape(text: str) -> str:
    """``text`` with the three characters Slack's mrkdwn reads as markup
    escaped, so an agent's words can't make a link or a mention."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _cut(text: str, limit: int = SECTION_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def blocks(
    alert: Alert, link: str = "", file_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """The message: headline, its lines, the picture, where it happened, and a
    button to the page it is answered on."""
    out: List[Dict[str, Any]] = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": _cut(f"*{escape(alert.headline)}*")},
        }
    ]
    if alert.lines:
        text = "\n".join(escape(line) for line in alert.lines)
        out.append({"type": "section", "text": {"type": "mrkdwn", "text": _cut(text)}})
    if file_id:
        out.append(
            {
                "type": "image",
                "slack_file": {"id": file_id},
                "alt_text": _cut(alert.headline, 2000),
            }
        )
    if alert.context:
        out.append(
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": _cut(escape(alert.context))}],
            }
        )
    if link:
        url = f"{link.rstrip('/')}/{alert.page}"
        out.append(
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Open"},
                        "url": url,
                    }
                ],
            }
        )
    return out


class SlackChannel:
    """Posts alerts to one channel, named (``#microscope-alerts``) or by its
    ID. ``send`` returns the channel's ID as Slack answers it."""

    def __init__(
        self,
        token: str,
        channel: str,
        link: str = "",
        client: Optional[httpx.Client] = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._token = token
        self._channel = channel.strip()
        self._link = link
        self._client = client or httpx.Client(timeout=TIMEOUT_S)
        self._sleep = sleep

    def _call(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        response = self._client.post(
            API + method, headers={"Authorization": f"Bearer {self._token}"}, **kwargs
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("ok"):
            raise SlackError(method, str(body.get("error") or "unknown"))
        return body

    def _upload(
        self, data: bytes, name: str, channel: str = "", thread: str = ""
    ) -> str:
        ticket = self._call(
            "files.getUploadURLExternal", data={"filename": name, "length": len(data)}
        )
        # The upload URL is Slack's own, signed; it takes no token.
        self._client.post(ticket["upload_url"], content=data).raise_for_status()
        complete: Dict[str, Any] = {
            "files": json.dumps([{"id": ticket["file_id"], "title": name}])
        }
        if channel:
            complete["channel_id"] = channel
        if thread:
            complete["thread_ts"] = thread
        self._call("files.completeUploadExternal", data=complete)
        return ticket["file_id"]

    def _post(self, alert: Alert, file_id: Optional[str] = None) -> Dict[str, Any]:
        return self._call(
            "chat.postMessage",
            json={
                "channel": self._channel,
                # What a phone's notification and a screen reader show.
                "text": alert.headline,
                "blocks": blocks(alert, self._link, file_id),
                "unfurl_links": False,
            },
        )

    def send(self, alert: Alert, image: Optional[bytes] = None) -> str:
        if image:
            file_id = self._upload(image, f"{alert.kind}.png")
            for attempt in range(IMAGE_TRIES):
                try:
                    return str(self._post(alert, file_id)["channel"])
                except SlackError as e:
                    if e.error != "invalid_blocks":
                        raise
                    self._sleep(2.0 * (attempt + 1))  # still processing the file
            logging.info(
                "Slack kept refusing the image block; sending the image under the message"
            )
            posted = self._post(alert)
            channel = str(posted["channel"])
            self._upload(
                image,
                f"{alert.kind}.png",
                channel=channel,
                thread=str(posted.get("ts") or ""),
            )
            return channel
        return str(self._post(alert)["channel"])

    def close(self) -> None:
        self._client.close()

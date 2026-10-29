"""The operator's routes for Slack notifications (``fibsem.alerts``): read and
change this instrument's settings, and send a test.

Behind the operator token only: the agent can't turn notifications on, change
where they go, or post. The bot token is never served or taken here; it is the
``FIBSEM_SLACK_TOKEN`` environment variable on this PC.
"""

import dataclasses
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Body, Depends, Request

from fibsem.alerts import send_test
from fibsem.alerts.service import TOKEN_ENV, slack_token
from fibsem.alerts.slack import SlackError
from fibsem.config import (
    NotificationPreferences,
    load_user_preferences,
    update_user_preferences,
)
from fibsem.server.auth import refusal_error, require_operator
from fibsem.server.responses import NotificationSettings, NotificationTest, answer

# Each setting's type by name; a string under postponed annotations.
_FIELDS = {
    f.name: getattr(f.type, "__name__", f.type)
    for f in dataclasses.fields(NotificationPreferences)
}
_TYPES = {"bool": bool, "str": str, "float": (int, float)}


def _view() -> NotificationSettings:
    return NotificationSettings(
        settings=load_user_preferences().install.notifications,
        token_set=bool(slack_token()),
        token_from=TOKEN_ENV,
    )


def _checked(body: Dict[str, Any]) -> Dict[str, Any]:
    """The fields of *body*, each the type its setting is; refuses anything else."""
    changes = {}
    for key, value in body.items():
        kind = _TYPES.get(_FIELDS.get(key, ""))
        if kind is None:
            raise refusal_error(422, "unknown_field", f"No setting {key!r}.")
        if not isinstance(value, kind) or (
            kind is not bool and isinstance(value, bool)
        ):
            raise refusal_error(422, "wrong_type", f"{key!r} is a {_FIELDS[key]}.")
        changes[key] = value.strip() if isinstance(value, str) else value
    if "wait_minutes" in changes and not 0 <= changes["wait_minutes"] <= 24 * 60:
        raise refusal_error(422, "out_of_range", "Wait between 0 and 1440 minutes.")
    # A channel named anew is found again by the next test.
    if "channel" in changes and "channel_id" not in changes:
        changes["channel_id"] = ""
    return changes


def build_notifications_router() -> APIRouter:
    """Under ``/app/notifications``, the operator's only."""
    router = APIRouter(
        prefix="/app/notifications", dependencies=[Depends(require_operator())]
    )

    @router.get("", response_model=NotificationSettings)
    def get_notifications():
        """This instrument's settings, and whether the token is set."""
        return answer(_view())

    @router.put("", response_model=NotificationSettings)
    def put_notifications(body: Dict[str, Any] = Body(...)):
        """Change the settings named in the body; the rest stay. Counted on
        the next pass, with no restart."""
        changes = _checked(body)

        def change(preferences):
            for key, value in changes.items():
                setattr(preferences.install.notifications, key, value)

        update_user_preferences(change)
        return answer(_view())

    @router.post("/test", response_model=NotificationTest)
    def test_notifications(request: Request):
        """Post a test message to the channel, on or off, and keep the ID
        Slack answers for it, so a renamed channel is still found."""
        settings = load_user_preferences().install.notifications
        try:
            channel_id = send_test(settings, link=f"{request.base_url}ui")
        except ValueError as e:
            raise refusal_error(422, "not_set_up", str(e))
        except SlackError as e:
            raise refusal_error(
                502, "slack_refused", f"Slack refused: {e.error}.", slack_error=e.error
            )
        except httpx.HTTPError as e:
            raise refusal_error(
                502, "slack_unreachable", f"Couldn't reach Slack ({type(e).__name__})."
            )

        def keep(preferences):
            preferences.install.notifications.channel_id = channel_id

        update_user_preferences(keep)
        return answer(NotificationTest(channel_id=channel_id))

    return router

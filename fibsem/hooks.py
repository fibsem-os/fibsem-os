"""Hook system for task lifecycle events.

Application-agnostic on purpose: this module knows nothing about AutoLamella, Qt or the
microscope, so any application can fire the same events. AutoLamella is the only producer
today (see AutoLamellaTask._fire_hook and TaskManager).
"""

import logging
import threading
import time
import warnings
import yaml
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type


class HookEvent(str, Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"
    TASK_SKIPPED = "task_skipped"
    # Ordered by widening scope: one task, then the item all its tasks were for, then
    # the run, then the whole experiment. "item" rather than "lamella" to match
    # HookContext.item_name — the application decides what an item is.
    ITEM_COMPLETED = "item_completed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    EXPERIMENT_COMPLETED = "experiment_completed"


@dataclass
class HookContext:
    event: str
    task_name: str = ""
    task_type: str = ""
    # Name of the item the task ran against. Today that is always a lamella, but the
    # hook contract stays domain-agnostic so grids (or anything else) can fire hooks.
    item_name: str = ""
    # Stable keys beside the display names. Names are what a person reads and can
    # change; these are what a consumer stores and joins on. task_id identifies this
    # *execution*, so it matches the entry the run leaves in the task history.
    item_id: str = ""
    task_id: str = ""
    # Which run this belongs to. Without it, a consumer handling more than one
    # experiment cannot tell their events apart.
    experiment_id: Optional[str] = None
    experiment_name: str = ""
    # How much of the run is left. A failure does not stop a run — the queue moves to
    # the next item — so without this a bare "FAILED" reads as a dead workflow.
    tasks_remaining: Optional[int] = None
    tasks_total: Optional[int] = None
    # Application-owned state for the run. AutoLamella passes an AutoLamellaTaskState;
    # left untyped so this module does not depend on any one application's structures.
    task_state: Optional[Any] = None
    # The exception message, not a traceback: this payload can leave the machine via a
    # webhook, and a traceback is unbounded. The logfile has the full one.
    error: Optional[str] = None
    # Why a task was skipped, on TASK_SKIPPED. The producer decides the vocabulary;
    # AutoLamella uses not_required / failure / missing_prereqs / lamella_not_found.
    skip_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Normalize HookEvent enums to plain strings so f-strings and
        # comparisons work consistently regardless of Python version.
        if hasattr(self.event, "value"):
            self.event = self.event.value
        self._snapshot_task_state()

    def _snapshot_task_state(self) -> None:
        """Copy task_state, so a hook that defers work reads what happened here.

        Producers pass live state. AutoLamella's is a single object reused for every
        run, so a hook that touches it asynchronously — a webhook posting from its
        daemon thread, a callback queued onto another thread — would otherwise read
        whatever the *next* task has since written into it. WebhookHook is safe today
        only because it happens to read plain strings.

        Guarded, and deliberately so: this runs in the producer's call stack, before
        HookManager.fire's try/except exists to contain anything. An application object
        that cannot be deepcopied must not take the workflow down with it, so the copy
        degrades to the live reference and says so.
        """
        if self.task_state is None:
            return
        try:
            self.task_state = deepcopy(self.task_state)
        except Exception:
            logging.exception(
                "Could not snapshot task_state for the %s hook; passing the live "
                "object, which a deferred hook may see mutated", self.event,
            )

    @property
    def lamella_name(self) -> str:
        """Deprecated alias for item_name, kept so hooks written against the pre-rename
        context keep working. Read-only: construct with item_name."""
        warnings.warn(
            "HookContext.lamella_name is deprecated, use HookContext.item_name",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.item_name


DEFAULT_MESSAGE_TEMPLATE = "Task {task_name} {event} for {item_name}"


def _blank_if_none(value: Optional[int]) -> str:
    """Render an absent count as empty rather than the string "None"."""
    return "" if value is None else str(value)


def _template_fields(context: HookContext) -> Dict[str, str]:
    """Placeholder values available to a hook message template."""
    return {
        "event": context.event,
        "task_name": context.task_name,
        "task_type": context.task_type,
        "item_name": context.item_name,
        "lamella_name": context.item_name,  # deprecated alias, renders pre-rename templates
        "item_id": context.item_id,
        "task_id": context.task_id,
        "experiment_id": context.experiment_id or "",
        "experiment_name": context.experiment_name,
        "tasks_remaining": _blank_if_none(context.tasks_remaining),
        "tasks_total": _blank_if_none(context.tasks_total),
        "error": context.error or "",
        "skip_reason": context.skip_reason or "",
    }


class _TemplateFieldMap(dict):
    """Format mapping that leaves unknown placeholders untouched instead of raising.

    HookManager.fire() swallows hook exceptions, so a KeyError from one bad placeholder
    would silently drop the whole notification. Rendering `{typo}` literally makes the
    mistake visible while the rest of the message still gets through.
    """

    def __missing__(self, key: str) -> str:
        logging.warning(f"Hook message template references unknown field '{key}'")
        return "{" + key + "}"


def render_template(template: str, context: HookContext) -> str:
    """Render a hook message template against a context."""
    return template.format_map(_TemplateFieldMap(_template_fields(context)))


@dataclass
class Hook(ABC):
    name: str = ""
    events: List[str] = field(default_factory=list)
    enabled: bool = True
    task_types: List[str] = field(default_factory=list)  # [] = all types

    @abstractmethod
    def run(self, context: HookContext) -> None:
        pass

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "events": [e.value if hasattr(e, "value") else e for e in self.events],
            "enabled": self.enabled,
            "task_types": list(self.task_types),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Hook":
        raise NotImplementedError


@dataclass
class LoggingHook(Hook):
    level: str = "INFO"

    def run(self, context: HookContext) -> None:
        level = getattr(logging, self.level.upper(), logging.INFO)
        logging.log(
            level,
            f"[Hook] {context.event} | task={context.task_name} item={context.item_name} error={context.error}",
        )

    def to_dict(self) -> dict:
        return {**super().to_dict(), "level": self.level}

    @classmethod
    def from_dict(cls, d: dict) -> "LoggingHook":
        return cls(
            name=d.get("name", ""),
            events=d.get("events", []),
            enabled=d.get("enabled", True),
            task_types=d.get("task_types", []),
            level=d.get("level", "INFO"),
        )


@dataclass
class NotificationHook(Hook):
    notification_type: str = "info"
    message_template: str = DEFAULT_MESSAGE_TEMPLATE
    _notify: Optional[Callable[[str, str], None]] = field(default=None, repr=False)

    def run(self, context: HookContext) -> None:
        message = render_template(self.message_template, context)
        if self._notify:
            self._notify(message, self.notification_type)
        else:
            logging.info(f"[NotificationHook] {self.notification_type.upper()}: {message}")

    def to_dict(self) -> dict:
        return {
            **super().to_dict(),
            "notification_type": self.notification_type,
            "message_template": self.message_template,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationHook":
        return cls(
            name=d.get("name", ""),
            events=d.get("events", []),
            enabled=d.get("enabled", True),
            task_types=d.get("task_types", []),
            notification_type=d.get("notification_type", "info"),
            message_template=d.get("message_template", DEFAULT_MESSAGE_TEMPLATE),
        )


_WEBHOOK_ALLOWED_METHODS = {"POST", "GET"}
_WEBHOOK_ALLOWED_SCHEMES = {"http", "https"}


def _validate_webhook_url(url: str) -> str:
    """Raise ValueError if url is empty or uses a disallowed scheme."""
    if not url:
        raise ValueError("WebhookHook: url must not be empty")
    from urllib.parse import urlparse
    scheme = urlparse(url).scheme.lower()
    if scheme not in _WEBHOOK_ALLOWED_SCHEMES:
        raise ValueError(
            f"WebhookHook: url scheme '{scheme}' is not allowed (must be http or https)"
        )
    return url


@dataclass
class WebhookHook(Hook):
    """Hook that POSTs a JSON payload to a URL. Fires in a daemon thread (non-blocking)."""
    url: str = ""
    method: str = "POST"
    timeout: int = 5

    def __post_init__(self):
        if self.method.upper() not in _WEBHOOK_ALLOWED_METHODS:
            raise ValueError(
                f"WebhookHook: method '{self.method}' is not allowed (must be one of {sorted(_WEBHOOK_ALLOWED_METHODS)})"
            )
        self.method = self.method.upper()

    def run(self, context: HookContext) -> None:
        _validate_webhook_url(self.url)
        threading.Thread(target=self._post, args=(context,), daemon=True).start()

    def _post(self, context: HookContext) -> None:
        try:
            import requests
            payload = {
                "event": context.event,
                "task_name": context.task_name,
                "task_type": context.task_type,
                "item_name": context.item_name,
                # Deprecated duplicate of item_name: the payload shipped in v0.5.1 with
                # this key, so endpoints reading it keep working. Drop after v0.6.
                "lamella_name": context.item_name,
                "item_id": context.item_id,
                "task_id": context.task_id,
                "experiment_id": context.experiment_id,
                "experiment_name": context.experiment_name,
                "tasks_remaining": context.tasks_remaining,
                "tasks_total": context.tasks_total,
                "error": context.error,
                "skip_reason": context.skip_reason,
                "timestamp": context.timestamp,
            }
            requests.request(self.method, self.url, json=payload, timeout=self.timeout)
        except Exception:
            logging.exception(f"WebhookHook '{self.name}' failed")

    def to_dict(self) -> dict:
        return {**super().to_dict(), "url": self.url, "method": self.method, "timeout": self.timeout}

    @classmethod
    def from_dict(cls, d: dict) -> "WebhookHook":
        method = d.get("method", "POST").upper()
        if method not in _WEBHOOK_ALLOWED_METHODS:
            logging.warning(f"WebhookHook: invalid method '{method}', defaulting to POST")
            method = "POST"
        return cls(
            name=d.get("name", ""),
            events=d.get("events", []),
            enabled=d.get("enabled", True),
            task_types=d.get("task_types", []),
            url=d.get("url", ""),
            method=method,
            timeout=d.get("timeout", 5),
        )


@dataclass
class FunctionHook(Hook):
    """Hook that calls a Python callable. Not serializable — registered in code only."""
    callback: Optional[Callable[["HookContext"], None]] = field(default=None, repr=False)

    def run(self, context: HookContext) -> None:
        if self.callback:
            self.callback(context)


HOOK_TYPES: Dict[str, Type[Hook]] = {
    "LoggingHook": LoggingHook,
    "NotificationHook": NotificationHook,
    "WebhookHook": WebhookHook,
}


class HookManager:
    def __init__(self) -> None:
        self._hooks: List[Hook] = []
        self._notifier: Optional[Callable[[str, str], None]] = None

    def register(self, hook: Hook) -> None:
        self._apply_notifier(hook)
        self._hooks.append(hook)

    def fire(self, context: HookContext) -> None:
        for hook in self._hooks:
            if not hook.enabled:
                continue
            if context.event not in hook.events:
                continue
            if hook.task_types and context.task_type not in hook.task_types:
                continue
            try:
                hook.run(context)
            except Exception:
                logging.exception(f"Hook '{hook.name}' failed silently")

    def set_notifier(self, notify: Optional[Callable[[str, str], None]]) -> None:
        """Supply the sink NotificationHooks deliver to, as notify(message, type).

        Hooks loaded from YAML have no callable — it cannot be serialized — so the
        application injects one here. It applies to hooks already registered and to
        every hook registered afterwards, so hooks added later (e.g. from the config
        widget) notify too instead of silently falling back to logging.

        The caller owns thread-safety: hooks run on the worker thread that fired them,
        so a Qt application should pass a signal's ``emit``. Leave unset for headless
        runs and NotificationHook logs instead.
        """
        self._notifier = notify
        for hook in self._hooks:
            self._apply_notifier(hook)

    def _apply_notifier(self, hook: Hook) -> None:
        """Attach the notifier to a NotificationHook that does not already have one."""
        if self._notifier is None:
            return
        if isinstance(hook, NotificationHook) and hook._notify is None:
            hook._notify = self._notifier

    def to_dict(self) -> dict:
        return {
            "hooks": [h.to_dict() for h in self._hooks if h.__class__.__name__ in HOOK_TYPES]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HookManager":
        manager = cls()
        for hook_dict in d.get("hooks", []):
            hook_type = hook_dict.get("type")
            if hook_type in HOOK_TYPES:
                manager.register(HOOK_TYPES[hook_type].from_dict(hook_dict))
            else:
                logging.warning(f"Unknown hook type '{hook_type}', skipping")
        return manager

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f)

    @classmethod
    def load_yaml(cls, path: str) -> "HookManager":
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)


def fire_event(
    hook_manager: Optional[HookManager],
    event: str,
    **fields: Any,
) -> None:
    """Fire ``event`` on ``hook_manager``, building the context from ``fields``.

    A no-op when hook_manager is None, so a producer can call this unconditionally
    instead of guarding every call site. This is the whole producer-side API: anything
    that runs work — AutoLamella tasks today, milling or acquisition next — can emit
    lifecycle events without importing anything else from this module.

    Fires on the calling thread, so a slow hook slows the caller. HookManager.fire()
    isolates hook exceptions, so a bad hook cannot break the producer.
    """
    if hook_manager is None:
        return
    hook_manager.fire(HookContext(event=event, **fields))

"""Hook system for AutoLamella task lifecycle events."""

import logging
import time
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from fibsem.applications.autolamella.structures import AutoLamellaTaskState
    from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI


class HookEvent(str, Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"


@dataclass
class HookContext:
    event: str
    task_name: str = ""
    task_type: str = ""
    lamella_name: str = ""
    task_state: Optional["AutoLamellaTaskState"] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        # Normalize HookEvent enums to plain strings so f-strings and
        # comparisons work consistently regardless of Python version.
        if hasattr(self.event, "value"):
            self.event = self.event.value


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
            f"[Hook] {context.event} | task={context.task_name} lamella={context.lamella_name} error={context.error}",
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
    message_template: str = "Task {task_name} {event} for {lamella_name}"
    _notify: Optional[Callable[[str, str], None]] = field(default=None, repr=False)

    def run(self, context: HookContext) -> None:
        message = self.message_template.format(
            task_name=context.task_name,
            event=context.event,
            lamella_name=context.lamella_name,
            error=context.error or "",
        )
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
            message_template=d.get("message_template", "Task {task_name} {event} for {lamella_name}"),
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
}


class HookManager:
    def __init__(self) -> None:
        self._hooks: List[Hook] = []

    def register(self, hook: Hook) -> None:
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

    def wire(self, parent_ui: "AutoLamellaUI") -> None:
        """Inject runtime dependencies into hooks after loading from YAML.

        Called after load_yaml() to attach Qt-dependent callables that cannot
        be serialized (e.g. the _notify callable on NotificationHook).
        """
        if parent_ui is None:
            return

        def _notify(message: str, notification_type: str) -> None:
            parent_ui._hook_toast_signal.emit(message, notification_type)

        for hook in self._hooks:
            if isinstance(hook, NotificationHook):
                hook._notify = _notify

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

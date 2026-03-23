# Hook System Design for AutoLamella Task Execution

## What Exists Today

The current system has these event/notification mechanisms:

1. **PyQt Signals** (`workflow_update_signal`, `step_update_signal`) — tightly coupled to UI, only work when `parent_ui` is set, main thread only
2. **Logging** — `logging.info/debug` calls scattered through task lifecycle, one-way, no extensibility
3. **Structured status dicts** — emitted via signals with task/lamella metadata, but only consumed by the workflow timeline widget
4. **psygnal `@evented` dataclasses** — `AutoLamellaTaskState` fires change events, but nothing subscribes to them for side effects

## What a Hook System Adds

### New Capabilities
| Capability | Current | With Hooks |
|---|---|---|
| UI toast on task complete | Not possible without modifying task code | Register `NotificationHook` — zero task code changes |
| Slack/Teams alert on failure | Would need to edit `TaskManager._run_queue()` | Register `WebhookHook` with failure event |
| Custom post-processing after task | Must subclass the task or modify `post_task()` | Register `FunctionHook` — attach any callable |
| AI model evaluates task output | Would need deep integration into task code | Hook receives `HookContext` with task state, can inspect results |
| Per-user/per-experiment customization | Hardcoded behavior, same for everyone | Different hook configurations per workflow |
| External service integration | Each integration needs code changes | Generic webhook/function hooks handle any service |
| Headless operation (no UI) | Signals go nowhere when `parent_ui=None` | Hooks work without UI — logging, webhooks, functions all fire |

### What It Doesn't Add
- No new data — hooks consume existing `AutoLamellaTaskState` information
- No new lifecycle points — hooks fire at existing pre_task/post_task boundaries
- No performance improvement — slight overhead from dispatching

## Complexity Analysis

### Implementation Cost

| Component | Lines of Code | Complexity |
|---|---|---|
| `hooks.py` (new file) | ~150 | Low — dataclasses, ABC, simple dispatch loop |
| Modify `tasks.py` (`run()`) | ~10 lines changed | Low — add 3 `_fire_hook()` calls |
| Modify `manager.py` | ~10 lines changed | Low — accept `HookManager`, fire workflow events |
| Built-in hooks (Logging, Notification, Webhook, Function) | ~60 | Low — each is 5-15 lines |
| **Total** | **~230 lines** | **Low** |

### Maintenance Burden
- **Low** — hooks are decoupled by design. Adding a new hook type doesn't touch task/manager code
- **Risk**: hook failures must be silently caught (`try/except` in `fire()`), otherwise a bad hook could crash the workflow
- **Testing**: each hook is independently testable, no mocking of microscope needed

### Alternative: Just Use Signals
Could we achieve the same by adding more PyQt signals?

| | Signals | Hooks |
|---|---|---|
| Works without UI | No (`parent_ui` required) | Yes |
| Works from worker threads | Requires `QMetaObject.invokeMethod` | Yes (hooks run in calling thread) |
| Serializable to YAML config | No (signals are code-only) | Yes (hooks have `to_dict`/`from_dict`) |
| User-configurable | No | Yes (protocol file or UI) |
| External service integration | Awkward (signal -> slot -> HTTP call) | Direct (WebhookHook) |
| Entry point plugins | Not natural | Natural (mirrors existing `fibsem.tasks` pattern) |

## When Is It Worth It?

### Worth it if:
- You want users/experiments to have different notification behaviors without code changes
- You plan to integrate external services (Slack, email, webhooks, AI evaluation)
- You want headless/CLI workflows to have the same extensibility as UI workflows
- The plugin/entry-point pattern for tasks should extend to lifecycle events

### Not worth it if:
- The only need is "show a toast when task finishes" — that's a 5-line signal connection
- Hook configuration via YAML/UI is not needed — just hardcode the behavior
- All workflows are UI-driven and `parent_ui` is always available

## Evaluation

The hook system is **worth it** given the stated goals (FIB-87: notifications, function calls, other services). The implementation is small (~230 lines), low-risk (hooks are fire-and-forget with error isolation), and follows existing codebase patterns (mirrors the task plugin registry).

The key value is **decoupling**: new integrations don't require modifying task execution code. The alternative — adding each integration directly to `TaskManager` or `AutoLamellaTask` — works for 1-2 integrations but becomes messy at 5+.

## Proposed API

### Hook Events

```python
class HookEvent(str, Enum):
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
```

### Hook Context

```python
@dataclass
class HookContext:
    event: str
    task_name: str = ""
    task_type: str = ""
    lamella_name: str = ""
    task_state: Optional[AutoLamellaTaskState] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=lambda: time.time())
```

### Hook Base Class

```python
@dataclass
class Hook(ABC):
    name: str = ""
    events: List[str] = field(default_factory=list)

    @abstractmethod
    def run(self, context: HookContext) -> None:
        pass
```

### Hook Manager

```python
class HookManager:
    def __init__(self):
        self._hooks: List[Hook] = []

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)

    def fire(self, context: HookContext) -> None:
        for hook in self._hooks:
            if context.event in hook.events:
                try:
                    hook.run(context)
                except Exception:
                    logging.exception(f"Hook '{hook.name}' failed")
```

### Built-in Hooks

```python
@dataclass
class LoggingHook(Hook):
    level: str = "INFO"
    def run(self, context): ...

@dataclass
class NotificationHook(Hook):
    notification_type: str = "info"
    def run(self, context): ...

@dataclass
class WebhookHook(Hook):
    url: str = ""
    def run(self, context): ...

@dataclass
class FunctionHook(Hook):
    callback: Optional[Callable] = None
    def run(self, context): ...
```

### Integration into Task Lifecycle

```python
# AutoLamellaTask.run()
def run(self) -> None:
    self.pre_task()
    self._fire_hook(HookEvent.TASK_STARTED)
    try:
        self._run()
    except Exception as e:
        self._fire_hook(HookEvent.TASK_FAILED, error=str(e))
        raise
    self.post_task()
    self._fire_hook(HookEvent.TASK_COMPLETED)
```

### Usage Example

```python
from fibsem.applications.autolamella.workflows.tasks.hooks import (
    HookManager, HookEvent, LoggingHook, NotificationHook, FunctionHook
)

hook_manager = HookManager()

# Log all task events
hook_manager.register(LoggingHook(
    name="task_logger",
    events=[HookEvent.TASK_STARTED, HookEvent.TASK_COMPLETED, HookEvent.TASK_FAILED],
))

# Toast on completion
hook_manager.register(NotificationHook(
    name="completion_toast",
    events=[HookEvent.TASK_COMPLETED],
    notification_type="success",
))

# Custom callback
def on_task_done(context):
    print(f"Task {context.task_name} finished for {context.lamella_name}")

hook_manager.register(FunctionHook(
    name="custom_callback",
    events=[HookEvent.TASK_COMPLETED],
    callback=on_task_done,
))

# Wire into task manager
task_manager = TaskManager(
    microscope=microscope,
    experiment=experiment,
    parent_ui=parent_ui,
    hook_manager=hook_manager,
)
task_manager.run(task_names, lamella_names)
```

## Implementation Plan

**Phase 1**: `hooks.py` with base class + `LoggingHook` + `FunctionHook`. Wire `HookManager` into `TaskManager` and `AutoLamellaTask.run()`. No existing code removed or modified beyond adding hook fire calls. Existing signals, logging, and UI callbacks continue unchanged.

**Phase 2**: Add `NotificationHook`, `WebhookHook`. Optionally replicate existing signal-based behaviors as hooks to validate parity.

**Phase 3** (optional): Migrate specific behaviors from signals to hooks where beneficial (e.g., headless workflows). UI signal connections stay for anything that needs them.

Skip YAML serialization and UI configuration initially — wire hooks in code first, add config later if needed.

---

## YAML Serialization

### Pattern

Follows the established `to_dict()` / `from_dict()` pattern used throughout the codebase (`AutoLamellaTaskConfig`, `UserPreferences`, `ImageSettings`, etc.). Each hook serializes itself to a plain dict; `HookManager` owns the list and handles round-tripping.

### Hook `to_dict` / `from_dict`

Every `Hook` subclass adds a `type` discriminator key to allow reconstruction on load. Non-serializable fields (e.g. `FunctionHook.callback`, `NotificationHook._notify`) are excluded.

```python
@dataclass
class Hook(ABC):
    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "name": self.name,
            "events": list(self.events),
        }

@dataclass
class LoggingHook(Hook):
    level: str = "INFO"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "level": self.level}

    @classmethod
    def from_dict(cls, d: dict) -> "LoggingHook":
        return cls(name=d.get("name", ""), events=d.get("events", []), level=d.get("level", "INFO"))


@dataclass
class NotificationHook(Hook):
    notification_type: str = "info"
    message_template: str = "Task {task_name} {event} for {lamella_name}"
    # _notify callable is NOT serialized — injected at runtime

    def to_dict(self) -> dict:
        return {**super().to_dict(), "notification_type": self.notification_type, "message_template": self.message_template}

    @classmethod
    def from_dict(cls, d: dict) -> "NotificationHook":
        return cls(
            name=d.get("name", ""),
            events=d.get("events", []),
            notification_type=d.get("notification_type", "info"),
            message_template=d.get("message_template", "Task {task_name} {event} for {lamella_name}"),
        )


@dataclass
class WebhookHook(Hook):
    url: str = ""
    method: str = "POST"

    def to_dict(self) -> dict:
        return {**super().to_dict(), "url": self.url, "method": self.method}

    @classmethod
    def from_dict(cls, d: dict) -> "WebhookHook":
        return cls(name=d.get("name", ""), events=d.get("events", []), url=d.get("url", ""), method=d.get("method", "POST"))

# FunctionHook is NOT serializable — registered in code only.
```

### HookManager serialization

```python
HOOK_TYPES: Dict[str, Type[Hook]] = {
    "LoggingHook": LoggingHook,
    "NotificationHook": NotificationHook,
    "WebhookHook": WebhookHook,
}

class HookManager:
    def to_dict(self) -> dict:
        return {"hooks": [h.to_dict() for h in self._hooks if h.__class__.__name__ in HOOK_TYPES]}

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
```

### YAML file format

Stored alongside the protocol file as `autolamella-hooks.yaml`:

```yaml
hooks:
  - type: LoggingHook
    name: task_logger
    events: [task_started, task_completed, task_failed]
    level: INFO

  - type: NotificationHook
    name: completion_toast
    events: [task_completed]
    notification_type: success
    message_template: "Task {task_name} complete for {lamella_name}"

  - type: NotificationHook
    name: failure_toast
    events: [task_failed]
    notification_type: error
    message_template: "Task {task_name} FAILED: {error}"

  - type: WebhookHook
    name: slack_alert
    events: [task_failed, workflow_completed]
    url: https://hooks.slack.com/services/XXX
    method: POST
```

---

## UI Configuration

### Approach

A `HookConfigWidget` (standalone `QWidget`) showing registered hooks in a list with add/remove/edit controls. Follows the `MillingTaskConfigWidget2` pattern — config object in, edit, config object out.

Two integration options:
- **Embedded**: new "Hooks" `TitledPanel` section in the existing `PreferencesDialog`
- **Standalone**: "Manage Hooks..." button in the workflow panel that opens the widget as a modal

### Widget structure

```
HookConfigWidget
├── QListWidget  (each row: icon + name + events summary)
├── [Add ▼]  [Edit]  [Remove]
└── Add → HookTypeDialog (pick type) → HookEditDialog (fill fields)
    Edit → HookEditDialog (pre-populated)
```

```python
class HookConfigWidget(QWidget):
    hooks_changed = pyqtSignal(list)  # emits List[Hook] on any change

    def get_hooks(self) -> List[Hook]: ...
```

### HookEditDialog fields per type

| Hook type | Fields |
|---|---|
| `LoggingHook` | name, events (checkboxes), level (`QComboBox`: DEBUG/INFO/WARNING) |
| `NotificationHook` | name, events, notification_type (`QComboBox`: info/success/warning/error), message_template (`QLineEdit`) |
| `WebhookHook` | name, events, url (`QLineEdit`), method (`QComboBox`: POST/GET) |

Events shown as checkboxes: `task_started`, `task_completed`, `task_failed`, `workflow_started`, `workflow_completed`.

---

## Toast Notification Hook — End-to-End Example

### Thread-safety challenge

`NotificationHook.run()` needs to call `show_toast()` on `AutoLamellaMainUI`, but tasks execute on a worker thread. Qt widgets must be touched only from the main thread.

### Solution: inject a `pyqtSignal`-backed callable

Add a signal to `AutoLamellaMainUI` and pass a `_notify` lambda that emits it. `pyqtSignal.emit()` is thread-safe in PyQt5 — signals emitted from a worker thread are automatically queued to the main thread. This is the same pattern used by `workflow_update_signal`.

```python
# NotificationHook — _notify is injected at runtime, not serialized
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
```

```python
# AutoLamellaMainUI — add signal + wiring
class AutoLamellaMainUI(QMainWindow):
    _hook_toast_signal = pyqtSignal(str, str)   # (message, notification_type)

    def setup_hooks(self) -> HookManager:
        self._hook_toast_signal.connect(self.show_toast)

        def _notify(message: str, notification_type: str):
            self._hook_toast_signal.emit(message, notification_type)

        manager = HookManager()
        manager.register(NotificationHook(
            name="completion_toast",
            events=[HookEvent.TASK_COMPLETED],
            notification_type="success",
            message_template="Task {task_name} complete for {lamella_name}",
            _notify=_notify,
        ))
        manager.register(NotificationHook(
            name="failure_toast",
            events=[HookEvent.TASK_FAILED],
            notification_type="error",
            message_template="Task {task_name} FAILED: {error}",
            _notify=_notify,
        ))
        return manager
```

### Call trace

1. `AutoLamellaTask.run()` → `_fire_hook(HookEvent.TASK_COMPLETED)`
2. `HookManager.fire()` finds `completion_toast` (worker thread)
3. `NotificationHook.run(context)` formats message, calls `_notify(message, "success")`
4. `_notify` emits `_hook_toast_signal` from worker thread
5. Qt queues delivery to main thread
6. `show_toast()` fires → `ToastManager.show_toast()` → `ToastNotification` appears

### Headless fallback

When `parent_ui` is `None`, `_notify` is never set. `NotificationHook.run()` falls back to `logging.info()` — no Qt dependency, no crash.

---

## Design Gaps & Additional Considerations

### 1. Post-load wiring

`HookManager.load_yaml()` reconstructs hooks from YAML, but runtime dependencies (`_notify` on `NotificationHook`) won't be set — they are not serialized. Without explicit wiring, hooks loaded from disk silently fall back to logging-only, which is confusing.

Solution: add a `wire(parent_ui)` method to `HookManager` that runs after load, before the manager is passed to `TaskManager`:

```python
class HookManager:
    def wire(self, parent_ui) -> None:
        """Inject runtime dependencies into hooks after loading from YAML."""
        if parent_ui is None:
            return
        def _notify(message: str, notification_type: str):
            parent_ui._hook_toast_signal.emit(message, notification_type)
        for hook in self._hooks:
            if isinstance(hook, NotificationHook):
                hook._notify = _notify
```

Call sequence:
```python
manager = HookManager.load_yaml("autolamella-hooks.yaml")
manager.wire(parent_ui)   # inject runtime deps
task_manager = TaskManager(..., hook_manager=manager)
```

### 2. `enabled` flag per hook

Hooks should be toggleable without being removed from the config. Add `enabled: bool = True` to the base class, serialized in YAML, and checked in `HookManager.fire()`:

```python
@dataclass
class Hook(ABC):
    name: str = ""
    events: List[str] = field(default_factory=list)
    enabled: bool = True

class HookManager:
    def fire(self, context: HookContext) -> None:
        for hook in self._hooks:
            if hook.enabled and context.event in hook.events:
                ...
```

YAML:
```yaml
- type: NotificationHook
  name: completion_toast
  enabled: false   # temporarily disabled
  events: [task_completed]
```

UI: each row in `HookConfigWidget` gets a toggle checkbox.

### 3. `WebhookHook` must be fire-and-forget

HTTP requests in `run()` execute on the task worker thread. A slow or unreachable endpoint adds latency to every task. `WebhookHook.run()` should dispatch to a daemon thread:

```python
@dataclass
class WebhookHook(Hook):
    def run(self, context: HookContext) -> None:
        threading.Thread(target=self._post, args=(context,), daemon=True).start()

    def _post(self, context: HookContext) -> None:
        try:
            import requests
            requests.request(self.method, self.url, json=context.__dict__, timeout=5)
        except Exception:
            logging.exception(f"WebhookHook '{self.name}' failed")
```

The `try/except` in `HookManager.fire()` still catches thread-start failures, but the HTTP work itself is isolated.

### 4. Task-type filtering (optional)

`HookContext` already carries `task_type`. A hook might only make sense for specific task types (e.g. toast on `MillTask` completion, not `ImagingTask`). Reserve the field on the base class now — empty list means "all types":

```python
@dataclass
class Hook(ABC):
    name: str = ""
    events: List[str] = field(default_factory=list)
    enabled: bool = True
    task_types: List[str] = field(default_factory=list)  # [] = match all
```

`HookManager.fire()` check:
```python
if hook.task_types and context.task_type not in hook.task_types:
    continue
```

Not needed for Phase 1 but costs one field and one `if` line. Avoids a breaking change to the YAML schema later.

"""The event buffer: a bounded in-memory log adapting push to pull.

Events are born on worker threads — the microscope's psygnal signals fire
mid-mill, the hook system fires as tasks finish — but agents consume over
HTTP, whenever they next ask. This module is the adapter between those two
tempos: taps subscribe to each source and drop plain-data records into a
bounded, sequence-numbered buffer; the ``/app/events`` long-poll drains it.

Rules the taps live by:

* **Copy, serialize, return.** psygnal subscribers run synchronously on the
  emitting thread (the documented THREADING CONTRACT) — a slow tap literally
  slows the mill. Serialization to plain dicts happens immediately, and
  nothing downstream ever holds a live object.
* **No Qt, ever.** Taps run on whatever thread emits; the process disables
  automatic GC precisely because off-thread Qt finalization crashes.
* **Metadata, not pixels.** Acquisition events carry beam/field-of-view/shape;
  the images themselves are fetched through the preview endpoints on demand.
  Known upstream gap: the acquisition signals fire only from the streaming
  live-view worker -- a one-shot ``acquire_image`` emits nothing on them, so
  single-shot acquisitions are invisible to this stream until that changes.

Sequence numbers make polling honest: a client asks for everything after seq
N, and if eviction has eaten past N the response's ``oldest_available`` says
so — a visible gap instead of silent continuity.

Wiring that deliberately does NOT live here (it belongs to the embedded
hosting, FIB-845): constructing the buffer in the app, attaching the
microscope taps at connect time, and re-registering the lifecycle hook inside
``setup_hooks()`` — the app rebuilds its hook set every run, so a
once-at-startup registration silently goes deaf after the first run.
"""

import dataclasses
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "EventBuffer",
    "attach_microscope_taps",
    "make_lifecycle_hook",
    "to_plain",
]


def to_plain(value: Any) -> Any:
    """Recursively convert a typed record into JSON-able plain data."""
    # Enum before str: str-enums (MillingProgressStatus et al.) must serialize
    # as their wire value ("stage-update"), the vocabulary consumers already
    # parse — not the Python member name.
    if isinstance(value, Enum):
        return value.value if isinstance(value.value, str) else value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, bytes):
        try:  # numpy scalars
            return to_plain(value.item())
        except (AttributeError, ValueError):
            pass
    return str(value)


class EventBuffer:
    """Bounded, sequence-numbered, thread-safe event log with long-poll wait."""

    def __init__(self, maxlen: int = 1000):
        self._events: "deque[Dict[str, Any]]" = deque(maxlen=maxlen)
        self._seq = 0
        self._cond = threading.Condition()

    def append(self, kind: str, payload: Dict[str, Any]) -> int:
        """Record one event. Cheap and non-blocking; safe from any thread."""
        with self._cond:
            self._seq += 1
            self._events.append(
                {
                    "seq": self._seq,
                    "timestamp": time.time(),
                    "kind": kind,
                    "payload": payload,
                }
            )
            self._cond.notify_all()
            return self._seq

    def events_since(self, since: int = 0) -> Dict[str, Any]:
        with self._cond:
            events = [e for e in self._events if e["seq"] > since]
            oldest = self._events[0]["seq"] if self._events else None
        return {
            "available": True,
            "events": events,
            "latest_seq": self._seq,
            # A client that asked for `since < oldest_available - 1` missed
            # evicted events: re-snapshot state rather than assume continuity.
            "oldest_available": oldest,
        }

    def wait_for(self, since: int = 0, timeout: float = 0.0) -> Dict[str, Any]:
        """events_since, but park up to ``timeout`` seconds for something new."""
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._seq <= since:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
        return self.events_since(since)


# --- taps: the sources feeding the buffer ------------------------------------------


def _image_metadata(image: Any) -> Dict[str, Any]:
    """Metadata-only record of an acquisition; pixels never enter the buffer."""
    payload: Dict[str, Any] = {
        "shape": list(getattr(image, "data", None).shape)
        if getattr(image, "data", None) is not None
        else None,
        "beam_type": None,
        "hfw": None,
    }
    metadata = getattr(image, "metadata", None)
    settings = getattr(metadata, "image_settings", None) if metadata else None
    if settings is not None:
        beam_type = getattr(settings, "beam_type", None)
        payload["beam_type"] = beam_type.name if beam_type is not None else None
        payload["hfw"] = to_plain(getattr(settings, "hfw", None))
    return payload


def attach_microscope_taps(buffer: EventBuffer, microscope) -> List[Callable[[], None]]:
    """Subscribe the buffer to a microscope's progress signals.

    Returns disposer callables (one per tap) so a hosting can detach cleanly.
    Every callback serializes immediately and returns — nothing blocks the
    emitting thread beyond building one small dict.
    """
    disposers: List[Callable[[], None]] = []

    def tap(signal, kind: str, serialize: Callable[[Any], Dict[str, Any]]):
        def callback(value):
            buffer.append(kind, serialize(value))

        signal.connect(callback)
        disposers.append(lambda: signal.disconnect(callback))

    tap(microscope.milling_progress_signal, "milling_progress", to_plain)
    tap(microscope.spot_burn_progress_signal, "spot_burn_progress", to_plain)
    tap(microscope.tiled_acquisition_signal, "tiled_acquisition", to_plain)
    tap(
        microscope.stage_position_changed,
        "stage_position_changed",
        lambda position: {"position": to_plain(position.to_dict())},
    )
    tap(microscope.sem_acquisition_signal, "sem_acquisition", _image_metadata)
    tap(microscope.fib_acquisition_signal, "fib_acquisition", _image_metadata)

    fm = getattr(microscope, "fm", None)
    if fm is not None:
        tap(fm.acquisition_progress_signal, "fm_acquisition_progress", to_plain)

    return disposers


def make_lifecycle_hook(buffer: EventBuffer):
    """A FunctionHook feeding workflow lifecycle events into the buffer.

    Must be registered per run (the app rebuilds its HookManager in
    setup_hooks each run); HookContext already deep-copies task_state, so
    serializing here reads a stable snapshot.
    """
    from fibsem.hooks import FunctionHook, HookContext, HookEvent

    def callback(context: "HookContext") -> None:
        task_state = getattr(context, "task_state", None)
        buffer.append(
            context.event,
            {
                "task_name": context.task_name,
                "task_type": context.task_type,
                "item_name": context.item_name,
                "item_id": context.item_id,
                "task_id": context.task_id,
                "experiment_id": context.experiment_id,
                "experiment_name": context.experiment_name,
                "tasks_remaining": context.tasks_remaining,
                "tasks_total": context.tasks_total,
                "error": context.error,
                "skip_reason": context.skip_reason,
                "timestamp": to_plain(context.timestamp),
                "task_state": to_plain(task_state) if task_state is not None else None,
            },
        )

    # An empty subscription list matches nothing, so name every event: the
    # buffer is the one consumer that genuinely wants the whole lifecycle.
    return FunctionHook(callback=callback, events=[e.value for e in HookEvent])


def buffer_or_unavailable(
    buffer: Optional[EventBuffer], since: int, timeout: float
) -> Dict[str, Any]:
    """The facade-facing read: tolerate a hosting that wired no buffer."""
    if buffer is None:
        return {
            "available": False,
            "events": [],
            "latest_seq": None,
            "oldest_available": None,
        }
    return buffer.wait_for(since=since, timeout=timeout)

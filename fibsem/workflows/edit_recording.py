"""Edits to an experiment's plan, for the experiment's record.

Before an edit is applied, the code applying it *touches* what it is about to
change: it hands over a reader of the value. The value is read then, as the
edit's ``before``, the first time it is touched, and read again when the edits
settle, as its ``after``, on :meth:`PendingEdits.flush`: at once for an
agent's edit, after a pause in the editors (``fibsem/ui/edit_recording.py``). So an
edit made in a hundred drag steps is one ``edit`` event, and one that ends
where it started is none.

Who made it is the envelope's ``actor`` (``fibsem.acting``) as it was when the
edit was first touched: an edit can settle while something else is acting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

from fibsem.acting import acting, current_actor

EDIT = "edit"


def serialise(value: Any) -> Any:
    """A value as the record keeps it: the object's own ``to_dict()``."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): serialise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialise(v) for v in value]
    return value


@dataclass
class _Edit:
    item: Optional[Dict[str, Any]]
    task: Optional[str]
    target: str
    via: str
    actor: Optional[str]
    read: Callable[[], Any]
    before: Any


class PendingEdits:
    """The edits touched since they last settled, one per thing edited.

    ``microscope`` is asked for the microscope when the edits are recorded, so
    an editor that outlives a reconnect records to the current one. Nothing
    here raises: an edit that cannot be recorded is still made.
    """

    def __init__(
        self,
        microscope: Callable[[], Any],
        via: str,
    ):
        self._microscope = microscope
        self._via = via
        self._pending: Dict[Tuple[Any, Optional[str], str], _Edit] = {}

    def touch(
        self,
        lamella: Any,
        task: Optional[str],
        target: str,
        read: Callable[[], Any],
        via: Optional[str] = None,
    ) -> None:
        """Call before changing *target*; *read* returns its current value."""
        try:
            key = (getattr(lamella, "id", None), task, target)
            if key not in self._pending:
                item = (
                    None
                    if lamella is None
                    else {"id": lamella.id, "name": lamella.name}
                )
                self._pending[key] = _Edit(
                    item=item,
                    task=task,
                    target=target,
                    via=via or self._via,
                    actor=current_actor(),
                    read=read,
                    before=serialise(read()),
                )
            self._touched()
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug(f"could not note an edit to {target}", exc_info=True)

    def touch_task_configs(
        self, lamellae: Iterable[Any], tasks: Iterable[str], via: Optional[str] = None
    ) -> None:
        """Each lamella's config for each task: for an edit that replaces them."""
        tasks = list(tasks)
        for lamella in lamellae:
            for task in tasks:
                self.touch(
                    lamella,
                    task,
                    "task_config",
                    lambda lamella=lamella, task=task: lamella.task_config.get(task),
                    via,
                )

    def touch_patterns(self, lamella: Any, via: Optional[str] = None) -> None:
        """Every milling config of *lamella*: for an edit that can move any of them."""
        try:
            keys = [
                (task, key)
                for task, config in lamella.task_config.items()
                for key in getattr(config, "milling", None) or {}
            ]
        except Exception:  # noqa: BLE001 - recording must not matter
            logging.debug("could not list the patterns an edit may move", exc_info=True)
            return
        for task, key in keys:
            self.touch(
                lamella,
                task,
                f"milling.{key}",
                lambda task=task, key=key: lamella.task_config[task].milling.get(key),
                via,
            )

    def _touched(self) -> None:
        """After each touch: where the edits settle on a timer, restart it."""

    def flush(self) -> None:
        """Record every touched edit that changed something. Safe to call any time."""
        pending, self._pending = self._pending, {}
        if not pending:
            return
        record = getattr(self._microscope(), "record_event", None)
        for edit in pending.values():
            try:
                after = serialise(edit.read())
                if record is None or _same(after, edit.before):
                    continue  # nowhere to record it, or put back as it was
                with acting(edit.actor):  # who made it, not who flushed it
                    record(
                        EDIT,
                        {
                            "item": edit.item,
                            "task": edit.task,
                            "target": edit.target,
                            "before": edit.before,
                            "after": after,
                            "via": edit.via,
                        },
                    )
            except Exception:  # noqa: BLE001 - recording must not matter
                logging.debug(
                    f"could not record an edit to {edit.target}", exc_info=True
                )


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - an array compares elementwise; call it changed
        return False

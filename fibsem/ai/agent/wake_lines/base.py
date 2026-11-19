"""What every kind's lines share (`wake_lines`)."""

from typing import Any, Callable, Dict, Optional, Tuple

Line = Optional[Tuple[str, str]]
Lines = Dict[str, Callable[[Dict[str, Any]], Line]]


def item_of(event) -> str:
    """`` item=<name>`` for the lamella or grid the event happened on, or ''."""
    item = event.get("item")
    name = item.get("name") if isinstance(item, dict) else None
    return f" item={name}" if name else ""

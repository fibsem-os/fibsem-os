"""The selected item of an app's item list, which the shared Image and
Fluorescence tabs follow to save an acquisition with the selected item."""

from typing import Any, Optional, Protocol

__all__ = ["ItemSelection"]


class ItemSelection(Protocol):
    # what an item is called in the tabs' labels, e.g. "Lamella"
    item_noun: str
    # emits the newly selected item, or None
    item_selected: Any

    @property
    def selected_item(self) -> Optional[Any]: ...

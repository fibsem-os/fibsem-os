"""What a `ViewInfoBar` shows, read off its labels."""

from typing import List


def visible_fields(bar) -> list:
    """The labelled fields the bar has room for, in order."""
    return [
        item
        for item, label in zip(bar._labelled, bar.field_labels)
        if not label.isHidden()
    ]


def hidden_fields(bar) -> list:
    """The labelled fields dropped for want of room, behind the ``+N`` chip."""
    return [
        item for item, label in zip(bar._labelled, bar.field_labels) if label.isHidden()
    ]


def highlighted_fields(bar) -> List[str]:
    """The keys of the fields lit up now."""
    return list(bar._highlights)

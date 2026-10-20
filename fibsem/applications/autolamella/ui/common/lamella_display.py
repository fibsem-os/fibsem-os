"""How a lamella reads wherever it is listed: its verdict's look and menu, its status
text, its defect icon and menu, and its grid label."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

from PyQt5.QtWidgets import QAction, QLabel, QMenu

from fibsem.applications.autolamella.structures import Lamella
from fibsem.ui import stylesheets
from fibsem.ui.icon import fibsem_icon
from fibsem.ui.tokens import ACCENT_COLOR, NEUTRAL_550
from fibsem.ui.widgets.preflight import format_clock
from fibsem.util.timestamps import now
from fibsem.workflows.quality import QualityRecord, Verdict
from fibsem.workflows.status import TaskState, TaskStatus

# What the lamella displays know about grids: `GridRecord.id` -> (name, loaded).
# Built by the window from the experiment's records and the stage's inventory and
# pushed to every list and card; the rows never read the stage themselves.
GridContext = Dict[str, Tuple[str, bool]]

# The cards' type scale: name / row text / secondary line. The list's rows use
# the workflow panel's (`list_chrome`).
NAME_FONT_PX = 12
DETAIL_FONT_PX = 10


def grid_of(lamella, context: Optional[GridContext]) -> Optional[Tuple[str, bool]]:
    """(grid name, on the stage) for a lamella linked to a grid the context
    knows, else None -- an unlinked lamella, or one whose grid has no record,
    shows no grid and has no stage actions withheld."""
    grid_id = getattr(lamella, "grid_id", None)
    if not grid_id or not context:
        return None
    return context.get(grid_id)


def grids_named(context: Optional[GridContext]) -> bool:
    """Whether the rows name their grid at all: only when the experiment has
    more than one, since with one grid the name says nothing."""
    return bool(context) and len(context) > 1


def not_on_stage_reason(grid: Optional[Tuple[str, bool]]) -> str:
    """Why a stage action is withheld, or "" when it is not."""
    if grid is None or grid[1]:
        return ""
    return f"{grid[0]} is not on the stage"


def apply_grid_label(
    label: QLabel,
    grid: Optional[Tuple[str, bool]],
    shown: bool,
    followed: bool = True,
    font_px: int = DETAIL_FONT_PX,
) -> None:
    """The grid's name ahead of the status: accent while that grid is on the
    stage, muted while it is not; hidden when there is nothing to say. The
    separator is drawn only when a status follows (*followed*)."""
    if grid is None or not shown:
        label.setVisible(False)
        label.setText("")
        return
    name, loaded = grid
    label.setText(f"{name} ·" if followed else name)
    label.setStyleSheet(
        f"font-size: {font_px}px; background: transparent; "
        f"color: {ACCENT_COLOR if loaded else NEUTRAL_550};"
    )
    label.setToolTip(
        f"{name} is on the stage" if loaded else f"{name} is not on the stage"
    )
    label.setVisible(True)


def short_status(text: str) -> str:
    """The task name without its time, for a line with no room for it:
    "Rough Milling · 1:44PM" (`completed_text`) is "Rough Milling"."""
    return text.split(" \u00b7 ", 1)[0]


def has_defect(lamella) -> bool:
    """Whether someone has judged the lamella to need rework or to have failed.

    Not "has any verdict": a lamella judged good has one too, and is not defective.
    """
    defect = getattr(lamella, "defect", None)
    return defect is not None and defect.verdict in (Verdict.REWORK, Verdict.FAILED)


def has_verdict(lamella) -> bool:
    """Whether anyone has judged the lamella at all -- good, rework or failed."""
    defect = getattr(lamella, "defect", None)
    return defect is not None and defect.verdict is not Verdict.UNASSESSED


# What each verdict looks like wherever one is shown or set, for lamellae and grids
# alike: icon, colour, words. Filled circles for a judgement, an empty one for none.
VERDICT_LOOK = {
    Verdict.UNASSESSED: ("mdi:circle-outline", NEUTRAL_550, "Not assessed"),
    Verdict.GOOD: ("mdi:check-circle", stylesheets.GREEN_COLOR, "Good"),
    Verdict.REWORK: ("mdi:alert-circle", stylesheets.DEFECT_ORANGE_COLOR, "Rework"),
    Verdict.FAILED: ("mdi:close-circle", stylesheets.DEFECT_RED_COLOR, "Failed"),
}


def add_verdict_actions(menu: QMenu) -> Dict[Verdict, QAction]:
    """Good, Rework and Failed, then "Not assessed" after a separator, on *menu*.

    The one layout for a verdict menu, used by a lamella's and by a grid's, so
    the two cannot drift apart again: setting a judgement is the common action,
    taking one back the rare one, so it comes last.
    """
    actions: Dict[Verdict, QAction] = {}
    for verdict in (Verdict.GOOD, Verdict.REWORK, Verdict.FAILED, Verdict.UNASSESSED):
        if verdict is Verdict.UNASSESSED:
            menu.addSeparator()
        icon, colour, text = VERDICT_LOOK[verdict]
        action = menu.addAction(fibsem_icon(icon, color=colour), text)
        action.setCheckable(True)
        actions[verdict] = action
    return actions


def mark_current_verdict(actions: Dict[Verdict, QAction], current: Verdict) -> None:
    """Check the current verdict and set it in bold.

    Both, because with an icon on every item the style draws the icon in place of
    a checkmark, so the check alone does not show which is current.
    """
    for verdict, action in actions.items():
        is_current = verdict is current
        action.setChecked(is_current)
        font = action.font()
        font.setBold(is_current)
        action.setFont(font)


def add_defect_menu(menu: QMenu, lamella, on_changed) -> QMenu:
    """A "Verdict" submenu on *menu* writing `lamella.defect`; *on_changed* is
    called after a write. The one place a person sets a lamella's verdict.

    It offered "No defect", which wrote `Verdict.NONE` -- an alias of `UNASSESSED`
    -- so "I looked and it is fine" was stored as "nobody has looked", and there was
    no way to mark a lamella good. Good, rework and failed are offered now, with
    "Not assessed" to take a judgement back.

    The record is replaced rather than edited in place: every display of the
    verdict redraws off `lamella.events.defect`, which only a replacement raises.
    Who set it is filled in by whatever saves the change, from the experiment's
    operator; this menu has no experiment to ask.
    """
    sub = menu.addMenu("Verdict")
    actions = add_verdict_actions(sub)

    for verdict, action in actions.items():

        def _set(_checked=False, verdict=verdict):
            if verdict is lamella.defect.verdict:
                _show_current()  # re-tick: choosing the current one changes nothing
                return
            lamella.defect = QualityRecord(
                verdict=verdict,
                updated_at=None if verdict is Verdict.UNASSESSED else now(),
            )
            on_changed()

        action.triggered.connect(_set)

    def _show_current() -> None:
        mark_current_verdict(actions, lamella.defect.verdict)

    sub.aboutToShow.connect(_show_current)
    _show_current()
    return sub


def completed_text(task: TaskState) -> str:
    """The last task done and when, for a list's status column: "Rough Milling ·
    1:44PM", "· Yesterday, 1:44PM", and further back "· Sep 14, 1:44PM".

    The clock the preflight and timeline use, in the viewer's zone, after a dot.
    Further back than yesterday that clock gives the whole date and time, which
    is long for a column; a short date says as much. `task.completed` ("Rough
    Milling (01:44PM)") is left as it is: the reports and the exported table
    read it.
    """
    when = task.end_timestamp
    if when is None:
        return task.name
    if when.tzinfo is not None:
        when = when.astimezone()
    today = datetime.now(when.tzinfo).date()
    if (today - when.date()).days > 1:
        return f"{task.name} \u00b7 {when:%b} {when.day}, {format_clock(when, None)}"
    return f"{task.name} \u00b7 {format_clock(when, datetime.now(when.tzinfo))}"


def status_look(lamella: Lamella) -> tuple[str, str]:
    """Return (text, stylesheet) for the status column."""
    ts = lamella.task_state
    if ts and ts.status == TaskStatus.InProgress:
        return (
            f"{ts.name}",
            f"color: {stylesheets.PRIMARY_COLOR}; background: transparent;",
        )
    last = lamella.last_completed_task
    if last:
        return completed_text(last), f"color: {NEUTRAL_550}; background: transparent;"
    return "", "background: transparent;"


def defect_icon(lamella: Lamella) -> tuple[str, str, str]:
    """Return (icon_name, icon_color, tooltip) for the defect indicator button."""
    d = lamella.defect
    icon, colour, text = VERDICT_LOOK[d.verdict]
    return (
        icon,
        colour,
        f"{text}{': ' + d.reason if d.reason else ''}",
    )

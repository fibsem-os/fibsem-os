"""Every kind of event the app records, in one place.

An event is a kind plus a payload (`fibsem.workflows.events`). This module lists
the kinds: a constant for each, the group the Events page files it under, and
whether it is live progress (many a second, left out of a run's history). A kind
read only from older files is here too, marked ``legacy``, so a reader of an old
experiment still knows it.

A kind written anywhere must be registered here; the tests fail on one that is
not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet

# The Events page's groups.
PROPOSALS = "proposals"
NOTES = "notes"
REQUESTS = "requests"
WORKFLOW = "workflow"
STAGE = "stage"
IMAGES = "images"
MILLING = "milling"
EDITS = "edits"
LIVE = "live"
OTHER = "other"

GROUPS = (PROPOSALS, NOTES, REQUESTS, WORKFLOW, STAGE, IMAGES, MILLING, EDITS)
GROUPS += (LIVE, OTHER)


@dataclass(frozen=True)
class EventKind:
    name: str
    group: str
    about: str
    # Many a second while milling or imaging: a run's history leaves it out.
    live: bool = False
    # Written by an older version only; still read from its files.
    legacy: bool = False


KINDS: Dict[str, EventKind] = {}


def _kind(name: str, group: str, about: str, **flags: bool) -> str:
    assert name not in KINDS, name
    KINDS[name] = EventKind(name, group, about, **flags)
    return name


def kind_of(name: str) -> EventKind:
    """The registered kind *name*, or an unregistered one in the other group."""
    return KINDS.get(name) or EventKind(name, OTHER, "")


def group_of(name: str) -> str:
    return kind_of(name).group


# --- the run's tasks (`fibsem.hooks.HookEvent`) ---------------------------------
TASK_STARTED = _kind("task_started", WORKFLOW, "A task started on an item.")
TASK_COMPLETED = _kind("task_completed", WORKFLOW, "A task finished on an item.")
TASK_FAILED = _kind("task_failed", WORKFLOW, "A task failed; carries error.")
TASK_CANCELLED = _kind("task_cancelled", WORKFLOW, "A task was stopped.")
TASK_SKIPPED = _kind("task_skipped", WORKFLOW, "A task was skipped, and why.")
TASK_STEP = _kind("task_step", WORKFLOW, "A step inside a running task.")
ITEM_COMPLETED = _kind("item_completed", WORKFLOW, "Every task for an item is done.")
WORKFLOW_START = _kind(
    "workflow_start", WORKFLOW, "The agent asked for a run of these tasks and items."
)
WORKFLOW_STARTED = _kind("workflow_started", WORKFLOW, "A run started.")
WORKFLOW_COMPLETED = _kind("workflow_completed", WORKFLOW, "A run finished.")
WORKFLOW_CANCELLED = _kind("workflow_cancelled", WORKFLOW, "A run was stopped.")
WORKFLOW_STALLED = _kind(
    "workflow_stalled", WORKFLOW, "A run gave up waiting on a decision."
)
WORKFLOW_FAILED = _kind("workflow_failed", WORKFLOW, "A run died; carries error.")
EXPERIMENT_COMPLETED = _kind(
    "experiment_completed", WORKFLOW, "Every task in the experiment is done."
)
QUEUE_CHANGED = _kind("queue_changed", OTHER, "The run's queue changed.")
WORKFLOW_CHANGED = _kind(
    "workflow_changed", EDITS, "A workflow setting changed: field and value."
)
EDIT = _kind("edit", EDITS, "A task config or item record was edited.")
GRIDS_CHANGED = _kind("grids_changed", OTHER, "A grid was renamed.")

# --- proposals and decisions -------------------------------------------------------
PROPOSAL_ASKED = _kind("proposal_asked", PROPOSALS, "A run proposed a value.")
PROPOSAL_DECIDED = _kind("proposal_decided", PROPOSALS, "A proposal was decided.")
PROPOSAL_SUGGESTED = _kind(
    "proposal_suggested", PROPOSALS, "The agent suggested a decision."
)
SUGGESTION_DECLINED = _kind(
    "suggestion_declined", PROPOSALS, "The operator declined the agent's suggestion."
)
PROPOSAL_HELD = _kind("proposal_held", PROPOSALS, "The run waits on a proposal.")
PROPOSAL_RELEASED = _kind(
    "proposal_released", PROPOSALS, "The run no longer waits on a proposal."
)
PROPOSAL_WAITING = _kind(
    "proposal_waiting", PROPOSALS, "A held proposal still waits; who may decide it."
)
PROMPT_RAISED = _kind("prompt_raised", PROPOSALS, "A question.", legacy=True)
PROMPT_ANSWERED = _kind("prompt_answered", PROPOSALS, "An answer.", legacy=True)
PROMPT_SUGGESTED = _kind("prompt_suggested", PROPOSALS, "A suggestion.", legacy=True)
PROMPT_CANCELLED = _kind(
    "prompt_cancelled", PROPOSALS, "A question withdrawn.", legacy=True
)
PROMPT_SUGGESTION_DISMISSED = _kind(
    "prompt_suggestion_dismissed", PROPOSALS, "A suggestion dismissed.", legacy=True
)
QUESTION_ASKED = _kind("question_asked", PROPOSALS, "A question.", legacy=True)
QUESTION_RELEASED = _kind(
    "question_released", PROPOSALS, "A question released.", legacy=True
)
REVIEW_DECIDED = _kind("review_decided", PROPOSALS, "A review.", legacy=True)

# --- actions, notes and messages ---------------------------------------------------
ACTION = _kind("action", NOTES, "Someone acted through the App Server: what, and ok.")
NOTE = _kind("note", NOTES, "A note or message on the record.")
EVIDENCE = _kind("evidence", NOTES, "An image kept as evidence.")
OPERATOR_MESSAGE = _kind("operator_message", NOTES, "The operator wrote to the agent.")
AGENT_NOTE = _kind("agent_note", NOTES, "The agent's note.", legacy=True)
AGENT_STOP = _kind("agent_stop", NOTES, "The agent stopped something.", legacy=True)
AGENT_REFUSED = _kind("agent_refused", NOTES, "A call refused.", legacy=True)
AGENT_CALLED = _kind("agent_called", NOTES, "The agent called the App Server.")
WORKFLOW_STOP = _kind("workflow_stop", NOTES, "A stop.", legacy=True)

# --- the agent and its session -----------------------------------------------------
AGENT_LEVEL_SET = _kind("agent_level_set", OTHER, "The agent level changed.")
AGENT_SESSION_STARTED = _kind(
    "agent_session_started", OTHER, "An agent session started."
)
AGENT_SESSION_ENDED = _kind("agent_session_ended", OTHER, "An agent session ended.")
AGENT_SESSION_BUDGET = _kind(
    "agent_session_budget", OTHER, "The agent has used much of its budget."
)
AGENT_MESSAGES_READ = _kind(
    "agent_messages_read", OTHER, "The agent read the operator's messages."
)
AGENT_PLAN_EDITED = _kind("agent_plan_edited", OTHER, "The agent edited its plan.")
AGENT_SAMPLE_SET = _kind("agent_sample_set", OTHER, "The agent set the sample.")

# --- requests (the agent's actions held for the operator) ----------------------------
REQUEST_RAISED = _kind("request_raised", REQUESTS, "A request.", legacy=True)
REQUEST_DECIDED = _kind("request_decided", REQUESTS, "A request was approved or not.")
REQUEST_APPLIED = _kind("request_applied", REQUESTS, "One of a request's actions ran.")
REQUEST_FINISHED = _kind("request_finished", REQUESTS, "A request finished.")
REQUEST_CANCELLED = _kind("request_cancelled", REQUESTS, "A request was cancelled.")

# --- service commands (`fibsem.devices.runs`) ------------------------------------------
COMMAND_STARTED = _kind("command_started", OTHER, "A long service command started.")
COMMAND_ENDED = _kind("command_ended", OTHER, "A long service command ended.")

# --- stage, beam and devices -----------------------------------------------------------
STAGE_MOVED = _kind("stage_moved", STAGE, "The stage moved: how, from and to.")
BEAM_SHIFTED = _kind("beam_shifted", STAGE, "A beam shift.")
ALIGNMENT = _kind("alignment", STAGE, "An alignment, and its check.")
AUTOFOCUS = _kind("autofocus", STAGE, "An autofocus, and its check.")
FM_AUTOFOCUS = _kind("fm_autofocus", STAGE, "An FM autofocus.")
COINCIDENCE_MEASURED = _kind(
    "coincidence_measured", STAGE, "The FIB-SEM coincidence was measured."
)
OBJECTIVE_STATE_CHANGED = _kind(
    "objective_state_changed", STAGE, "The FM objective moved in or out."
)
GRID_LOADED = _kind("grid_loaded", STAGE, "A grid was loaded.")
GRID_UNLOADED = _kind("grid_unloaded", STAGE, "A grid was unloaded.")
CORRELATION = _kind("correlation", OTHER, "An FM to FIB correlation was saved.")

# --- images and files ------------------------------------------------------------------
IMAGE_ACQUIRED = _kind("image_acquired", IMAGES, "An image was taken.")
FM_IMAGE_ACQUIRED = _kind("fm_image_acquired", IMAGES, "An FM image was taken.")
EXPERIMENT_FILE_ADDED = _kind(
    "experiment_file_added", OTHER, "A file was added to the experiment."
)
EXPERIMENT_FILE_READ = _kind(
    "experiment_file_read", OTHER, "A file in the experiment was read."
)

# --- milling ---------------------------------------------------------------------------
MILLING = _kind("milling", MILLING, "A milling run, and how it ended.")
MILLING_STAGE_STARTED = _kind(
    "milling_stage_started", MILLING, "A milling stage started."
)
SPOT_BURN_STARTED = _kind("spot_burn_started", MILLING, "A spot burn started.")
MILLING_READING = _kind(
    "milling_reading", MILLING, "A milling monitor's notable reading: a trigger."
)

# --- live progress ---------------------------------------------------------------------
MILLING_PROGRESS = _kind("milling_progress", LIVE, "Milling progress.", live=True)
SPOT_BURN_PROGRESS = _kind("spot_burn_progress", LIVE, "Spot burn progress.", live=True)
FM_ACQUISITION_PROGRESS = _kind(
    "fm_acquisition_progress", LIVE, "FM acquisition progress.", live=True
)
FM_ACQUISITION = _kind("fm_acquisition", LIVE, "A live FM frame.", live=True)
TILED_ACQUISITION = _kind(
    "tiled_acquisition", LIVE, "Tiled acquisition progress.", live=True
)
STAGE_POSITION_CHANGED = _kind(
    "stage_position_changed", LIVE, "The stage position, while it moves.", live=True
)
SEM_ACQUISITION = _kind("sem_acquisition", LIVE, "A live SEM frame.", live=True)
FIB_ACQUISITION = _kind("fib_acquisition", LIVE, "A live FIB frame.", live=True)

LIVE_KINDS: FrozenSet[str] = frozenset(k.name for k in KINDS.values() if k.live)

"""The agent's suggestion on a pending proposal, as the card shows it: what the
proposal asks, the decision approving it makes, and why; and the same decision
as a request, for the previews to draw on the proposal's image."""

import os
from typing import Any, Dict, Optional, Tuple

from fibsem.util import wording
from fibsem.workflows.evidence import find as find_evidence
from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_EDIT,
    ACTION_RUN,
    ACTION_SKIP,
    DecisionOutcome,
    Proposal,
    _encode_values,
    kind_label,
)

_ACTIONS = {
    ACTION_RUN: "Run the pass",
    ACTION_EDIT: "Take the edit, run nothing",
    ACTION_SKIP: "Skip the pass",
    ACTION_ACCEPT: "Accept what ran",
}


def front_suggestion(experiment: Any) -> Optional[Tuple[Any, str, Proposal]]:
    """``(item, task_name, proposal)``: the first pending proposal carrying a
    suggestion nobody declined, or None."""
    if experiment is None:
        return None
    for item, task_name, proposal in experiment.pending_proposals():
        suggestion = proposal.suggestion
        if suggestion is not None and not suggestion.declined:
            return item, task_name, proposal
    return None


def decision_text(proposal: Proposal) -> str:
    """What approving the suggestion on ``proposal`` decides, in words."""
    suggestion = proposal.suggestion
    if suggestion.outcome is DecisionOutcome.Rejected:
        negative = proposal.provenance.get("negative")
        return str(negative) if negative else f"Reject: {suggestion.reason}"
    if suggestion.action:
        text = _ACTIONS.get(suggestion.action, suggestion.action)
    else:
        text = str(proposal.provenance.get("positive") or "Continue")
    if suggestion.patch:
        text += f", setting {wording.values_text(suggestion.patch)}"
    elif suggestion.values:
        values = _encode_values(suggestion.values)
        place = values.get("stage_position") if len(values) == 1 else None
        if isinstance(place, dict):
            # A position kept or moved to, not "Continue with stage position
            # name none, ..., coordinate system RAW" (FIB-1492).
            kept = values == _encode_values(proposal.values)
            return f"{'Keep the position' if kept else 'Move to'}: {wording.stage_place(place)}"
        text += f" with {wording.values_text(values)}"
    return text


def _proposal_image(item: Any, proposal: Proposal, experiment_path: Optional[str]):
    """The image the proposal was asked on, as a path, or None: the card's
    picture when the agent cited none."""
    image = proposal.provenance.get("reference_image")
    if not image:
        return None
    image = str(image)
    if not os.path.isabs(image) and experiment_path:
        image = os.path.join(experiment_path, item.name, image)
    return image if os.path.isfile(image) else None


def suggestion_card(
    item: Any,
    task_name: str,
    proposal: Proposal,
    experiment_path: Optional[str] = None,
) -> Dict[str, Any]:
    """The card's suggestion: ``proposal``, ``decision``, ``reason`` and the
    ``evidence`` it cites (its record, or None). The proposal reads by what it
    is, not the task's prompt to the operator ("Press Continue when done"),
    which is the window's (FIB-1492). With no evidence, ``no_evidence`` says
    what the picture is instead."""
    card = {
        "proposal": f"{item.name} · {task_name} · {kind_label(proposal.kind)}",
        "decision": decision_text(proposal),
        "reason": proposal.suggestion.reason,
        "evidence": find_evidence(experiment_path, proposal.suggestion.evidence),
    }
    if card["evidence"] is None and _proposal_image(item, proposal, experiment_path):
        card["no_evidence"] = "No evidence: drawn on the image it was asked on"
    return card


def suggestion_request(
    item: Any, task_name: str, proposal: Proposal, experiment_path: Optional[str]
) -> Dict[str, Any]:
    """The suggestion as the ``proposal_decide`` request that would make its
    decision, for the previews: drawn as a request is."""
    suggestion = proposal.suggestion
    arguments: Dict[str, Any] = {
        "item_id": item.id,
        "task_name": task_name,
        "proposal_id": proposal.id,
        "outcome": suggestion.outcome.name,
    }
    values = suggestion.values
    if not values and suggestion.outcome is DecisionOutcome.Confirmed:
        # confirmed as proposed: the proposal's own values are what is decided
        values = proposal.values
    if values:
        arguments["values"] = _encode_values(values)
    if suggestion.patch:
        arguments["patch"] = dict(suggestion.patch)
    if suggestion.action:
        arguments["action"] = suggestion.action
    # the agent's image only: without one, the picture takes the image the
    # proposal was asked on, not as the agent's
    image = suggestion.image
    if image and experiment_path and not os.path.isabs(image):
        image = os.path.join(experiment_path, image)
    return {
        "request_id": suggestion.id,
        "status": "pending",
        "image": image or None,
        "evidence": find_evidence(experiment_path, suggestion.evidence),
        "actions": [
            {
                "tool": "proposal_decide",
                "what": f"{item.name} · {task_name}",
                "arguments": arguments,
            }
        ],
    }

"""What a loop's pass says on the prompt bar: a mill's, a spot burn's.

A pass is a ``milling`` or ``spot_burn`` proposal (one per pass, see
``fibsem.workflows.proposals``) whose allowed actions depend on where the loop
is: before any pass it is run or skip, after one run again or accept, with
running off only accept. The prompt bar has two buttons, so each pass names
what they do here, and the words come from the kind.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from fibsem.workflows.proposals import (
    ACTION_ACCEPT,
    ACTION_RUN,
    ACTION_SKIP,
    MILLING,
    PROPOSAL_KINDS,
    SPOT_BURN,
    Proposal,
)

__all__ = ["LoopPrompt", "loop_prompt", "LOOP_KINDS"]


@dataclass(frozen=True)
class _Words:
    run: str
    again: str
    noun: str
    without: str
    failed: str
    off: str = "Press Continue when it is right."


_WORDS = {
    MILLING: _Words(
        run="Run Milling",
        again="Mill Again",
        noun="mill",
        without="without milling",
        failed="Milling failed",
        off="Press Continue when the config is right.",
    ),
    SPOT_BURN: _Words(
        run="Run Spot Burn",
        again="Burn Again",
        noun="burn",
        without="without burning",
        failed="Spot burn failed",
    ),
}

LOOP_KINDS = tuple(_WORDS)


@dataclass(frozen=True)
class LoopPrompt:
    """The prompt bar for one pass: its text, its buttons, and the action each
    button decides (None: no such button)."""

    message: str
    positive: str
    negative: Optional[str]
    yes: str
    no: Optional[str]


def loop_prompt(proposal: Proposal) -> LoopPrompt:
    """The prompt bar for ``proposal``, a pass of a loop."""
    words = _WORDS[proposal.kind]
    allowed = PROPOSAL_KINDS[proposal.kind].allowed_actions(proposal)
    passes = int(proposal.provenance.get("passes") or 0)
    message = str(proposal.provenance.get("message") or "")
    if ACTION_RUN not in allowed:
        prompt = LoopPrompt(
            f"{message} {words.off}", "Continue", None, ACTION_ACCEPT, None
        )
    elif passes == 0:
        prompt = LoopPrompt(
            f"{message} Press {words.run} to {words.noun}, or Skip to go on "
            f"{words.without}.",
            words.run,
            "Skip",
            ACTION_RUN,
            ACTION_SKIP,
        )
    else:
        prompt = LoopPrompt(
            f"{message} Press {words.again} to {words.noun} again, or Accept to go on.",
            words.again,
            "Accept",
            ACTION_RUN,
            ACTION_ACCEPT,
        )
    error = proposal.provenance.get("last_error")
    if error:
        prompt = replace(prompt, message=f"{words.failed}: {error}\n\n{prompt.message}")
    return prompt

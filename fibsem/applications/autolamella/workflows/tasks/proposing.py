"""Proposing and settling a finished task's result, for any item a task runs on.

A lamella task and a grid task end the same way: the run is recorded, then its
result is proposed on the item and decided -- by the operator's inline answer,
by the Review tab, or by the producer itself. What differs between them is
supplied by the task (its proposer, whether it is under review, the decision it
took inline, which output roles are its result images), so the two task bases
share this code rather than each keeping a copy.

The item is anything with a ``name``, an ``id``, a ``task_state``, a
``proposals`` dict and ``set_task_status``: a ``Lamella`` or a ``GridRecord``.
"""

import logging
from typing import Any, Dict, Mapping, Optional

from fibsem.applications.autolamella.proposals import (
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
    Proposal,
    Proposer,
    TaskResultProposer,
    auto_author,
)
from fibsem.applications.autolamella.structures import AutoLamellaTaskStatus

# Provenance key -> output role, for a lamella task: the last final ion and
# electron images. The last is the tightest field of view, the one the Review
# tab shows.
LAMELLA_RESULT_IMAGES: Dict[str, str] = {
    "reference_image": "final_fib",
    "reference_image_eb": "final_sem",
}


def propose(
    task: Any,
    item: Any,
    proposer: Optional[Proposer],
    result_images: Mapping[str, str],
    failure: str = "",
) -> Optional[Proposal]:
    """Record what ``task`` did on ``item`` as a proposal, for someone to look at.

    Every proposal is the task's result -- which task, when, how it ended, the
    result images -- and that part is filled in here for every kind. The
    proposer adds the values a decision can edit, if its kind has any; a failed
    task proposes no values, its record is the failure. A re-run supersedes a
    decided proposal like any other; a pending one is replaced. None when the
    task type proposes nothing, or its proposer declines (nothing consumes what
    it would propose).
    """
    if proposer is None:
        return None
    if failure:
        proposer = TaskResultProposer()  # no values from a run that failed
    proposal = proposer.propose(task)
    if proposal is None:
        logging.info(f"{item.name}: {task.task_name} has nothing to propose.")
        return None
    carried = PROPOSAL_KINDS[proposer.kind].values
    unknown = [n for n in proposal.values if n not in carried]
    if unknown:
        # a producer bug, not a record to keep
        raise ValueError(
            f"{proposer.kind} does not carry {unknown}; it carries {carried}"
        )
    state = item.task_state
    outputs = state.outputs
    # The file names are relative to the item's folder; readers join them onto
    # its path, which also survives a moved experiment. The values are in the
    # milling frame, so any image at the stored pose would do.
    images = {
        key: (outputs.get(role) or [""])[-1] for key, role in result_images.items()
    }
    proposal.provenance = {
        "proposer": proposer.name or task.task_name,
        "version": proposer.version,
        "task_name": task.task_name,
        "task_id": state.task_id,
        "status": state.status.name,
        "started_at": state.start_timestamp,
        "ended_at": state.end_timestamp,
        **images,
        "failure": failure,
        **proposal.provenance,
    }
    decided = item.proposal(task.task_name)
    if decided is not None and decided.pending:
        decided = None  # replaced, not kept
    if decided is not None and decided.task_id == state.task_id:
        # Not a re-run: a question this same run asked and had answered
        # (FIB-1025). It stays on the record before the run's own result --
        # but saying "re-run" here would put a run in the log that never
        # happened.
        logging.info(
            f"{item.name}: {task.task_name} asked a question during this run; "
            "its answer is on the record before the run's result."
        )
    elif decided is not None:
        logging.info(
            f"{item.name}: {task.task_name} re-run; the decided "
            "proposal stays on the record and a new one is pending."
        )
    item.record_proposal(task.task_name, proposal)
    logging.info(
        {
            "msg": "proposal_recorded",
            "lamella": item.name,
            "task_name": task.task_name,
            "kind": proposal.kind,
            "values": {
                k: getattr(v, "to_dict", lambda: v)()
                for k, v in proposal.values.items()
            },
            "provenance": proposal.provenance,
        }
    )
    return proposal


def settle(
    task: Any,
    item: Any,
    *,
    proposer: Optional[Proposer],
    result_images: Mapping[str, str],
    review: bool,
    inline_decision: Optional[Decision],
    experiment: Any,
    failure: str = "",
) -> None:
    """Record the proposal and decide it, once the run is over and the outcome
    recorded. After post_task on purpose: Experiment.decide refuses a decision
    on a task in progress.

    The decision, in every mode, goes through Experiment.decide -- same lock,
    same thread, same write-through, one place that appends. Which decision:
    the operator's inline answer when the task asked one (supervised; recorded
    without a second write-through of a value that is already applied), else
    under review none -- a task that completed moves to AwaitingDecision and the
    decision in the Review tab finishes it -- else the producer's own
    confirmation, as proposed, so nothing downstream waits. A failed task stays
    Failed; its record waits for someone to look, but no decision changes the
    outcome.
    """
    proposal = propose(task, item, proposer, result_images, failure=failure)
    if proposal is None:
        return
    decision = inline_decision
    if decision is None:
        if review:
            if item.task_state.status is AutoLamellaTaskStatus.Completed:
                item.set_task_status(
                    task.task_name, AutoLamellaTaskStatus.AwaitingDecision
                )
                logging.info(
                    f"{item.name}: {task.task_name} awaits a decision "
                    "in the Review tab."
                )
            return
        decision = Decision(
            outcome=DecisionOutcome.Confirmed,
            author=auto_author(proposal.provenance["proposer"]),
            values=dict(proposal.values),
            via="workflow",
        )
    if experiment is None:
        return
    # The producer decides the proposal it has just made: its own run.
    decision.task_id = proposal.task_id
    decision.proposal_id = proposal.id
    try:
        result = experiment.decide(item.id, task.task_name, decision)
    except Exception:
        logging.exception(
            f"{item.name}: could not record the decision on the "
            f"{task.task_name} proposal; it is left pending, so its consumer "
            "will wait."
        )
        return
    if not result.applied:
        logging.warning(
            f"{item.name}: the decision on the {task.task_name} "
            f"proposal was not recorded ({result.reason}); it is left pending."
        )

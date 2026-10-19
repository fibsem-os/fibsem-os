"""Someone answering a run's questions, for tests that run supervised tasks
with no window.

A supervised question waits for an answer, whoever is there, nobody included,
so a headless test that runs one has to answer it.
``answering`` does what a reviewer who agrees with everything would: each
question a task stops on is confirmed as proposed, a loop's pass with the
action its kind takes when nobody is asked.
"""

from contextlib import contextmanager
from typing import Iterator, List

from fibsem.workflows.proposals import PROPOSAL_KINDS, Decision, DecisionOutcome

ANSWERER = "operator:test"


@contextmanager
def answering(experiment, actor: str = ANSWERER) -> Iterator[List]:
    """Confirm, as proposed, each question asked while the block runs.
    Yields the proposals answered, in order."""
    answered: List = []

    def on_asked(item_id: str, task_name: str) -> None:
        item = experiment.get_item_by_id(item_id)
        proposal = item.proposal(task_name) if item is not None else None
        if proposal is None or not proposal.asking:
            return
        kind = PROPOSAL_KINDS[proposal.kind]
        action = kind.silent_action(proposal)
        result = experiment.decide(
            item_id,
            task_name,
            Decision(
                outcome=DecisionOutcome.Confirmed,
                actor=actor,
                values={} if action else dict(proposal.values),
                proposal_id=proposal.id,
                action=action,
            ),
        )
        assert result.applied, result.reason
        answered.append(proposal)

    experiment.asked.connect(on_asked)
    try:
        yield answered
    finally:
        experiment.asked.disconnect(on_asked)

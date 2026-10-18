"""Deciding what tasks proposed: the one way a decision reaches an experiment.

:class:`ProposalReview` is what an experiment does with the proposals on its
items -- decide, ask, apply as proposed, expire, record unasked, withdraw --
and the two lists the review inbox shows. It reads the items through
``items()`` and ``get_item_by_id``, saves through ``save``, and fires the
experiment's ``decided`` and ``asked``; which item field a confirmed value is
written to is the value writer's, registered per proposal kind
(``register_value_writer``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from fibsem.acting import Actor
from fibsem.workflows.experiment import EXPERIMENT_WRITE_LOCK
from fibsem.workflows.proposals import (
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
    DecisionResult,
    Proposal,
    Standing,
    ValueRefused,
    _encode_values,
    _quietly,
    prepare_values,
    rejected_fails_task,
)
from fibsem.workflows.status import TaskStatus


def _call_on_main_thread(func, *args, **kwargs):
    """Run ``func`` on the Qt main thread and wait for its result, when there
    is a Qt application to have one. Without one (a script, a headless
    review, the test job that installs no Qt) it is a plain call.

    Lazy on purpose: superqt and PyQt are in the ``ui`` extra, and this module
    is imported everywhere.

    What ``func`` raises is raised here, in the caller, as a plain call would
    (FIB-1352). Left to escape on the main thread it would be raised inside a
    Qt slot, where PyQt5 aborts the process, and the caller would still be
    waiting for a result that never comes.
    """
    try:
        from PyQt5.QtCore import QCoreApplication, QThread
        from superqt import ensure_main_thread
    except ImportError:
        return func(*args, **kwargs)
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is app.thread():
        return func(*args, **kwargs)

    def carried():
        try:
            return func(*args, **kwargs), None
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            return None, e

    result, error = ensure_main_thread(await_return=True)(carried)()
    if error is not None:
        raise error
    return result


def _same_values(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether two value sets say the same thing, compared in their file form
    so a Point and a re-read Point agree."""
    try:
        return _encode_values(a) == _encode_values(b)
    except Exception:  # noqa: BLE001 - a value that cannot be encoded is a change
        return False


def standing(item: Any, task_name: str, proposal: Proposal) -> Standing:
    """The one lifecycle, read off the item: see ``Standing``. Pending is
    one of three by who waits; a task's own question and a supervised task's
    held result are on the proposal and the item, and a pending proposal
    with neither is open. Decided is unchecked until a person looks."""
    if proposal.pending:
        if proposal.asking:
            return Standing.Asking
        if item.is_awaiting_decision(task_name):
            return Standing.Awaiting
        return Standing.Open
    return Standing.Unchecked if proposal.to_check else Standing.Closed


def _emit_on_main_thread(signal, *args) -> None:
    """Deliver ``signal`` on the Qt main thread without waiting for it.

    The waiting is the difference from ``_call_on_main_thread``. A caller that
    needs the result has to block; one that is only telling the GUI something
    happened must not, because it may be the workflow thread about to park on
    an answer -- and the main thread is not always free to run the call back.
    """
    try:
        from PyQt5.QtCore import QCoreApplication, QThread
        from superqt import ensure_main_thread
    except ImportError:
        signal.emit(*args)
        return
    app = QCoreApplication.instance()
    if app is None or QThread.currentThread() is app.thread():
        signal.emit(*args)
        return
    ensure_main_thread(await_return=False)(signal.emit)(*args)


class ProposalReview:
    """Decisions on an experiment's proposals. A mixin for an experiment: it
    needs ``items()``, ``get_item_by_id``, ``save`` and the ``decided`` and
    ``asked`` signals."""

    def decide(
        self, item_id: str, task_name: str, decision: Decision
    ) -> DecisionResult:
        """The one way a decision reaches the experiment.

        Runs on the Qt main thread when there is one and blocks the caller until
        it has: the Review tab and the App Server are the same client of this
        function, and a confirm can end in Qt work (a generative review adds
        lamellae, whose list rebuild must not fire off the GUI thread). Takes
        the same lock as ``save`` so a save never serialises a half-applied
        decision.

        Confirmed: the decision is appended and each decided value is written
        through to the item (``poi`` moves the point and syncs the patterns
        that follow it). No value is written through to a grid; a confirm
        carrying values on one is refused. The proposed values are left as they were, so the
        delta survives. A task that ended AwaitingDecision is finished by the
        decision: Completed on confirm, Failed on reject when the kind says a
        reject fails the task (a ``state`` reject is "don't", and leaves it as
        it was), so what requires it runs or does not by the ordinary
        prerequisite rule. The same for a
        lamella and a grid. A decision on a
        task that already finished (a result someone checks) changes nothing
        but the record.

        Refused, without a write, when there is no such proposal; when the run
        being decided is the run in progress (a decision then is a stop, not a
        decision); or when a decision carrying values arrives while any task
        runs on the item. Looking at what an earlier run did, while a later one
        runs, is not refused: it writes nothing -- including an earlier run of
        the task that is running now.
        """
        return _call_on_main_thread(self._decide, item_id, task_name, decision)

    def _decide(
        self, item_id: str, task_name: str, decision: Decision
    ) -> DecisionResult:
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return DecisionResult(
                    applied=False, reason=f"No item with id {item_id!r}."
                )
            proposal = item.proposal(task_name)
            if proposal is None:
                return DecisionResult(
                    applied=False,
                    reason=f"{item.name} has no proposal from {task_name!r}.",
                )
            if decision.proposal_id and decision.proposal_id != proposal.id:
                # The task made more than one kind of proposal -- a question it
                # asked, then its result -- and the decider named the other
                # one. Any current proposal can be decided; a replaced one is
                # refused further down as stale.
                for other in item.current_proposals(task_name):
                    if other.id == decision.proposal_id:
                        proposal = other
                        break
            if proposal.withdrawn:
                # The question was taken back because whatever asked it is
                # gone. There is nothing left to answer, and an answer now
                # would be written against a run that is over.
                return DecisionResult(
                    applied=False,
                    error_type="stale_review",
                    reason=f"{task_name} on {item.name} was withdrawn before it "
                    f"was answered; re-run {task_name} to be asked again.",
                )
            # A decision while something runs on the item is refused only where
            # it could reach that run: the proposal is from the run in progress
            # (the answer there is Stop, not a decision), or the decision writes
            # values the running task may be reading. The run, not the task
            # name -- a task that has re-run is still running when its earlier
            # result is looked at, and that look writes nothing. A grid's tasks
            # run back to back, so anything stricter refuses the ordinary case
            # (FIB-1008).
            running = item.task_state.status is TaskStatus.InProgress
            # An open value was written through when its task ended. Confirming
            # it as it stands writes nothing again, so it is a look and lands
            # even while the item is busy with another task; a changed value
            # is a write and waits.
            unchanged_open = (
                standing(item, task_name, proposal) is Standing.Open
                and bool(decision.values)
                and _same_values(decision.values, proposal.values)
            )
            # The one case a decision may land on a running task: the task is
            # parked on this very proposal, waiting to be told the answer
            # (FIB-1025). The hazard both refusals below guard against is a
            # decision arriving *unasked* while a task may be reading those
            # values -- and a task stopped on a future is not reading, it is
            # waiting. Phrased about the waiting rather than about the values,
            # so a question that carries none is allowed on the same grounds.
            # Per proposal, not per item: another proposal on this item is not
            # what the task is waiting for, and the task will resume and may
            # read it, so it stays refused.
            answering_the_run = (
                running
                and proposal.asking
                and proposal.task_id == item.task_state.task_id
            )
            if running and not answering_the_run:
                if proposal.task_id == item.task_state.task_id:
                    return DecisionResult(
                        applied=False,
                        running=True,
                        reason=f"{item.name} is running {task_name}; "
                        "stop it rather than deciding under it.",
                    )
                if decision.values and not unchanged_open:
                    return DecisionResult(
                        applied=False,
                        running=True,
                        reason=f"{item.name} is running {item.task_state.name}; "
                        f"stop it before writing {sorted(decision.values)} through.",
                    )
            if decision.outcome is DecisionOutcome.Rejected and not decision.reason:
                return DecisionResult(applied=False, reason="A reject needs a reason.")
            if decision.outcome in (
                DecisionOutcome.Withdrawn,
                DecisionOutcome.Unreviewed,
            ):
                # Not something a decider says. Withdrawn is what the record
                # shows when whatever asked is gone; Unreviewed, when nobody
                # was asked. Only the record writes either.
                return DecisionResult(
                    applied=False,
                    error_type="invalid_value",
                    reason=f"{decision.outcome.name} is not a decision: confirm "
                    "or reject. It is what the record says when nobody did.",
                )
            # The decision is on the result the decider saw. Named by the
            # proposal when the decider can: a task may ask several questions
            # in one run, they share its task_id, and only the proposal's own
            # id tells the one that was shown from the one that replaced it.
            if decision.proposal_id:
                if decision.proposal_id != proposal.id:
                    # Which of the two it was, when the decider also said what
                    # run it saw: a different run is a re-run, the same one is
                    # a task that has asked again since.
                    what = f"{task_name} on {item.name}"
                    if not decision.task_id:
                        changed = f"{what} is not what you looked at any more"
                    elif decision.task_id != proposal.task_id:
                        changed = f"{what} has re-run since you looked"
                    else:
                        changed = f"{what} has asked again since you looked"
                    return DecisionResult(
                        applied=False,
                        error_type="stale_review",
                        reason=f"{changed}; look at the current one and decide that.",
                    )
                if not decision.task_id:
                    # The run is still worth having on the decision, as a fact
                    # about where the proposal came from.
                    decision.task_id = proposal.task_id
            # By the run otherwise, as every caller did before proposals had
            # ids: the run it names must be the run the proposal is from.
            elif not decision.task_id:
                return DecisionResult(
                    applied=False,
                    error_type="missing_field",
                    reason="A decision names what it decides: pass the "
                    "proposal_id of the proposal you looked at.",
                )
            elif decision.task_id != proposal.task_id:
                return DecisionResult(
                    applied=False,
                    error_type="stale_review",
                    reason=(
                        f"{task_name} on {item.name} has re-run since you looked; "
                        "look at the current result and decide that."
                        if proposal.task_id
                        else f"{task_name} on {item.name} was proposed before runs "
                        "were named; re-run it to decide it."
                    ),
                )
            # An acknowledgement cannot become an edit: a proposal that already
            # has a decision is looked at, not changed (changing a value is a
            # re-run). A pending proposal is confirmed with its values, as
            # proposed or adjusted, never with none.
            if decision.outcome is DecisionOutcome.Confirmed:
                if not proposal.pending and decision.values:
                    return DecisionResult(
                        applied=False,
                        error_type="invalid_value",
                        reason=f"{task_name} on {item.name} is already decided; "
                        "confirming it again records a look and carries no values. "
                        f"To change a value, re-run {task_name}.",
                    )
            apply_values = None
            if decision.outcome is DecisionOutcome.Confirmed and not unchanged_open:
                # All or nothing: every value is checked and every write planned
                # before the decision is appended, so a refusal -- or a planning
                # error -- leaves the record, the item and the task as they were.
                try:
                    apply_values = prepare_values(
                        self, item, proposal.kind, decision.values
                    )
                except ValueRefused as e:
                    return DecisionResult(
                        applied=False, error_type="invalid_value", reason=str(e)
                    )
                except Exception as e:
                    logging.exception(
                        f"{item.name}: could not plan the decision on {task_name}"
                    )
                    return DecisionResult(
                        applied=False, reason=f"Could not apply the values: {e}"
                    )
                carried = (
                    PROPOSAL_KINDS[proposal.kind].values
                    if proposal.kind in PROPOSAL_KINDS
                    else ()
                )
                missing = [n for n in carried if n not in decision.values]
                # A question the task is parked on may be confirmed with no
                # values at all: "as it stands". The task, which can read the
                # instrument, fills the decision in once it is released
                # (``fill_in_decision``), so nothing here has to be told what
                # the values are to confirm them.
                as_it_stands = proposal.asking and not decision.values
                if proposal.pending and missing and not as_it_stands:
                    return DecisionResult(
                        applied=False,
                        error_type="invalid_value",
                        reason=f"Confirming a pending {proposal.kind} proposal "
                        f"needs its values {missing}, as proposed or adjusted.",
                    )

            # Apply, then commit. The writes and the status move are assignments
            # on evented records, whose subscribers can raise; if anything does,
            # every one is undone and the decision is not appended, so a failed
            # decision leaves the item, the task and the record as they were.
            result = DecisionResult(applied=True)
            statuses = [
                (state, state.status, state.status_message)
                for state in (
                    next(
                        (t for t in reversed(item.task_history) if t.name == task_name),
                        None,
                    ),
                    item.task_state,
                )
                if state is not None
            ]
            try:
                if apply_values is not None:
                    result.synced_tasks.extend(apply_values.apply())
                if item.is_awaiting_decision(task_name):
                    if decision.outcome is DecisionOutcome.Confirmed:
                        item.set_task_status(task_name, TaskStatus.Completed)
                    elif rejected_fails_task(proposal.kind):
                        item.set_task_status(
                            task_name,
                            TaskStatus.Failed,
                            f"Rejected by {decision.actor.label}: {decision.reason}",
                        )
            except Exception as e:
                logging.exception(
                    f"{item.name}: applying the decision on {task_name} failed; "
                    "undoing it."
                )
                try:
                    if apply_values is not None:
                        apply_values.undo()
                    for state, status, message in statuses:
                        with _quietly(state):
                            state.status = status
                            state.status_message = message
                except Exception:
                    logging.exception(
                        f"{item.name}: could not undo the decision on {task_name}"
                    )
                return DecisionResult(
                    applied=False, reason=f"Could not apply the decision: {e}"
                )
            proposal.decisions.append(decision)
            # Answered: nothing is waiting on it any more, so it stops being
            # the exception that lets a decision land on a running task. The
            # asker clears this too on its way out; here it is closed the
            # instant the answer lands, leaving no window in between.
            proposal.asking = False
            if apply_values is not None:
                result.delta = proposal.delta(decision)
            logging.debug(
                {
                    "msg": "proposal_decided",
                    "item": item.name,
                    "task_name": task_name,
                    "outcome": decision.outcome.name,
                    "actor": str(decision.actor),
                    "delta": {
                        k: getattr(v, "to_dict", lambda: v)()
                        for k, v in result.delta.items()
                    },
                }
            )
        # Committed: a subscriber that raises is logged, and does not turn a
        # decision that landed into an error for the decider.
        try:
            self.decided.emit(item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return result

    def ask_proposal(self, item_id: str, task_name: str, proposal: Proposal) -> bool:
        """Record a question the task is about to park on, and say so.

        Marks the proposal as the one being waited on, so a decision may land
        on it while its task runs (FIB-1025), and fires ``asked`` so the inbox
        re-derives -- nothing else would, since the tab refreshes when a task
        *finishes* and this one is only halfway through. ``asking`` is the
        live flag and is not saved; ``provenance["asked"]`` is the fact, and
        is, so a load can tell a question the task was waiting on from a
        value left open (FIB-1046).

        Unlike ``decide`` this does **not** run on the main thread. It is
        called from the workflow thread by a responder whose contract is not to
        block, with the task about to park on a future; waiting for the main
        thread there is a deadlock whenever that thread is not free to run the
        call back. It does not need to: ``proposals`` is a plain dict with no
        listeners, so the write is safe under the lock that guards every other
        write, and only the notification is handed to the GUI thread.
        """
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return False
            # Same rule as a task's own proposal: a decided one stays on the
            # record before the new question, a pending one is replaced -- it
            # was never answered, so there is nothing to keep.
            proposal.provenance["asked"] = True
            item.record_proposal(task_name, proposal)
            proposal.asking = True
        try:
            _emit_on_main_thread(self.asked, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to asked raised for {task_name}")
        return True

    def apply_proposed(
        self, item_id: str, task_name: str, proposal_id: str
    ) -> DecisionResult:
        """Write a proposal's values through as they stand, with no decision:
        what an automated task's value being live from the moment it is
        proposed means. On the main thread, through the same planned writes a
        decision uses; nothing is appended to the record, so the proposal
        stays open to correct until its consumer starts."""
        return _call_on_main_thread(
            self._apply_proposed, item_id, task_name, proposal_id
        )

    def _apply_proposed(
        self, item_id: str, task_name: str, proposal_id: str
    ) -> DecisionResult:
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return DecisionResult(
                    applied=False, reason=f"No item with id {item_id!r}."
                )
            proposal = next(
                (p for p in item.current_proposals(task_name) if p.id == proposal_id),
                None,
            )
            if proposal is None:
                return DecisionResult(
                    applied=False, reason=f"{item.name} has no such proposal."
                )
            if not proposal.values:
                return DecisionResult(applied=True)
            try:
                writes = prepare_values(self, item, proposal.kind, proposal.values)
            except ValueRefused as e:
                return DecisionResult(
                    applied=False, error_type="invalid_value", reason=str(e)
                )
            except Exception as e:
                logging.exception(
                    f"{item.name}: could not plan the proposed {task_name} values"
                )
                return DecisionResult(
                    applied=False, reason=f"Could not apply the values: {e}"
                )
            result = DecisionResult(applied=True)
            try:
                result.synced_tasks.extend(writes.apply())
            except Exception as e:
                logging.exception(
                    f"{item.name}: applying the proposed {task_name} values failed; "
                    "undoing."
                )
                try:
                    writes.undo()
                except Exception:
                    logging.exception(f"{item.name}: could not undo {task_name}")
                return DecisionResult(
                    applied=False, reason=f"Could not apply the values: {e}"
                )
            logging.debug(
                {
                    "msg": "proposal_applied_as_proposed",
                    "item": item.name,
                    "task_name": task_name,
                    "kind": proposal.kind,
                    "proposal_id": proposal.id,
                    "synced_tasks": list(result.synced_tasks),
                }
            )
            self.save()
        return result

    def expire_open(self, item_id: str, task_name: str, reason: str) -> int:
        """Close every open proposal from ``task_name`` on the item as
        ``Unreviewed``, carrying the values as proposed: the task that
        consumes them has started before anyone looked, or the task re-ran.
        Under the write lock, so a decision landing at the same moment either
        got there first or is refused as a late edit. Returns how many.

        Only those two moments close an open proposal. A run's end does not:
        a value is for the task that uses it, whichever run that is, and a
        result is consumed by nothing, so it stays open until a person looks.
        """
        expired = 0
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return 0
            for proposal in item.current_proposals(task_name):
                if standing(item, task_name, proposal) is not Standing.Open:
                    continue
                proposal.decisions.append(
                    Decision(
                        outcome=DecisionOutcome.Unreviewed,
                        actor=Actor.task(
                            str(proposal.provenance.get("proposer") or task_name)
                        ),
                        values=dict(proposal.values),
                        reason=reason,
                        via="workflow",
                        task_id=proposal.task_id,
                        proposal_id=proposal.id,
                    )
                )
                expired += 1
                logging.debug(
                    {
                        "msg": "proposal_unreviewed",
                        "item": item.name,
                        "task_name": task_name,
                        "kind": proposal.kind,
                        "proposal_id": proposal.id,
                        "reason": reason,
                    }
                )
        if expired:
            try:
                _emit_on_main_thread(self.decided, item_id, task_name)
            except Exception:
                logging.exception(f"a subscriber to decided raised for {task_name}")
        return expired

    def expire_all_open(self, reason: str) -> int:
        """Close every open proposal on every item. A supervised task's
        result the run is holding for is not open, and is left for the
        decision it waits on."""
        expired = 0
        for item in self.items():
            for task_name in item.asking_tasks():
                expired += self.expire_open(item.id, task_name, reason)
        return expired

    def record_unasked(
        self, item_id: str, task_name: str, proposal: Proposal, reason: str
    ) -> bool:
        """Put a question nobody was asked on the record: the proposal, and an
        ``Unreviewed`` decision on it carrying the values the task went on
        to use. What an automated task leaves behind for someone to check.

        Written by the task, on its own thread, the way ``ask_proposal`` is:
        the write is under the lock every writer takes, and only the
        notification goes to the GUI thread. Not through ``decide``, which
        refuses ``Unreviewed`` from a decider and would write the values
        through, when the task applies them itself.
        """
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return False
            item.record_proposal(task_name, proposal)
            proposal.decisions.append(
                Decision(
                    outcome=DecisionOutcome.Unreviewed,
                    actor=Actor.task(
                        str(proposal.provenance.get("proposer") or task_name)
                    ),
                    values=dict(proposal.values),
                    reason=reason,
                    via="workflow",
                    task_id=proposal.task_id,
                    proposal_id=proposal.id,
                )
            )
        logging.debug(
            {
                "msg": "proposal_unreviewed",
                "lamella": item.name,
                "task_name": task_name,
                "kind": proposal.kind,
                "proposal_id": proposal.id,
                "reason": reason,
            }
        )
        try:
            _emit_on_main_thread(self.decided, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return True

    def fill_in_decision(
        self, item_id: str, task_name: str, proposal_id: str, values: Dict[str, Any]
    ) -> bool:
        """Put ``values`` on a confirmation that carried none: what a task
        read from the instrument once the operator said "as it stands"
        (``AutoLamellaTask.ask`` with ``decided``). Written by the task on its
        own thread, under the write lock; only the notification goes to the
        GUI thread, as ``ask_proposal`` does. False when the proposal is not
        current, is not confirmed, or already carries values."""
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return False
            proposal = next(
                (p for p in item.current_proposals(task_name) if p.id == proposal_id),
                None,
            )
            decision = proposal.current if proposal is not None else None
            if (
                decision is None
                or decision.outcome is not DecisionOutcome.Confirmed
                or decision.values
            ):
                return False
            decision.values = dict(values)
        try:
            _emit_on_main_thread(self.decided, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return True

    def withdraw_what_was_asked(self, reason: str) -> int:
        """Close every question a task was waiting on that is still pending:
        nothing is waiting on it any more. A question exists only while its
        task waits; Stop and a failing task withdraw it, but a process that
        ends without unwinding leaves it on disk as pending, and on the next
        load it would read as a value to decide (FIB-1046). Called by
        ``load``; the second load finds nothing pending and appends nothing.
        Returns how many."""
        withdrawn = 0
        with EXPERIMENT_WRITE_LOCK:
            for item in self.items():
                for proposal in item.proposals:
                    if not proposal.pending or not proposal.provenance.get("asked"):
                        continue
                    proposal.asking = False
                    proposal.decisions.append(
                        Decision(
                            outcome=DecisionOutcome.Withdrawn,
                            actor=Actor.task("workflow"),
                            reason=reason,
                            via="workflow",
                            task_id=proposal.task_id,
                            proposal_id=proposal.id,
                        )
                    )
                    withdrawn += 1
                    logging.debug(
                        {
                            "msg": "proposal_withdrawn",
                            "item": item.name,
                            "asked_by": str(proposal.asked_by),
                            "reason": reason,
                        }
                    )
        return withdrawn

    def withdraw_proposal(
        self, item_id: str, task_name: str, reason: str
    ) -> DecisionResult:
        """Close a proposal nobody answered, because whatever asked it is gone.

        Not a decision, and deliberately not routed through ``decide``: a
        question is withdrawn exactly when its task is failing or the run is
        stopping, which is the state ``decide`` refuses a decision in. The
        record shape is shared -- a ``Decision`` with outcome ``Withdrawn``,
        authored automatically -- because the fact that a question was raised
        and abandoned belongs in the same log as the answers.

        It never touches a task's status. The task has its own ending, and the
        withdrawal is a consequence of it rather than a cause: what requires
        that task is gated on the task, not on this.

        Like ``ask_proposal`` and unlike ``decide``, this does **not** run on
        the main thread. It is called while a run is unwinding -- an abort, a
        failing task -- which is exactly when the main thread may be waiting
        on the workflow thread. Waiting for it here stalls for the marshal's
        timeout and then raises out of the abort path, while the queued call
        still lands later. The write is an append to a plain list under the
        write lock; only the notification is handed to the GUI thread.
        """
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            if item is None:
                return DecisionResult(
                    applied=False, reason=f"No item with id {item_id!r}."
                )
            proposal = item.proposal(task_name)
            if proposal is None:
                return DecisionResult(
                    applied=False,
                    reason=f"{item.name} has no proposal from {task_name!r}.",
                )
            if not proposal.pending:
                # Answered, or already withdrawn. Either way there is nothing
                # open to take back, and appending would bury the answer.
                return DecisionResult(
                    applied=False,
                    reason=f"{task_name} on {item.name} is already decided.",
                )
            proposal.asking = False
            proposal.decisions.append(
                Decision(
                    outcome=DecisionOutcome.Withdrawn,
                    actor=Actor.task("workflow"),
                    reason=reason,
                    via="workflow",
                    task_id=proposal.task_id,
                    proposal_id=proposal.id,
                )
            )
            logging.debug(
                {
                    "msg": "proposal_withdrawn",
                    "item": item.name,
                    "task_name": task_name,
                    "reason": reason,
                }
            )
        try:
            _emit_on_main_thread(self.decided, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return DecisionResult(applied=True)

    def pending_proposals(
        self,
    ) -> List[Tuple[Any, str, Proposal]]:
        """Every proposal something waits on -- the task asking it, or the
        consumers of a held result -- in item order: the review inbox.
        Derived, never stored, so it is the same list from the GUI and the
        server."""
        return self._with_standing(Standing.Asking, Standing.Awaiting)

    def proposals_to_check(
        self,
    ) -> List[Tuple[Any, str, Proposal]]:
        """Every proposal nobody was asked about and nobody has looked at: an
        automated task's value or result, open to correct until its consumer
        starts and unchecked after; a decision an agent made. The inbox's
        second group. Derived like the first."""
        return self._with_standing(Standing.Open, Standing.Unchecked)

    def _with_standing(self, *standings: Standing) -> List[Tuple[Any, str, Proposal]]:
        listed = []
        for item in self.items():
            for task_name in item.asking_tasks():
                for proposal in item.current_proposals(task_name):
                    if standing(item, task_name, proposal) in standings:
                        listed.append((item, task_name, proposal))
        return listed

"""Deciding what tasks proposed: the one way a decision reaches an experiment.

:class:`ProposalDecisions` is what an experiment does with the proposals on its
items -- ask, decide, record unasked, withdraw -- and the list of proposals
tasks are waiting on. It reads the items through
``items()`` and ``get_item_by_id``, saves through ``save``, and fires the
experiment's ``decided`` and ``asked``; which item field a confirmed value is
written to is the value writer's, registered per proposal kind
(``register_value_writer``).
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from fibsem.acting import Actor
from fibsem.workflows.proposals import (
    EXPERIMENT_WRITE_LOCK,
    PROPOSAL_KINDS,
    Decision,
    DecisionOutcome,
    DecisionResult,
    Proposal,
    Suggestion,
    ValueRefused,
    _quietly,
    kind_label,
    prepare_values,
)
from fibsem.workflows.status import TaskStatus


def _log_failure(outcome) -> None:
    error = outcome.exception()
    if error is not None:
        logging.error("a subscriber to a proposal signal raised", exc_info=error)


class ProposalDecisions:
    """Decisions on an experiment's proposals. A mixin for an experiment: it
    needs ``items()``, ``get_item_by_id``, ``save`` and the ``decided`` and
    ``asked`` signals."""

    # Where the experiment is edited: the session's ``run_edit`` once a session
    # adopts the experiment (the window's GUI thread, or a lock). None: here.
    edit_executor: Optional[Callable[[Callable[[], Any]], Any]] = None

    def _run_edit(self, func, *args):
        """``func(*args)`` where the experiment is edited, waiting for it.
        What it raises is raised here, in the caller (FIB-1352)."""
        executor = self.edit_executor
        if executor is None:
            return func(*args)
        return executor(functools.partial(func, *args)).result()

    def _notify(self, signal, *args) -> None:
        """Deliver ``signal`` where the experiment is edited, without waiting:
        the caller may be the workflow thread about to park on an answer, and
        the GUI thread is not always free to run the call back."""
        executor = self.edit_executor
        if executor is None:
            signal.emit(*args)
            return
        outcome = executor(functools.partial(signal.emit, *args))
        outcome.add_done_callback(_log_failure)

    def decide(
        self, item_id: str, task_name: str, decision: Decision
    ) -> DecisionResult:
        """The one way a decision reaches the experiment.

        Runs where the experiment is edited (``edit_executor``) and blocks the
        caller until it has: the window and the App Server are the same client of this
        function, and a confirm can end in Qt work (a write that changes the
        item list, whose rebuild must not fire off the GUI thread). Takes
        the same lock as ``save`` so a save never serialises a half-applied
        decision.

        Confirmed: the decision is appended and each decided value is written
        through to the item (``poi`` moves the point and syncs the patterns
        that follow it). No value is written through to a grid; a confirm
        carrying values on one is refused. The proposed values are left as
        they were, so the delta survives. A decision never moves a task's
        status: a question's decision releases the task asking it, which
        carries on (or fails, on a reject that fails it).

        Refused, without a write, when there is no such proposal; when it is
        already decided (to change a value, re-run the task); when the run
        being decided is the run in progress (a decision then is a stop, not a
        decision); or when a decision carrying values arrives while any task
        runs on the item.
        """
        return self._run_edit(self._decide, item_id, task_name, decision)

    def _decide(
        self, item_id: str, task_name: str, decision: Decision, check_only=False
    ) -> DecisionResult:
        """``check_only``: whether it would land, writing nothing."""
        if item_id is None:
            return self._decide_on_experiment(decision, check_only)
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
                # The task asked more than one kind in this run (a state, then a
                # point of interest) and the decider named the other one. Any
                # current proposal can be decided; a replaced one is refused
                # further down as stale.
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
                    error_type="stale_proposal",
                    reason=f"{task_name} on {item.name} was withdrawn before it "
                    f"was answered; re-run {task_name} to be asked again.",
                )
            if not proposal.pending:
                # Decided already: by a person, an agent, or the record when
                # nobody was asked. Changing a value is a re-run.
                return DecisionResult(
                    applied=False,
                    error_type="invalid_value",
                    reason=f"{task_name} on {item.name} is already decided. "
                    f"To change a value, re-run {task_name}.",
                )
            # A decision while something runs on the item is refused only where
            # it could reach that run: the proposal is from the run in progress
            # (the answer there is Stop, not a decision), or the decision writes
            # values the running task may be reading. The run, not the task
            # name. A grid's tasks run back to back, so anything stricter
            # refuses the ordinary case (FIB-1008).
            running = item.task_state.status is TaskStatus.InProgress
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
                if decision.values:
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
            # A kind with actions (a loop's pass) is decided as one of the
            # actions this proposal allows; a yes/no kind takes none.
            registered = PROPOSAL_KINDS.get(proposal.kind)
            allowed = (
                registered.allowed_actions(proposal)
                if registered is not None and registered.actions
                else ()
            )
            if decision.outcome is DecisionOutcome.Confirmed and allowed:
                if decision.action not in allowed:
                    return DecisionResult(
                        applied=False,
                        error_type="invalid_value",
                        reason=f"{kind_label(proposal.kind)} on {item.name} is "
                        f"decided as one of {list(allowed)}"
                        + (f", not {decision.action!r}." if decision.action else "."),
                    )
            elif decision.action and not (
                allowed and decision.outcome is DecisionOutcome.Rejected
            ):
                return DecisionResult(
                    applied=False,
                    error_type="invalid_value",
                    reason=f"A {kind_label(proposal.kind)} decision takes no action; "
                    "confirm or reject it.",
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
                        error_type="stale_proposal",
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
                    error_type="stale_proposal",
                    reason=(
                        f"{task_name} on {item.name} has re-run since you looked; "
                        "look at the current result and decide that."
                        if proposal.task_id
                        else f"{task_name} on {item.name} was proposed before runs "
                        "were named; re-run it to decide it."
                    ),
                )
            apply_values = None
            if decision.outcome is DecisionOutcome.Confirmed:
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
                if missing and not as_it_stands:
                    return DecisionResult(
                        applied=False,
                        error_type="invalid_value",
                        reason=f"Confirming a pending {proposal.kind} proposal "
                        f"needs its values {missing}, as proposed or adjusted.",
                    )

            if check_only:
                return DecisionResult(applied=True)
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
            # Saved here, so no decider can forget to: the prompt bar, the
            # App Server and a task's own decision all land on disk. A save
            # that fails leaves the decision in memory, and says so.
            try:
                self.save()
            except Exception as e:
                logging.exception(
                    f"{item.name}: the decision on {task_name} landed but the "
                    "experiment could not be saved"
                )
                result.save_error = str(e) or type(e).__name__
        # Committed: a subscriber that raises is logged, and does not turn a
        # decision that landed into an error for the decider.
        try:
            self.decided.emit(item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return result

    def _decide_on_experiment(
        self, decision: Decision, check_only: bool = False
    ) -> DecisionResult:
        """A decision on a proposal about the experiment itself (``item_id``
        None), named by its id. Nothing is written through: such a proposal is
        approved or rejected, and whoever asked acts on the decision."""
        with EXPERIMENT_WRITE_LOCK:
            proposal = next(
                (p for p in self.proposals if p.id == decision.proposal_id), None
            )
            if proposal is None:
                return DecisionResult(
                    applied=False,
                    error_type="stale_proposal",
                    reason=f"No proposal {decision.proposal_id!r} on the experiment.",
                )
            what = kind_label(proposal.kind) or proposal.kind
            if not proposal.pending:
                return DecisionResult(
                    applied=False,
                    error_type="stale_proposal",
                    reason=f"The {what} is no longer waiting.",
                )
            if decision.outcome not in (
                DecisionOutcome.Confirmed,
                DecisionOutcome.Rejected,
            ):
                return DecisionResult(
                    applied=False,
                    error_type="invalid_value",
                    reason=f"{decision.outcome.name} is not a decision: confirm "
                    "or reject.",
                )
            if decision.values or decision.action:
                return DecisionResult(
                    applied=False,
                    error_type="invalid_value",
                    reason=f"A {what} is confirmed or rejected as it stands.",
                )
            if check_only:
                return DecisionResult(applied=True)
            proposal.decisions.append(decision)
            proposal.asking = False
            result = self._save_quietly(f"the decision on the {what}")
        try:
            self.decided.emit("", "")
        except Exception:
            logging.exception(f"a subscriber to decided raised for the {what}")
        return result

    def suggest(
        self,
        item_id: str,
        task_name: str,
        proposal_id: str,
        suggestion: Suggestion,
    ) -> DecisionResult:
        """Put ``suggestion`` on the pending proposal ``proposal_id``, replacing
        the one there. Checked as the decision it would make is (``decide``),
        so one that could not be approved is refused now; nothing is decided."""
        return self._run_edit(
            self._suggest, item_id, task_name, proposal_id, suggestion
        )

    def _suggest(
        self, item_id: str, task_name: str, proposal_id: str, suggestion: Suggestion
    ) -> DecisionResult:
        if not suggestion.reason:
            return DecisionResult(
                applied=False,
                error_type="missing_field",
                reason="A suggestion says why: pass a reason.",
            )
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            proposal = self._current(item, task_name, proposal_id)
            if proposal is None or not proposal.pending:
                return DecisionResult(
                    applied=False,
                    error_type="stale_proposal",
                    reason=f"{task_name} has no pending proposal {proposal_id!r}; "
                    "suggest on one proposal_list shows.",
                )
            try:
                values = suggestion.values_for(proposal)
            except ValueRefused as e:
                return DecisionResult(
                    applied=False, error_type="invalid_value", reason=str(e)
                )
            checked = self._decide(
                item_id,
                task_name,
                self._approving(proposal, suggestion, values),
                check_only=True,
            )
            if not checked.applied:
                return checked
            proposal.suggestion = suggestion
            result = self._save_quietly(f"{item.name}: the {task_name} suggestion")
        self._notify(self.suggested, item_id, task_name)
        return result

    def decline_suggestion(
        self, item_id: str, task_name: str, proposal_id: str, reason: str = ""
    ) -> DecisionResult:
        """Mark the suggestion on ``proposal_id`` declined, with the decider's
        reason for the suggester. The proposal stays pending."""
        return self._run_edit(
            self._decline_suggestion, item_id, task_name, proposal_id, reason
        )

    def _decline_suggestion(
        self, item_id: str, task_name: str, proposal_id: str, reason: str
    ) -> DecisionResult:
        with EXPERIMENT_WRITE_LOCK:
            item = self.get_item_by_id(item_id)
            proposal = self._current(item, task_name, proposal_id)
            suggestion = proposal.suggestion if proposal is not None else None
            if suggestion is None or suggestion.declined:
                return DecisionResult(
                    applied=False,
                    error_type="stale_proposal",
                    reason="There is no suggestion to decline.",
                )
            suggestion.declined = True
            suggestion.declined_reason = reason
            result = self._save_quietly(f"{item.name}: the {task_name} suggestion")
        self._notify(self.suggested, item_id, task_name)
        return result

    def approve_suggestion(
        self, item_id: str, task_name: str, proposal_id: str, actor: Actor, via: str
    ) -> DecisionResult:
        """Decide ``proposal_id`` as its suggestion says, as ``actor``: the
        decision names the suggestion it approved."""
        item = self.get_item_by_id(item_id)
        proposal = self._current(item, task_name, proposal_id)
        suggestion = proposal.suggestion if proposal is not None else None
        if suggestion is None or suggestion.declined:
            return DecisionResult(
                applied=False,
                error_type="stale_proposal",
                reason="There is no suggestion to approve.",
            )
        try:
            values = suggestion.values_for(proposal)
        except ValueRefused as e:
            return DecisionResult(
                applied=False, error_type="invalid_value", reason=str(e)
            )
        decision = self._approving(proposal, suggestion, values)
        decision.actor = Actor.parse(actor)
        decision.via = via
        return self.decide(item_id, task_name, decision)

    @staticmethod
    def _current(item, task_name: str, proposal_id: str) -> Optional[Proposal]:
        if item is None:
            return None
        return next(
            (p for p in item.current_proposals(task_name) if p.id == proposal_id),
            None,
        )

    @staticmethod
    def _approving(
        proposal: Proposal, suggestion: Suggestion, values: Dict[str, Any]
    ) -> Decision:
        return Decision(
            outcome=suggestion.outcome,
            actor=suggestion.actor,
            values=values,
            reason=suggestion.reason
            if suggestion.outcome is DecisionOutcome.Rejected
            else "",
            proposal_id=proposal.id,
            action=suggestion.action,
            suggestion_id=suggestion.id,
        )

    def _save_quietly(self, what: str) -> DecisionResult:
        result = DecisionResult(applied=True)
        try:
            self.save()
        except Exception as e:
            logging.exception(f"could not save {what}")
            result.save_error = str(e) or type(e).__name__
        return result

    def ask_proposal(
        self, item_id: Optional[str], task_name: str, proposal: Proposal
    ) -> bool:
        """Record a question the task is about to park on, and say so. With
        ``item_id`` None, a proposal about the experiment itself.

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
            item = self.get_item_by_id(item_id) if item_id is not None else self
            if item is None:
                return False
            # Same rule as a task's own proposal: a decided one stays on the
            # record before the new question, a pending one is replaced -- it
            # was never answered, so there is nothing to keep.
            proposal.provenance["asked"] = True
            if item_id is None:
                self.record_proposal(proposal)
            else:
                item.record_proposal(task_name, proposal)
            proposal.asking = True
        try:
            self._notify(self.asked, item_id or "", task_name)
        except Exception:
            logging.exception(f"a subscriber to asked raised for {task_name}")
        return True

    def record_unasked(
        self, item_id: str, task_name: str, proposal: Proposal, reason: str
    ) -> bool:
        """Put a question nobody was asked on the record: the proposal, and an
        ``Unreviewed`` decision on it carrying the values the task went on
        to use. What an automated task leaves on the record.

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
                    # What silence means for the kind: a loop's pass runs once,
                    # then goes on.
                    action=(
                        PROPOSAL_KINDS[proposal.kind].silent_action(proposal)
                        if proposal.kind in PROPOSAL_KINDS
                        else ""
                    ),
                )
            )
        try:
            self._notify(self.decided, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return True

    def fill_in_decision(
        self, item_id: str, task_name: str, proposal_id: str, values: Dict[str, Any]
    ) -> bool:
        """Put ``values`` on a confirmation that carried none: what a task
        read from the instrument once the operator said "as it stands"
        (``BaseTask.ask`` with ``decided``). Written by the task on its
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
            if proposal is not None:
                proposal.filling_in = False
            if (
                decision is None
                or decision.outcome is not DecisionOutcome.Confirmed
                or decision.values
            ):
                return False
            decision.values = dict(values)
        try:
            self._notify(self.decided, item_id, task_name)
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
            for item in [*self.items(), self]:
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
        try:
            self._notify(self.decided, item_id, task_name)
        except Exception:
            logging.exception(f"a subscriber to decided raised for {task_name}")
        return DecisionResult(applied=True)

    def withdraw_experiment_proposal(
        self, proposal_id: str, reason: str, actor: Optional[Actor] = None
    ) -> DecisionResult:
        """Close a proposal about the experiment nobody decided: whoever asked
        took it back, or can no longer act on it. Like ``withdraw_proposal``,
        not a decision and not run where the experiment is edited."""
        with EXPERIMENT_WRITE_LOCK:
            proposal = next((p for p in self.proposals if p.id == proposal_id), None)
            if proposal is None or not proposal.pending:
                return DecisionResult(
                    applied=False,
                    error_type="stale_proposal",
                    reason=f"No pending proposal {proposal_id!r} on the experiment.",
                )
            proposal.asking = False
            proposal.decisions.append(
                Decision(
                    outcome=DecisionOutcome.Withdrawn,
                    actor=actor or Actor.task("workflow"),
                    reason=reason,
                    via="workflow",
                    proposal_id=proposal.id,
                )
            )
            result = self._save_quietly("a withdrawn proposal")
        try:
            self._notify(self.decided, "", "")
        except Exception:
            logging.exception("a subscriber to decided raised for a withdrawal")
        return result

    def pending_proposals(
        self,
    ) -> List[Tuple[Any, str, Proposal]]:
        """Every question a task is waiting on, in item order. Derived, never
        stored, so it is the same list from the GUI and the server."""
        listed = []
        for item in self.items():
            for task_name in item.asking_tasks():
                for proposal in item.current_proposals(task_name):
                    if proposal.pending and proposal.asking:
                        listed.append((item, task_name, proposal))
        return listed

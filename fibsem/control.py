"""Who is in control of the microscope: one action at a time.

One answer for everyone who acts on the instrument -- a workflow run, a user
script, a server route, later the operator's own buttons -- instead of each
widget deciding for itself whether it may.

* **Taken at entry points only.** A run start, a script start, a server route;
  never inside microscope methods, so a workflow's own acquisitions never
  re-take control and nothing is re-entrant.
* **A second take is refused at once**, naming who is in control. It never
  queues: a caller blocked behind an hour-long mill should be told, not held.
* **Stop is always allowed.** Stopping never takes control, so it is never
  refused.
* **A run lends control during a question.** While the workflow waits on an
  answer, whoever answers may act, one action at a time; when the question
  ends, the run waits for that action to finish and carries on.
* **A long command can be acted beside.** A mill the App Server runs on holds
  control until it ends, so a run lending control waits for it too. Whoever
  started it may still act beside it, one action at a time (an image during a
  coincidence mill, or a move, however unwise): ``let_act_beside``.

Lives on the microscope (``microscope.control``), not in the app, so headless
runs, scripts and the bench server share it.
"""

import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from psygnal import Signal

from fibsem.acting import OPERATOR, ActorKind

__all__ = [
    "HOLD_COMMANDS",
    "STOP_COMMANDS",
    "Control",
    "InControl",
    "MicroscopeBusy",
    "lending",
]


# Device and service commands that stop something (an FM z-stack stops through
# `imaging.stop`; milling's `end` ends the stage keeping what it milled). They
# take no control, so the running move or mill never refuses them.
STOP_COMMANDS = frozenset(("stop", "stop_live", "end"))

# Commands that hold or let go of what is running (milling's pause and resume).
# Like a stop they take no control: the mill they act on holds it.
HOLD_COMMANDS = frozenset(("pause", "resume"))


@dataclass(frozen=True)
class InControl:
    """Who is in control, doing what, since when (epoch seconds)."""

    by: ActorKind
    action: str
    since: float

    def describe(self) -> str:
        return f"the {self.by} is in control ({self.action})"


class MicroscopeBusy(RuntimeError):
    """A take refused because someone else is in control."""

    def __init__(self, in_control: InControl) -> None:
        self.in_control = in_control
        super().__init__(f"The microscope is busy: {in_control.describe()}.")


# The controls each thread took, so a run waiting on a question can lend the
# control it holds without being handed the microscope. Keyed by the thread
# that took, and removed on release from whichever thread releases: a server
# dependency enters and exits on different pool threads.
_held_lock = threading.Lock()
_held: Dict[int, List["Control"]] = {}


class Control:
    """Who is in control of one microscope.

    ``current`` is who is acting now: the borrower while a question is being
    answered, else whoever took control. ``changed`` emits it (or None) on every
    take and release, on the thread that took or released.
    """

    changed = Signal(object)

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._owner: Optional[InControl] = None
        self._owner_thread: Optional[int] = None
        self._lent = False
        self._borrower: Optional[InControl] = None
        # The holder its own actor may act beside (`let_act_beside`), and the
        # action beside it.
        self._open: Optional[InControl] = None
        self._beside: Optional[InControl] = None
        # The holder others may join (``take(join=True)``), and how many have.
        self._joinable: List[InControl] = []
        self._joined = 0
        # The last take, owner or borrower, kept after release: an action that
        # starts something and returns (start_milling) has let go of control
        # while what it started runs on, so Stop agent can still find it.
        self._last: Optional[InControl] = None
        # Epoch seconds of the operator's last take or release, for the
        # agent's Drive countdown. None until the operator acts.
        self.last_operator_action: Optional[float] = None

    @property
    def current(self) -> Optional[InControl]:
        with self._cond:
            return self._borrower or self._owner

    @property
    def last(self) -> Optional[InControl]:
        """The most recent take, whether or not it has been released."""
        with self._cond:
            return self._last

    @contextmanager
    def take(
        self, by: "ActorKind | str", action: str, join: bool = False
    ) -> Iterator[InControl]:
        """Be in control for the block, or raise :class:`MicroscopeBusy` at once.

        While the one in control has lent it (a question is standing), the take
        borrows instead: it succeeds if nobody else has borrowed it. While the
        holder is open to its own actor (`let_act_beside`), that actor's take
        acts beside it. Takes that *join* never refuse each other: one joins
        another that joins, for the operator's clicks in the window, and the
        holder's release waits for those that joined it.
        """
        by = ActorKind(by)
        entry = InControl(by=by, action=action, since=time.time())
        with self._cond:
            holder = self._borrower or self._owner
            beside = borrowed = joined = False
            if join and any(holder is e for e in self._joinable):
                joined = True
                self._joined += 1
            elif self._owner is None:
                self._owner = entry
                self._owner_thread = threading.get_ident()
            elif self._lent and self._borrower is None:
                borrowed = True
                self._borrower = entry
            elif holder is self._open and holder.by == by and self._beside is None:
                beside = True
                self._beside = entry
            else:
                raise MicroscopeBusy(self._beside or holder)
            if join and not joined:
                self._joinable.append(entry)
            self._note_operator(by)
            self._last = entry
        if joined:
            try:
                yield holder
            finally:
                with self._cond:
                    self._joined -= 1
                    self._note_operator(by)
                    self._cond.notify_all()
            return
        if not (borrowed or beside):
            _remember(self._owner_thread, self)
        self.changed.emit(entry)
        try:
            yield entry
        finally:
            with self._cond:
                if any(entry is e for e in self._joinable):
                    self._joinable = [e for e in self._joinable if e is not entry]
                    # Those that joined it end first, as with acting beside.
                    while self._joined:
                        self._cond.wait()
                if self._open is entry:
                    self._open = None
                    # What acts beside it ends first: nobody takes over mid-action.
                    while self._beside is not None:
                        self._cond.wait()
                if beside:
                    self._beside = None
                elif borrowed:
                    self._borrower = None
                else:
                    _forget(self._owner_thread, self)
                    self._owner = None
                    self._owner_thread = None
                    self._lent = False
                self._note_operator(by)
                self._cond.notify_all()
                now = self._borrower or self._owner
            self.changed.emit(now)

    def let_act_beside(self, entry: InControl) -> None:
        """Let *entry*'s actor take control beside it while it holds it, one
        action at a time: for a long command that runs on after its request
        (`fibsem.devices.runs`), whose starter may image or move during it."""
        with self._cond:
            if entry is (self._borrower or self._owner):
                self._open = entry

    def beside_of(self, entry: InControl) -> Optional[InControl]:
        """What *entry* acts beside (`let_act_beside`), or None when it does not."""
        with self._cond:
            return self._open if entry is self._beside else None

    @contextmanager
    def lend(self) -> Iterator[None]:
        """Let others act, one at a time, for the block: the one in control is
        waiting on a question. On leaving, wait for the borrower's action to end,
        so the owner never acts beside it. Nothing to lend, nothing happens."""
        with self._cond:
            lends = self._owner is not None and not self._lent
            if lends:
                self._lent = True
        try:
            yield
        finally:
            if lends:
                with self._cond:
                    self._lent = False
                    while self._borrower is not None:
                        self._cond.wait()

    def _note_operator(self, by: str) -> None:
        if by == OPERATOR:
            self.last_operator_action = time.time()


def _remember(thread: Optional[int], control: Control) -> None:
    with _held_lock:
        _held.setdefault(thread, []).append(control)


def _forget(thread: Optional[int], control: Control) -> None:
    with _held_lock:
        held = _held.get(thread, [])
        if control in held:
            held.remove(control)
        if not held:
            _held.pop(thread, None)


@contextmanager
def lending() -> Iterator[None]:
    """Lend whatever control this thread took, for the block.

    For a run's thread waiting on a question: it holds control without the
    question's code having the microscope to hand. A thread that holds none
    (a headless helper, a test) lends nothing.
    """
    with _held_lock:
        controls = list(_held.get(threading.get_ident(), ()))
    with ExitStack() as stack:
        for control in controls:
            stack.enter_context(control.lend())
        yield

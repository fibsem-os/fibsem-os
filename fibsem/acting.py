"""Who is acting: the one ``Actor`` every record names, and the mark that says
who is acting on the microscope right now.

An :class:`Actor` is a kind and a name, written ``kind:name``
(``operator:patrick``, ``agent:claude``, ``task:Mill Rough``). Proposals,
decisions, verdicts and who holds control all say who with it.

The code that makes a call marks it, for the thread of control it runs on: a
task for the length of its run, the App Server for each request. The
experiment's record reads the mark when an event is emitted, which is on the
thread that caused it.

A context variable rather than a thread-local. A new thread starts unmarked, as
it would with a thread-local, so a task's mark never reaches a UI worker running
beside it. But a web request's handler runs in a worker thread with the
request's context copied in, so the server's mark reaches the handler.

Work handed to another thread that is already running -- a Qt signal to the GUI
thread -- does not take the mark with it. :func:`carried` wraps the work where
it is handed over, so it runs there as whoever handed it over.

Only the task and the agent mark their calls. In the app, the operator is
everything else, and says so where the record is kept, not here.
"""

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Callable, Iterator, Optional, TypeVar, Union

from pydantic import PlainSerializer, PlainValidator

T = TypeVar("T")


class ActorKind(str, Enum):
    """What kind of thing acted. A str, so a kind compares equal to its name
    and is written as it."""

    operator = "operator"
    agent = "agent"
    task = "task"  # a task, or anything that ran on its own and used its own value
    workflow = "workflow"  # a workflow run holding control
    model = "model"
    script = "script"

    def __str__(self) -> str:
        return self.value


TASK = ActorKind.task
AGENT = ActorKind.agent
OPERATOR = ActorKind.operator
WORKFLOW = ActorKind.workflow
SCRIPT = ActorKind.script

# What the kinds were called on records written before there was one Actor.
_OLD_KINDS = {"human": ActorKind.operator, "auto": ActorKind.task}


@dataclass(frozen=True)
class Actor:
    """Who did something: the kind of thing, and its name -- the operator,
    the agent's model, the task that used its own value. Written ``kind:name``,
    which is also what ``str()`` gives."""

    kind: ActorKind
    name: str = ""

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.name}"

    @classmethod
    def parse(cls, text: Union[str, "Actor", None]) -> "Actor":
        """The ``kind:name`` form back into an Actor. Reads the older
        ``human:`` and ``auto:`` prefixes too. Text with no known prefix is an
        operator whose name is the whole string."""
        if isinstance(text, Actor):
            return text
        kind, sep, name = str(text or "").partition(":")
        if kind in _OLD_KINDS and sep:
            return cls(_OLD_KINDS[kind], name)
        if kind in ActorKind.__members__ and (sep or not name):
            return cls(ActorKind(kind), name)
        return cls(ActorKind.operator, str(text or ""))

    @classmethod
    def operator(cls, name: str) -> "Actor":
        return cls(ActorKind.operator, name)

    @classmethod
    def agent(cls, model: str) -> "Actor":
        return cls(ActorKind.agent, model)

    @classmethod
    def task(cls, name: str) -> "Actor":
        """A decision nobody made: the task used its own value so the run
        continues, and the record says so."""
        return cls(ActorKind.task, name)

    @property
    def label(self) -> str:
        """How it reads: the name for an operator, "agent · model", "task ·
        name"; "someone" / "unknown" when the name is blank."""
        if self.kind is ActorKind.operator:
            return self.name or "someone"
        return f"{self.kind.value} · {self.name or 'unknown'}"


# An Actor as a record saves it: its ``kind:name`` text.
SavedActor = Annotated[
    Actor,
    PlainValidator(Actor.parse, json_schema_input_type=str),
    PlainSerializer(str, return_type=str, when_used="json"),
]


_actor: "ContextVar[Optional[ActorKind]]" = ContextVar("fibsem_actor", default=None)


@contextmanager
def acting(actor: Union[ActorKind, str, None]) -> Iterator[None]:
    """Mark what runs inside this block, in this thread of control, as *actor*'s.

    None clears the mark: for recording, later, something done unmarked.
    Also a decorator: ``@acting(TASK)`` marks every call of the function.
    """
    token = _actor.set(ActorKind(actor) if actor is not None else None)
    try:
        yield
    finally:
        _actor.reset(token)


def current_actor() -> Optional[ActorKind]:
    """Who marked this thread of control, or None if nobody did."""
    return _actor.get()


def signer(operator: Actor, agent: str = "remote") -> Actor:
    """Who signs what this call changes: *operator* for a call marked as the
    operator's, otherwise the agent *agent*. Read it on the calling thread."""
    return operator if current_actor() is OPERATOR else Actor.agent(agent)


def carried(fn: Callable[..., T]) -> Callable[..., T]:
    """*fn*, marked as whoever called this, on whatever thread it later runs.

    The mark is read now, not when *fn* runs: hand the result to the other
    thread.
    """
    actor = current_actor()

    @functools.wraps(fn)
    def run(*args, **kwargs) -> T:
        with acting(actor):
            return fn(*args, **kwargs)

    return run

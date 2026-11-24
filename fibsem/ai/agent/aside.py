"""Side questions: the operator's ``/btw`` message, answered beside the agent's
work without joining its conversation.

The App Server records the message with ``aside: true``; the watch says it as
an `ASIDE` wake-up, which fibsem-agent hands here instead of to a turn. The
answer comes from the Claude Code CLI's own side question (the ``/btw`` of an
interactive session, a ``side_question`` control request): one model call on
the session's conversation as it stands, mid-turn included, with no tools, and
nothing of it written to the conversation. It goes back to the operator as a
message whose text says it is a side answer. Needs nothing from the SDK at
import.
"""

import asyncio
import collections
import logging
import re
from typing import Any, Callable, Deque, Dict, List, Optional

log = logging.getLogger(__name__)

PREFIX = "/btw"
# The payload key that marks the operator's message as a side question.
MARK = "aside"
# The watch's kind for one (`fibsem.ai.agent.watch.Aside`).
ASIDE = "aside"

_QUESTION = re.compile(r"^\s*/btw(?:\s+|$)(.*)$", re.IGNORECASE | re.DOTALL)

# The operator's message that cuts the agent's turn short: ``/now`` and what
# to say. Any other message waits for the turn to end. It stops the turn, not
# the run: the agent reads the message and carries on from there.
INTERRUPT_PREFIX = "/now"
# The payload key that marks it.
INTERRUPT = "interrupt"
_INTERRUPTION = re.compile(r"^\s*/now(?:\s+|$)(.*)$", re.IGNORECASE | re.DOTALL)

# ``/status``: a side question asked for the operator.
_STATUS = re.compile(r"^\s*/status(?:\s+|$)(.*)$", re.IGNORECASE | re.DOTALL)
STATUS = (
    "Where do things stand? In a few lines: what you are doing now, what is "
    "done, what is next, and anything you are waiting on or worried about."
)

# ``/note``: a note the operator keeps on the session's record. It is not to
# the agent and wakes it for nothing: it hears it with its next turn.
_NOTE = re.compile(r"^\s*/note(?:\s+|$)(.*)$", re.IGNORECASE | re.DOTALL)
# The payload key that marks one.
FOR_RECORD = "for_record"

# Earlier exchanges given with each question, as Claude Code's /btw does.
HISTORY = 20
# How long the CLI has to answer one.
TIMEOUT_S = 120.0
# Questions waiting while one is answered; past that the operator is told.
WAITING = 5
# The answer as the operator reads it: marked, and short enough for a message.
LEAD = "Side answer (/btw, not added to the agent's conversation): "
ANSWER_LIMIT = 3800

# Said with the question. The CLI already tells the model it has no tools and
# that the main agent carries on; this adds what a side question can't do here.
FRAME = (
    "The operator at the microscope asks this side question from the chat. "
    "Answer in a few sentences they can read at a glance. When a figure may "
    "have changed since you last saw it (a current, a position, a status), say "
    "when you saw it. If it is really an instruction or a decision (stop, "
    "move, mill, don't do something, change a setting or the plan), it has not "
    "reached you: say so, and ask them to send it again without /btw so it "
    "reaches you as a message.\n\nThe question: "
)

NO_ANSWER = (
    "No answer came back. Ask again, or send it without /btw to ask the agent "
    "in its conversation."
)


class Unavailable(Exception):
    """The session can't take a side question now."""


def interruption_of(text: Any) -> Optional[str]:
    """The words of an interrupting message in *text*, one starting ``/now``
    (any case): the text after it, maybe empty. None for an ordinary message,
    which waits for the agent's turn to end."""
    found = _INTERRUPTION.match(str(text or ""))
    if found is None:
        return None
    return found.group(1).strip()


def note_of(text: Any) -> Optional[str]:
    """The operator's note for the record in *text*, a message starting
    ``/note``: the text after it, or None (``/note`` alone says nothing)."""
    found = _NOTE.match(str(text or ""))
    if found is None:
        return None
    return found.group(1).strip() or None


def question_of(text: Any) -> Optional[str]:
    """The side question in *text*, a message starting ``/btw `` (any case),
    or None for an ordinary message. ``/btw`` alone asks nothing. ``/status``
    asks `STATUS`, with what follows it added."""
    found = _STATUS.match(str(text or ""))
    if found is not None:
        more = found.group(1).strip()
        return f"{STATUS} {more}" if more else STATUS
    found = _QUESTION.match(str(text or ""))
    if found is None:
        return None
    question = found.group(1).strip()
    return question or None


async def ask(
    client, question: str, history: List[Dict[str, str]], timeout_s: float
) -> Optional[str]:
    """The CLI's answer to *question* on *client*'s session (a connected
    ``ClaudeSDKClient``), after the earlier *history* ({question, response}
    each); None when it had none. The Python SDK has no call for it, so it goes
    as the control request the TypeScript SDK's ``askSideQuestion`` sends.
    Raises `Unavailable` when the client can't send one."""
    query = getattr(client, "_query", None)
    send = getattr(query, "_send_control_request", None)
    if send is None:
        raise Unavailable("this SDK can't send a side question")
    request: Dict[str, Any] = {"subtype": "side_question", "question": question}
    if history:
        request["history"] = list(history)
    response = await send(request, timeout=timeout_s)
    answer = response.get("response") if isinstance(response, dict) else None
    if isinstance(response, dict) and response.get("usage"):
        log.info("Side question usage: %s", response["usage"])
    return answer if isinstance(answer, str) and answer.strip() else None


def message_text(answer: str) -> str:
    """*answer* as the operator's message: marked, and cut to fit."""
    answer = answer.strip()
    if len(answer) > ANSWER_LIMIT:
        answer = answer[: ANSWER_LIMIT - 1] + "…"
    return LEAD + answer


class Asides:
    """The side questions of a run, answered one at a time by *agent*'s
    ``side_question(question, history, timeout_s)``. Each is marked read as its
    answer starts and answered on the App Server *client* gives (None while
    there is none), as the agent's message replying to it."""

    def __init__(
        self,
        agent,
        client: Callable[[], Optional[Any]],
        timeout_s: float = TIMEOUT_S,
    ) -> None:
        self._agent = agent
        self._client = client
        self._timeout_s = timeout_s
        self._waiting: Deque[Any] = collections.deque()
        self._ready: Optional[asyncio.Event] = None
        # The exchanges answered, oldest first: the CLI is given the newest.
        self.history: List[Dict[str, str]] = []

    def add(self, wake) -> None:
        """A side question the watch said (`watch.Aside`), on the loop."""
        if len(self._waiting) >= WAITING:
            log.warning("Side question %s: too many wait; it is refused.", wake.id)
            asyncio.ensure_future(
                self._post(
                    wake.id,
                    f"{WAITING} side questions are waiting already. Ask again "
                    "once they are answered.",
                )
            )
            return
        self._waiting.append(wake)
        if self._ready is not None:
            self._ready.set()

    async def run(self) -> None:
        """Answer the questions as they come, until cancelled."""
        self._ready = asyncio.Event()
        while True:
            if not self._waiting:
                self._ready.clear()
                await self._ready.wait()
                continue
            wake = self._waiting.popleft()
            await self.answer(wake.id, wake.question)

    async def answer(self, message_id: str, question: str) -> str:
        """Answer one; what the operator was sent."""
        client = self._client()
        if client is not None:
            from fibsem.ai.agent.watch import report_read

            await asyncio.to_thread(report_read, client, [message_id])
        log.info("Side question %s: %s", message_id, question)
        ask_it = getattr(self._agent, "side_question", None)
        try:
            if ask_it is None:
                raise Unavailable("this agent takes no side questions")
            said = await asyncio.wait_for(
                ask_it(FRAME + question, self.history[-HISTORY:], self._timeout_s),
                self._timeout_s + 5.0,
            )
        except asyncio.TimeoutError:
            said, why = None, f"no answer within {self._timeout_s:.0f} s"
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - the operator is told why
            log.warning("Side question %s failed: %s", message_id, error)
            said, why = None, str(error) or type(error).__name__
        else:
            why = None
        if said:
            self.history.append({"question": question, "response": said.strip()})
            text = said
        elif why:
            text = (
                f"It couldn't be answered as a side question ({why}). Send it "
                "without /btw to ask the agent in its conversation."
            )
        else:
            text = NO_ANSWER
        log.info("Side answer %s: %s", message_id, " ".join(text.split()))
        await self._post(message_id, text)
        return text

    async def _post(self, message_id: str, answer: str) -> bool:
        """*answer* on the record, as the agent's message replying to
        *message_id*; False when no App Server took it."""
        client = self._client()
        if client is None:
            log.warning("Side answer %s: no App Server to send it to.", message_id)
            return False
        body = {"text": message_text(answer), "reply_to": message_id}

        def post() -> bool:
            try:
                response = client.request(
                    "POST", "/app/experiment/messages", json=body, timeout=10.0
                )
            except Exception:  # noqa: BLE001 - logged; the run goes on
                log.warning("Sending side answer %s failed.", message_id, exc_info=True)
                return False
            if response.status_code >= 400:
                log.warning(
                    "Side answer %s was refused (%s).",
                    message_id,
                    response.status_code,
                )
                return False
            return True

        return await asyncio.to_thread(post)

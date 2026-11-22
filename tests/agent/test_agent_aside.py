"""Side questions (``/btw``): what makes one, how the watch says it, how it
reaches the CLI, and how its answer reaches the operator.

The SDK is real here: a ``ClaudeSDKClient`` over a stand-in for the CLI's end
of its stream, which answers the control protocol as the CLI does and holds a
turn open until it is told to end it. The App Server end to end is in
``tests/server/test_agent_harness.py``.
"""

import asyncio
import json
import logging

import pytest

from fibsem.ai.agent import aside
from fibsem.ai.agent import watch as watch_mod

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient  # noqa: E402
from claude_agent_sdk._internal.transport import Transport  # noqa: E402

from fibsem.ai.agent.harness import SdkAgent  # noqa: E402

RESULT = {
    "type": "result",
    "subtype": "success",
    "duration_ms": 10,
    "duration_api_ms": 5,
    "is_error": False,
    "num_turns": 1,
    "session_id": "main",
    "total_cost_usd": 0.5,
}


class StandInCli(Transport):
    """The CLI's end of the SDK's stream: it answers ``initialize`` and
    ``get_context_usage``, answers a ``side_question`` with *answer*, and ends
    a turn only once ``end_turn`` is set, as a long turn would."""

    def __init__(self, answer="The ion beam was at 30 pA at the last image."):
        self.written = []
        self.side_questions = []
        self.end_turn = asyncio.Event()
        self._answer = answer
        self._out: asyncio.Queue = asyncio.Queue()
        self._ready = False

    async def connect(self):
        self._ready = True

    def is_ready(self):
        return self._ready

    async def write(self, data):
        message = json.loads(data)
        self.written.append(message)
        if message.get("type") == "control_request":
            request = message["request"]
            response = {}
            if request["subtype"] == "side_question":
                self.side_questions.append(request)
                response = {"response": self._answer, "synthetic": False}
            await self._out.put(
                {
                    "type": "control_response",
                    "response": {
                        "subtype": "success",
                        "request_id": message["request_id"],
                        "response": response,
                    },
                }
            )
        elif message.get("type") == "user":
            asyncio.ensure_future(self._turn())

    async def _turn(self):
        await self.end_turn.wait()
        await self._out.put(RESULT)

    async def read_messages(self):
        while True:
            yield await self._out.get()

    async def close(self):
        self._ready = False

    async def end_input(self):
        pass


def _users(cli):
    return [m for m in cli.written if m.get("type") == "user"]


def _controls(cli):
    return [m["request"]["subtype"] for m in cli.written if "request" in m]


def test_a_side_question_is_a_message_starting_btw():
    assert aside.question_of("/btw what current is the ion beam at?") == (
        "what current is the ion beam at?"
    )
    assert aside.question_of("  /BTW\nwhich lamella is next?") == (
        "which lamella is next?"
    )
    # Anything else is an ordinary message, a "btw" in passing included.
    for text in ("/btw", "/btw   ", "/btwx what?", "btw stop after 02", "stop /btw"):
        assert aside.question_of(text) is None, text


def test_an_interrupting_message_is_one_starting_stop():
    assert aside.interruption_of("/now look at 01 first") == "look at 01 first"
    assert aside.interruption_of("  /NOW\nwrong lamella") == "wrong lamella"
    # /now alone interrupts with nothing more to say.
    assert aside.interruption_of("/now") == ""
    for text in ("/nowx please", "now do 02", "please /now", "/btw now?"):
        assert aside.interruption_of(text) is None, text


def test_the_watch_says_an_interrupting_message_and_marks_it_urgent():
    said = watch_mod._message_line(
        {"id": "ab12", "text": "/now wrong lamella", aside.INTERRUPT: True}
    )
    assert said.kind == watch_mod.MESSAGE and said.facts == {"interrupt": True}
    assert said.startswith(
        "MESSAGE id=ab12 from the operator, who stopped your turn to say this: "
        "wrong lamella --"
    )
    bare = watch_mod._message_line(
        {"id": "cd34", "text": "/now", aside.INTERRUPT: True}
    )
    assert "stop and ask them what they want" in bare
    # Unmarked (another client, an older server), it is an ordinary message.
    plain = watch_mod._message_line({"id": "ef56", "text": "/now later"})
    assert plain.facts == {} and "from the operator: /now later" in plain


def test_the_watch_says_a_side_question_apart_from_a_message():
    said = watch_mod._message_line(
        {"id": "ab12", "text": "/btw where is the report?", aside.MARK: True}
    )
    assert isinstance(said, watch_mod.Aside)
    assert said.kind == aside.ASIDE and said.kind not in {watch_mod.MESSAGE}
    assert (said.id, said.question, said.message_ids) == (
        "ab12",
        "where is the report?",
        ("ab12",),
    )
    # An agent reading the watch's output is still asked to answer it.
    assert said.startswith("MESSAGE id=ab12 from the operator, a side question:")
    assert "act on nothing" in said
    ordinary = watch_mod._message_line({"id": "cd34", "text": "/btw but unmarked"})
    assert not isinstance(ordinary, watch_mod.Aside)
    assert ordinary.kind == watch_mod.MESSAGE


def test_a_side_question_is_answered_while_a_turn_goes_on_and_joins_nothing():
    async def main():
        cli = StandInCli()
        agent = SdkAgent(ClaudeAgentOptions())
        agent._client = ClaudeSDKClient(ClaudeAgentOptions(), transport=cli)
        await agent._client.connect()
        try:
            turn = asyncio.ensure_future(agent.turn("The watcher printed:\nWAKE"))
            while not _users(cli):
                await asyncio.sleep(0.01)
            history = [{"question": "which lamella is next?", "response": "02"}]
            answer = await agent.side_question(
                aside.FRAME + "what current?", history, 5.0
            )
            assert not turn.done(), "the side question waited for the turn"
            cli.end_turn.set()
            end = await asyncio.wait_for(turn, 5.0)
        finally:
            await agent._client.disconnect()
        return cli, answer, end

    cli, answer, end = asyncio.run(main())
    assert answer == "The ion beam was at 30 pA at the last image."
    assert end.subtype == "success"
    (request,) = cli.side_questions
    # Only the question and the earlier exchanges: no tools, no prompt of its own.
    assert request == {
        "subtype": "side_question",
        "question": aside.FRAME + "what current?",
        "history": [{"question": "which lamella is next?", "response": "02"}],
    }
    # The turn's one prompt, no other, and nothing interrupted it.
    assert len(_users(cli)) == 1
    assert "interrupt" not in _controls(cli)


def test_a_session_that_has_not_started_takes_no_side_question():
    agent = SdkAgent(ClaudeAgentOptions())
    with pytest.raises(aside.Unavailable):
        asyncio.run(agent.side_question("what?", [], 1.0))
    with pytest.raises(aside.Unavailable):
        asyncio.run(aside.ask(object(), "what?", [], 1.0))


class Recording:
    """The App Server as the harness's client sees it: each request kept."""

    status_code = 200

    def __init__(self):
        self.requests = []

    def request(self, method, path, json=None, timeout=None):
        self.requests.append((method, path, json))
        return self


class Answering:
    def __init__(self, *answers):
        self.asked = []
        self._answers = list(answers)

    async def side_question(self, question, history, timeout_s):
        self.asked.append((question, [dict(h) for h in history]))
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_the_answer_is_marked_and_replies_to_the_question_and_keeps_the_exchange():
    server = Recording()
    agent = Answering("02 is next.", "At 14:10.")
    asides = aside.Asides(agent, lambda: server)

    async def main():
        await asides.answer("q1", "which lamella is next?")
        await asides.answer("q2", "when did it start?")

    asyncio.run(main())
    assert server.requests == [
        ("POST", "/app/agent/read", {"ids": ["q1"]}),
        (
            "POST",
            "/app/experiment/messages",
            {"text": aside.LEAD + "02 is next.", "reply_to": "q1"},
        ),
        ("POST", "/app/agent/read", {"ids": ["q2"]}),
        (
            "POST",
            "/app/experiment/messages",
            {"text": aside.LEAD + "At 14:10.", "reply_to": "q2"},
        ),
    ]
    assert agent.asked[0] == (aside.FRAME + "which lamella is next?", [])
    # The second is asked after the first exchange, as Claude Code's /btw does.
    assert agent.asked[1][1] == [
        {"question": "which lamella is next?", "response": "02 is next."}
    ]


def test_a_side_question_that_fails_says_to_ask_it_as_a_message(caplog):
    caplog.set_level(logging.INFO, logger=aside.log.name)
    server = Recording()
    asides = aside.Asides(Answering(RuntimeError("no CLI")), lambda: server)
    said = asyncio.run(asides.answer("q1", "what?"))
    assert "Send it without /btw" in said and "no CLI" in said
    assert server.requests[-1][2]["reply_to"] == "q1"
    assert asides.history == []
    # An agent with no side questions at all is told the same way.
    said = asyncio.run(aside.Asides(object(), lambda: server).answer("q2", "what?"))
    assert "Send it without /btw" in said


def test_a_long_answer_is_cut_to_fit_a_message():
    text = aside.message_text("x" * 10000)
    assert text.startswith(aside.LEAD)
    assert len(text) <= len(aside.LEAD) + aside.ANSWER_LIMIT


def test_the_sdk_still_sends_control_requests_the_way_ask_does():
    """`aside.ask` uses the SDK's private control-request call: an SDK that
    renames it or its timeout fails here, not quietly at the microscope."""
    pytest.importorskip("claude_agent_sdk")
    import inspect

    from claude_agent_sdk import ClaudeSDKClient
    from claude_agent_sdk._internal.query import Query

    send = getattr(Query, "_send_control_request", None)
    assert send is not None
    assert "timeout" in inspect.signature(send).parameters
    # The client keeps its query where `ask` looks for it.
    assert "self._query" in inspect.getsource(ClaudeSDKClient)

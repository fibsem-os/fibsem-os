"""Sending the transcript: what a refused send keeps, drops and logs."""

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from fibsem.ai.agent import harness as harness_mod


class _Client:
    def __init__(self, *statuses):
        self.statuses = list(statuses)
        self.sent = []

    def request(self, method, path, json=None, timeout=None):
        self.sent.append(json["entries"])
        return SimpleNamespace(status_code=self.statuses.pop(0))


def _send(*statuses, lines=2):
    harness = object.__new__(harness_mod.Harness)
    harness._client = _Client(*statuses)
    transcript = [
        {"kind": "text", "text": f"line {i}", "at": time.monotonic()}
        for i in range(lines)
    ]
    harness._agent = SimpleNamespace(transcript=transcript, session_id="s1")
    asyncio.run(harness._send_some_transcript())
    return harness._agent.transcript, harness._client


def test_a_sent_batch_leaves_the_queue():
    left, client = _send(200)
    assert left == [] and len(client.sent) == 1


@pytest.mark.parametrize("status", [409, 429, 500, 503])
def test_a_server_that_cannot_take_it_yet_keeps_the_lines(status):
    left, _ = _send(status)
    assert len(left) == 2


def test_a_refused_batch_is_dropped_with_a_log_line(caplog):
    with caplog.at_level(logging.WARNING, logger="fibsem.agent.harness"):
        left, _ = _send(403)
    assert left == []
    assert "refused the transcript (403): 2 line(s) dropped" in caplog.text


def test_one_bad_line_goes_alone_and_the_rest_are_sent(caplog):
    with caplog.at_level(logging.WARNING, logger="fibsem.agent.harness"):
        left, client = _send(422, 422, 200)
    assert left == []
    assert [len(entries) for entries in client.sent] == [2, 1, 1]
    assert "refused a transcript line: text 'line 0'" in caplog.text

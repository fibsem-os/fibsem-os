"""A read while the instrument is busy: a caller that asks is answered the last
value and its age instead of waiting (fibsem.server.kept_reads)."""

import threading

import pytest
from fastapi.testclient import TestClient

from fibsem.server import AuthConfig, build_server
from fibsem.server.kept_reads import (
    READ_AGE_HEADER,
    READ_WAIT_HEADER,
    KeptReads,
)

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
WAIT = {READ_WAIT_HEADER: "0.2"}


class _Owner:
    pass


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_a_live_read_is_kept_and_has_no_age():
    kept, owner = KeptReads(), _Owner()
    assert kept.read(owner, "x", lambda: 1, wait=None) == (1, None)
    assert kept.read(owner, "x", lambda: 2, wait=0.5) == (2, None)


def test_a_slow_read_answers_the_last_value_and_its_age():
    clock = _Clock()
    kept, owner = KeptReads(clock), _Owner()
    kept.read(owner, "x", lambda: "before", wait=None)
    clock.now += 3.0
    release = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        release.wait(5)
        return "after"

    assert kept.read(owner, "x", slow, wait=0.05) == ("before", 3.0)
    # The read still running is joined, not started again.
    assert kept.read(owner, "x", slow, wait=0.05) == ("before", 3.0)
    assert len(calls) == 1
    release.set()
    assert kept.read(owner, "x", lambda: "next", wait=5) in (
        ("after", None),
        ("next", None),
    )


def test_with_nothing_kept_a_slow_read_is_waited_for():
    kept, owner = KeptReads(), _Owner()
    release = threading.Event()
    threading.Timer(0.1, release.set).start()
    assert kept.read(owner, "x", lambda: release.wait(5) and 7, wait=0.01) == (7, None)


def test_a_failed_read_raises_and_keeps_the_last_value():
    kept, owner = KeptReads(), _Owner()
    kept.read(owner, "x", lambda: 1, wait=None)

    def fails():
        raise ValueError("no")

    with pytest.raises(ValueError):
        kept.read(owner, "x", fails, wait=5)
    assert kept.read(owner, "x", lambda: 2, wait=5) == (2, None)


def test_each_owner_keeps_its_own():
    kept, a, b = KeptReads(), _Owner(), _Owner()
    kept.read(a, "x", lambda: "a", wait=None)
    release = threading.Event()
    # b has nothing kept, so its slow read is waited for, not a's value.
    threading.Timer(0.1, release.set).start()
    assert kept.read(b, "x", lambda: release.wait(5) and "b", wait=0.01) == (
        "b",
        None,
    )


@pytest.fixture(scope="module")
def watching(microscope):
    app = build_server(microscope, auth=AuthConfig.generate(level="watch", token=TOKEN))
    with TestClient(app, raise_server_exceptions=False) as client:
        client.headers.update(AUTH)
        yield client


def _holding(lock: threading.RLock, release: threading.Event) -> threading.Thread:
    """A thread holding the imaging lock, as an image does, until *release*."""
    held = threading.Event()

    def hold():
        with lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    held.wait(5)
    return thread


def test_the_state_while_imaging_is_the_last_one_read(watching, microscope):
    first = watching.get("/microscope/state", headers=WAIT)
    assert first.status_code == 200, first.text
    assert READ_AGE_HEADER.lower() not in first.headers

    release = threading.Event()
    thread = _holding(microscope._threading_lock, release)
    try:
        busy = watching.get("/microscope/state", headers=WAIT)
        assert busy.status_code == 200, busy.text
        assert float(busy.headers[READ_AGE_HEADER]) >= 0
        assert busy.json() == first.json()
    finally:
        release.set()
        thread.join(5)
    after = watching.get("/microscope/state", headers={READ_WAIT_HEADER: "5"})
    assert READ_AGE_HEADER.lower() not in after.headers


def test_without_the_header_a_read_waits(watching, microscope):
    release = threading.Event()
    thread = _holding(microscope._threading_lock, release)
    threading.Timer(0.3, release.set).start()
    try:
        answer = watching.get("/microscope/state")
        assert answer.status_code == 200
        assert READ_AGE_HEADER.lower() not in answer.headers
        assert release.is_set()
    finally:
        release.set()
        thread.join(5)


def test_device_reads_take_the_header(watching):
    values = watching.get("/microscope/devices/electron/values", headers=WAIT)
    assert values.status_code == 200, values.text
    assert "values" in values.json()
    one = watching.get("/microscope/devices/electron/voltage", headers=WAIT)
    assert one.status_code == 200, one.text
    missing = watching.get("/microscope/devices/electron/nothing", headers=WAIT)
    assert missing.status_code == 404

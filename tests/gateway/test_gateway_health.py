"""The gateway's health of the App Server and its microscope: one shape, from
the App Server's answers and its silences."""

from fibsem.gateway import health as h
from fibsem.gateway.gateway import STARTING, GatewayState

URL = "http://127.0.0.1:8001"
HEALTHY = {"connected": True, "halted": None}


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeApp:
    """The App Server's answers by route: (status, json), or an exception;
    *took* is how long each answer takes on the watch's timer."""

    def __init__(self, timer):
        self.answers = {"/health": (200, {"status": "ok"})}
        self.timer = timer
        self.took = 0.0
        self.asked = []

    def __call__(self, url, route, timeout):
        self.asked.append(route)
        self.timer.now += self.took
        found = self.answers[route]
        if isinstance(found, Exception):
            raise found
        return found


def watch(url=URL, switch=None):
    clock, timer = Clock(), Clock()
    app = FakeApp(timer)
    found = {"url": url}
    state = GatewayState(mode="running", version="test", switch=switch)
    w = h.HealthWatch(
        lambda: state,
        discovery=lambda: found if found["url"] else None,
        get=app,
        clock=clock,
        timer=timer,
    )
    return w, app, clock, found, state


def test_no_app_is_stopped_and_so_is_its_microscope():
    w, app, _, _, _ = watch(url=None)
    w.check_app()
    w.check_microscope()
    state = w.state()
    assert state.app.status == h.STOPPED
    assert state.microscope.status == h.STOPPED
    assert app.asked == []


def test_an_app_starting_is_starting_not_hung():
    w, app, _, _, _ = watch(url=None, switch=STARTING)
    w.check_app()
    assert w.state().app.status == h.STARTING
    w2, app2, _, _, _ = watch(switch=STARTING)
    app2.answers["/health"] = h.Unreachable("connection refused")
    w2.check_app()
    assert w2.state().app.status == h.STARTING


def test_the_app_is_ok_slow_or_not_answering_and_says_since_when():
    w, app, clock, _, _ = watch()
    w.check_app()
    first = w.state().app
    assert first.status == h.OK and first.since == 1000.0
    clock.now += 5
    w.check_app()
    assert w.state().app.since == 1000.0, "since stays while the status does"

    app.took = 2.0
    w.check_app()
    slow = w.state().app
    assert slow.status == h.SLOW and slow.answer_s == 2.0 and slow.since == 1005.0

    app.answers["/health"] = h.Unreachable("timed out")
    w.check_app()
    gone = w.state().app
    assert gone.status == h.NOT_ANSWERING
    assert "No answer in" in gone.detail


def test_the_microscope_while_the_app_does_not_answer_is_unknown():
    w, app, _, _, _ = watch()
    app.answers["/health"] = h.Unreachable("refused")
    w.check_app()
    w.check_microscope()
    assert w.state().microscope.status == h.UNKNOWN
    assert "/microscope/health" not in app.asked


def test_the_microscope_from_its_health_answer():
    w, app, _, _, _ = watch()
    w.check_app()
    for answer, status in (
        ({"connected": False}, h.NOT_CONNECTED),
        ({"connected": False, "connecting": "Connecting the stage"}, h.CONNECTING),
        ({"connected": False, "error": "no AutoScript"}, h.FAULT),
        ({**HEALTHY, "halted": "door open"}, h.HALTED),
        (HEALTHY, h.OK),
    ):
        app.answers["/microscope/health"] = (200, answer)
        w.check_microscope()
        assert w.state().microscope.status == status, answer
    app.answers["/microscope/health"] = h.Unreachable("timed out")
    w.check_microscope()
    assert w.state().microscope.status == h.NOT_ANSWERING


def test_a_slow_microscope_and_an_old_app_without_the_route():
    w, app, _, _, _ = watch()
    w.check_app()
    app.answers["/microscope/health"] = (200, HEALTHY)
    app.took = 2.0
    w.check_microscope()
    assert w.state().microscope.status == h.SLOW
    app.answers["/microscope/health"] = (404, {"detail": "Not Found"})
    w.check_microscope()
    assert w.state().microscope.status == h.UNKNOWN


def test_the_gateway_state_carries_the_health(tmp_path):
    from fibsem.gateway.gateway import Gateway

    w, _, _, _, _ = watch(url=None)
    gateway = Gateway(
        home=tmp_path,
        app_registry=lambda: {},
        last=lambda: None,
        recent=lambda: [],
        new_folder=lambda: str(tmp_path),
        configurations=lambda: [],
        discovery=lambda: None,
        health=w,
    )
    w.check_app()
    assert gateway.state().health.app.status == h.STOPPED


def _stage(**heard):
    return {"connected": True, "devices": {"stage": heard}}


def test_a_call_running_long_is_slow_then_not_answering_but_a_mill_is_not():
    now = 10_000.0
    move = {"call": "move_absolute", "since": now - 200}
    assert h.microscope_health(_stage(running=[move]), now)[0] == h.SLOW
    move["since"] = now - 700
    status, detail = h.microscope_health(_stage(running=[move]), now)
    assert status == h.NOT_ANSWERING and "stage.move_absolute" in detail
    mill = {"call": "run", "since": now - 3600, "long": True}
    answer = {"connected": True, "services": {"milling": {"running": [mill]}}}
    assert h.microscope_health(answer, now)[0] == h.OK


def test_a_recent_driver_error_is_a_fault_until_a_call_comes_back():
    now = 10_000.0
    failed = dict(
        last_error="RuntimeError: lost",
        last_error_call="move_absolute",
        last_error_at=now - 60,
    )
    status, detail = h.microscope_health(_stage(**failed), now)
    assert status == h.FAULT and "stage.move_absolute failed" in detail
    assert h.microscope_health(_stage(**failed, last_ok=now - 10), now)[0] == h.OK
    old = {**failed, "last_error_at": now - 3 * 3600}
    assert h.microscope_health(_stage(**old), now)[0] == h.OK


def test_the_watch_keeps_what_each_device_said():
    w, app, _, _, _ = watch()
    w.check_app()
    app.answers["/microscope/health"] = (200, _stage(last_ok=1.0))
    w.check_microscope()
    assert w.state().heard["devices"]["stage"]["last_ok"] == 1.0
    app.answers["/microscope/health"] = (200, {"connected": True, "devices": 5})
    w.check_microscope()
    assert w.state().microscope.status == h.UNKNOWN

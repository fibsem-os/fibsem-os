"""What a device can't do now, and why: a command's availability with its reason, a
parameter that refuses an out-of-range value, and the instrument's Safety Halt.

A command that is not available is listed with its reason and refused at the App
Server. A parameter declared ``out_of_range="refuse"`` refuses a value outside its
limits instead of clipping it. A Safety Halt refuses every command and write on
every device of the instrument, whoever calls, until the operator clears it; reads
and safety commands still run.
"""

import pytest
from fastapi.testclient import TestClient

from fibsem.devices import (
    CommandUnavailable,
    Device,
    DeviceParameter,
    ParameterOutOfRange,
    ParameterReadOnly,
    Resources,
    SafetyHalted,
    command,
)
from fibsem.devices.errors import error_from
from fibsem.server.devices import build_device_app
from fibsem.structures.params import ParameterMetadata, RangeLimit


class Column(Device):
    """A column with a lockable focus: the kind of rule an instrument enforces."""

    focus = DeviceParameter(float, unit="m", out_of_range="refuse")
    gain = DeviceParameter(float)

    def __init__(self, name, resources=None):
        super().__init__(name, resources=resources)
        self.locked = False
        self.values = {"focus": 0.0, "gain": 0.5}
        self.blanked = False

    def read_focus(self):
        return self.values["focus"]

    def write_focus(self, value):
        self.values["focus"] = value

    def metadata_focus(self):
        return ParameterMetadata(limits=RangeLimit(-10e-6, 0.0))

    def read_gain(self):
        return self.values["gain"]

    def write_gain(self, value):
        self.values["gain"] = value

    def metadata_gain(self):
        if self.locked:
            return ParameterMetadata(
                limits=RangeLimit(0.0, 1.0),
                settable=False,
                reason="locked after alignment",
            )
        return ParameterMetadata(limits=RangeLimit(0.0, 1.0))

    @command(available=lambda self: True if self.locked else "not locked yet")
    def recalibrate(self) -> str:
        return "recalibrated"

    @command
    def acquire(self) -> str:
        return "image"

    @command(safety=True)
    def blank(self) -> None:
        # A safety command's own writes pass the halt.
        self.gain.set_value(0.0)
        self.blanked = True


class Halts(Device):
    """What sets and clears the Safety Halt: the instrument kind's own service."""

    @command
    def halt(self, reason: str) -> None:
        self.safety_halt.set(reason, clear_command=f"{self.name}.resume")

    @command(safety=True, operator_only="the operator resumes after a halt")
    def resume(self) -> None:
        self.safety_halt.clear()


@pytest.fixture
def instrument():
    shared = Resources()
    return (
        Column("column", resources=shared).connect(),
        Halts("session", resources=shared).connect(),
    )


@pytest.fixture
def client(instrument):
    app = build_device_app(list(instrument))
    with TestClient(app) as client:
        client.headers["Authorization"] = f"Bearer {app.state.auth.token}"
        yield client


# -- availability with a reason ---------------------------------------------------


def test_a_command_lists_why_it_is_not_available(instrument):
    column, _ = instrument
    info = column.commands["recalibrate"]
    assert (info.available, info.reason) == (False, "not locked yet")

    column.locked = True
    info = column.commands["recalibrate"]
    assert (info.available, info.reason) == (True, "")


def test_the_server_refuses_an_unavailable_command_with_its_reason(client):
    listed = client.get("/devices/column").json()["commands"]["recalibrate"]
    assert (listed["available"], listed["reason"]) == (False, "not locked yet")

    r = client.post("/devices/column/commands/recalibrate", json={"kwargs": {}})

    assert r.status_code == 409
    assert r.json()["detail"] == {
        "error_type": "CommandUnavailable",
        "message": "column.recalibrate is not available now: not locked yet",
    }


def test_a_parameter_not_settable_now_says_why(client, instrument):
    column, _ = instrument
    column.locked = True
    column.gain.refresh_metadata()

    described = client.get("/devices/column").json()["parameters"]["gain"]
    assert (described["settable"], described["reason"]) == (
        False,
        "locked after alignment",
    )
    r = client.put("/devices/column/gain", json={"value": 0.2})
    assert r.status_code == 409
    assert "locked after alignment" in r.json()["detail"]["message"]


def test_the_refusals_arrive_as_themselves_on_a_remote_copy():
    for kind, error in (
        ("CommandUnavailable", CommandUnavailable),
        ("SafetyHalted", SafetyHalted),
        ("ParameterOutOfRange", ParameterOutOfRange),
    ):
        raised = error_from({"error_type": kind, "message": "why"})
        assert isinstance(raised, error)


# -- refuse instead of clip -------------------------------------------------------


def test_a_refusing_parameter_refuses_and_writes_nothing(instrument):
    column, _ = instrument

    with pytest.raises(ParameterOutOfRange, match="outside"):
        column.focus.set_value(2e-6)

    assert column.values["focus"] == 0.0


def test_a_clipping_parameter_still_clips(instrument, caplog):
    column, _ = instrument

    assert column.gain.set_value(1.5) == 1.0
    assert "set to 1.0" in caplog.text


def test_the_server_answers_an_out_of_range_value_as_422(client):
    r = client.put("/devices/column/focus", json={"value": 2e-6})

    assert r.status_code == 422
    assert r.json()["detail"]["error_type"] == "ParameterOutOfRange"


def test_out_of_range_takes_only_clip_or_refuse():
    with pytest.raises(ValueError, match="clip"):
        DeviceParameter(float, out_of_range="ignore")


# -- the Safety Halt --------------------------------------------------------------


def test_a_safety_halt_refuses_commands_and_writes_on_every_device(instrument):
    column, session = instrument
    session.halt("heartbeat lost")

    with pytest.raises(SafetyHalted, match="heartbeat lost"):
        column.acquire()
    with pytest.raises(SafetyHalted, match=r"session\.resume clears it"):
        column.gain.set_value(0.2)
    with pytest.raises(SafetyHalted):
        column.gain.write_through(0.2)
    assert column.gain.get_value() == 0.5  # reads still work


def test_safety_commands_run_during_a_halt_with_their_own_writes(instrument):
    column, session = instrument
    session.halt("heartbeat lost")

    column.blank()

    assert column.blanked
    assert column.values["gain"] == 0.0


def test_a_halt_lists_everything_but_safety_commands_as_unavailable(instrument):
    column, session = instrument
    seen = []
    column.metadata_changed.connect(lambda name, metadata: seen.append(name))

    session.halt("heartbeat lost")

    assert column.commands["acquire"].available is False
    assert column.commands["acquire"].reason == "Safety Halt: heartbeat lost"
    assert column.commands["blank"].available is True
    assert session.commands["resume"].available is True
    assert column.gain.settable is False
    assert column.gain.reason == "Safety Halt: heartbeat lost"
    assert sorted(seen) == ["focus", "gain"]  # the page greys out at once


def test_clearing_the_halt_puts_everything_back(instrument):
    column, session = instrument
    session.halt("heartbeat lost")

    session.resume()

    assert column.acquire() == "image"
    assert column.gain.set_value(0.2) == 0.2
    assert column.commands["acquire"].available is True


def test_the_server_refuses_during_a_halt_and_lets_safety_through(client, instrument):
    _, session = instrument
    session.halt("heartbeat lost")

    r = client.post("/devices/column/commands/acquire", json={"kwargs": {}})
    assert r.status_code == 409
    assert "Safety Halt: heartbeat lost" in r.json()["detail"]["message"]
    r = client.put("/devices/column/gain", json={"value": 0.2})
    assert r.status_code == 409
    metadata = client.get("/devices/column/gain/metadata").json()
    assert (metadata["settable"], metadata["reason"]) == (
        False,
        "Safety Halt: heartbeat lost",
    )
    assert (
        client.post("/devices/column/commands/blank", json={"kwargs": {}}).status_code
        == 200
    )


def test_a_device_on_its_own_resources_is_not_halted(instrument):
    _, session = instrument
    other = Column("other").connect()
    session.halt("heartbeat lost")

    assert other.acquire() == "image"


def test_a_halt_needs_a_reason(instrument):
    _, session = instrument
    with pytest.raises(ValueError, match="reason"):
        session.safety_halt.set("")


def test_a_read_only_parameter_without_a_reason_is_unchanged(instrument):
    column, _ = instrument

    class Fixed(Device):
        size = DeviceParameter(float)

        def read_size(self):
            return 1.0

    fixed = Fixed("fixed").connect()
    assert fixed.size.reason == ""
    with pytest.raises(ParameterReadOnly, match="read-only$"):
        fixed.size.set_value(2.0)

"""The client's ``app``: the session of the app the App Server hosts, its calls the
agent's tools, offered by the parts the server serves.

Skipped when the [server] extra is not installed.
"""

import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from psygnal.containers import EventedDict  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.client import FibsemClient, ServerError  # noqa: E402
from fibsem.server import AuthConfig, build_server  # noqa: E402
from fibsem.structures import MicroscopeState  # noqa: E402
from tests.run_state import stub_session  # noqa: E402
from tests.server.test_client import TOKEN, InProcess  # noqa: E402


@pytest.fixture(scope="module")
def microscope():
    os.environ.setdefault("FIBSEM_SIM_NO_DELAY", "1")
    microscope, _ = utils.setup_session(manufacturer="Demo", ip_address="localhost")
    return microscope


@pytest.fixture
def auth():
    return AuthConfig.generate(level="drive", token=TOKEN)


@pytest.fixture
def host(tmp_path, microscope):
    host = stub_session()
    exp = Experiment(path=tmp_path / "exp", name="client-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    host.experiment = exp
    host.microscope = microscope
    return host


def _client(microscope, auth, app_context=None):
    app = build_server(microscope, app_context=app_context, auth=auth)
    return FibsemClient(transport=InProcess(app))


@pytest.fixture
def client(microscope, auth, host):
    from fibsem.workflows.events import EventBuffer

    context = AutoLamellaAppContext(host, event_buffer=EventBuffer())
    with _client(microscope, auth, context) as client:
        yield client


def test_the_app_offers_the_core_and_its_parts(client):
    app = client.app
    assert app.served and app.name == "AutoLamella"
    assert app.parts == ["items", "poses", "grids", "milling_stages", "lamellae"]
    assert {"app_get_status", "proposal_list", "item_get", "grid_list"} <= set(
        app.tools
    )
    assert all(spec.router == "app" for spec in app.tools.values())
    assert "item_get" in dir(app)


def test_a_read_takes_its_path_values(client):
    assert client.app.app_get_status()["experiment"]["name"] == "client-exp"
    (item,) = client.app.item_list()["items"]
    name = item["item_name"]
    assert client.app.item_get(item_name=name)["item_name"] == name


def test_a_call_lands_and_a_read_takes_its_query(client):
    assert client.app.experiment_add_note(text="checked from the client")["recorded"]
    events = client.app.app_get_events(since=0)
    (note,) = [e for e in events["events"] if e["kind"] == "note"]
    assert note["payload"]["text"] == "checked from the client"
    later = client.app.app_get_events(since=events["latest_seq"])
    assert later["events"] == []


def test_a_refusal_raises(client, auth):
    auth.set_level("watch")
    with pytest.raises(ServerError) as refused:
        client.app.workflow_stop(reason="test")
    assert refused.value.status == 403


class _CoreOnly:
    """An app context with no optional parts."""

    app_name = "Core"

    def status(self):
        return {"core": True}


def test_a_part_the_app_lacks_is_not_offered(microscope, auth):
    with _client(microscope, auth, _CoreOnly()) as client:
        assert client.app.parts == []
        assert client.app.app_get_status() == {"core": True}
        assert "grid_list" not in client.app.tools
        with pytest.raises(AttributeError, match="no tool 'grid_list'"):
            client.app.grid_list()


def test_no_app_offers_nothing(microscope, auth):
    with _client(microscope, auth) as client:
        assert not client.app.served
        assert client.app.tools == {}
        with pytest.raises(AttributeError, match="hosts no app"):
            client.app.app_get_status()

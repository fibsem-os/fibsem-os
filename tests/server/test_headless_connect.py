"""What the host connects to: the default configuration, as the app's own
Connect does, unless a flag names another microscope; with the window, only
when a flag names one."""

from fibsem import utils
from fibsem.server import headless


def connected(monkeypatch, argv, windowed=False):
    calls = []

    def setup_session(**kwargs):
        calls.append(kwargs)
        raise ConnectionError("not here")

    monkeypatch.setattr(utils, "setup_session", setup_session)
    try:
        headless.open_session(
            headless.connection_parser().parse_args(argv), windowed=windowed
        )
    except ConnectionError:
        pass
    return calls


def test_with_no_flags_it_connects_to_the_default_configuration(monkeypatch):
    (call,) = connected(monkeypatch, [])
    # No file and no override: setup_session reads the default configuration.
    assert call["config_path"] is None
    assert call["manufacturer"] is None and call["ip_address"] is None


def test_a_flag_still_names_another_microscope(monkeypatch):
    (call,) = connected(monkeypatch, ["--manufacturer", "Demo"])
    assert call["manufacturer"] == "Demo"
    (call,) = connected(
        monkeypatch, ["--config", "arctis.yaml", "--ip-address", "10.0.0.2"]
    )
    assert str(call["config_path"]) == "arctis.yaml"
    assert call["ip_address"] == "10.0.0.2"


def test_with_the_window_it_connects_only_when_a_flag_names_one(monkeypatch):
    assert connected(monkeypatch, [], windowed=True) == []
    assert connected(monkeypatch, ["--debug"], windowed=True) == []
    (call,) = connected(monkeypatch, ["--manufacturer", "Demo"], windowed=True)
    assert call["manufacturer"] == "Demo"


def test_no_connect_starts_with_no_microscope_unless_a_flag_names_one(monkeypatch):
    assert connected(monkeypatch, ["--no-connect"]) == []
    (call,) = connected(monkeypatch, ["--no-connect", "--manufacturer", "Demo"])
    assert call["manufacturer"] == "Demo"

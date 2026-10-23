"""What a headless server connects to: the default configuration, as the
app's own Connect does, unless a flag names another microscope."""

from fibsem.server import headless


def connected(monkeypatch, argv):
    calls = []

    def setup_session(**kwargs):
        calls.append(kwargs)
        return object(), None

    monkeypatch.setattr(headless.utils, "setup_session", setup_session)
    headless.connect(headless.connection_parser().parse_args(argv))
    (call,) = calls
    return call


def test_with_no_flags_it_connects_to_the_default_configuration(monkeypatch):
    call = connected(monkeypatch, [])
    # No file and no override: setup_session reads the default configuration.
    assert call["config_path"] is None
    assert call["manufacturer"] is None and call["ip_address"] is None


def test_a_flag_still_names_another_microscope(monkeypatch):
    assert connected(monkeypatch, ["--manufacturer", "Demo"])["manufacturer"] == "Demo"
    call = connected(
        monkeypatch, ["--config", "arctis.yaml", "--ip-address", "10.0.0.2"]
    )
    assert str(call["config_path"]) == "arctis.yaml"
    assert call["ip_address"] == "10.0.0.2"

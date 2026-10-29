"""A Device Server started from its own file, as a METEOR PC runs it.

``python -m fibsem.server.devices --config ~/.fibsem/device-server.yaml``: the file
names the FM as an ``fm`` entry and the address to serve on.
"""

import pytest
import yaml

from fibsem.devices.entries import disconnect_devices
from fibsem.server import devices as server_devices
from fibsem.server.devices import configured_devices

FM_PARTS = ["camera", "filter_set", "fm", "light_source", "objective"]


def _server_file(tmp_path, *entries, **settings):
    path = tmp_path / "device-server.yaml"
    path.write_text(
        yaml.safe_dump({**settings, "hardware": {"devices": list(entries)}})
    )
    return str(path)


def test_an_fm_entry_is_served_as_its_parts(tmp_path):
    path = _server_file(
        tmp_path,
        {"name": "fm", "type": "fm", "mount_transform": "flip-x"},
        info={"manufacturer": "Demo"},
    )
    built, cleanups = configured_devices(path)
    try:
        assert sorted(built) == FM_PARTS
        camera = built["camera"]
        assert camera.mount_transform.get_value() == "flip-x"
    finally:
        disconnect_devices(built, cleanups)


def test_the_serve_flag_is_the_same_one_entry_file():
    built, cleanups = configured_devices(server_devices.served_fm_settings("fm"))
    try:
        assert sorted(built) == FM_PARTS
    finally:
        disconnect_devices(built, cleanups)


def test_a_remote_entry_is_refused(tmp_path):
    path = _server_file(
        tmp_path,
        {"name": "fm", "type": "fm", "driver": "remote", "address": "10.0.0.2"},
    )
    with pytest.raises(ValueError, match="host's configuration"):
        configured_devices(path)


def test_keys_the_host_owns_are_refused(tmp_path):
    path = _server_file(
        tmp_path,
        {"name": "fm", "type": "fm", "driver": "Demo", "camera_tilt": 17},
    )
    with pytest.raises(ValueError, match="camera_tilt.*host's remote fm entry"):
        configured_devices(path)


def test_the_file_names_where_to_serve(tmp_path, monkeypatch):
    path = _server_file(
        tmp_path,
        {"name": "fm", "type": "fm"},
        info={"manufacturer": "Demo"},
        server={"host": "0.0.0.0", "port": 9001},
    )
    served = {}
    monkeypatch.setattr(
        server_devices.uvicorn,
        "run",
        lambda app, host, port, **_: served.update(host=host, port=port),
    )

    server_devices.main(["--config", path, "--token-file", str(tmp_path / "t.json")])
    assert served == {"host": "0.0.0.0", "port": 9001}

    # a flag still wins
    server_devices.main(
        ["--config", path, "--port", "9002", "--token-file", str(tmp_path / "t.json")]
    )
    assert served == {"host": "0.0.0.0", "port": 9002}


def test_a_file_that_builds_nothing_stops(tmp_path, caplog):
    path = _server_file(
        tmp_path,
        {"name": "fm", "type": "fm", "driver": "Demo", "enabled": False},
    )
    with pytest.raises(SystemExit):
        server_devices.main(["--config", path])

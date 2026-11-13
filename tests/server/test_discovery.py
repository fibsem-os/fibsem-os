"""The discovery file."""

import os
import subprocess
import sys

import pytest

from fibsem.server import discovery


def test_a_client_beside_a_second_host_reads_that_hosts_file(tmp_path):
    # An eval run's agent is told its own server's file, not this PC's.
    found = tmp_path / "app-server.json"
    code = (
        "from fibsem.server import discovery as d; "
        "print(d.DISCOVERY_FILE == d.LEGACY_DISCOVERY_FILE, d.DISCOVERY_FILE)"
    )
    env = {**os.environ, discovery.DISCOVERY_FILE_ENV: str(found)}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    ).stdout.split()
    assert out == ["True", str(found)]


def test_a_client_works_with_a_server_of_the_same_major_and_minor_only():
    major, minor, _ = discovery.API_VERSION.split(".")
    same = {"api_version": f"{major}.{minor}.99"}
    assert discovery.api_mismatch(same, "fibsem-mcp") is None
    discovery.check_api(same, "fibsem-mcp")

    newer = {"api_version": f"{major}.{int(minor) + 1}.0", "fibsem": {"version": "9.9"}}
    why = discovery.api_mismatch(newer, "fibsem-mcp")
    assert newer["api_version"] in why and "fibsem 9.9" in why
    assert discovery.API_VERSION in why and "same fibsem version" in why

    for other in (
        {"api_version": f"{int(major) + 1}.{minor}.0"},
        {},
        {"api_version": "x"},
    ):
        with pytest.raises(discovery.ApiMismatch):
            discovery.check_api(other, "fibsem-mcp")


def test_the_server_serves_the_version_clients_check():
    from fibsem.server import server

    assert server.API_VERSION is discovery.API_VERSION

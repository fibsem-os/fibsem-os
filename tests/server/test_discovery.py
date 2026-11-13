"""The discovery file."""

import os
import subprocess
import sys

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

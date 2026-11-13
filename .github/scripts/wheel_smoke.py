"""Install a built wheel the way a user does and start it.

The test jobs import fibsem from the checkout, so a file the wheel leaves out passes
them. This installs the wheel into a fresh environment outside the repository, then
runs ``fibsem-cli --help`` and starts the App Server on the simulator and asks it for
``/health``. Nothing it runs can see the checkout.

Usage: ``python .github/scripts/wheel_smoke.py dist/fibsem-*.whl [--extras ui]
[--freeze PATH]``. With ``--freeze``, the versions the environment installed are
written to PATH as a pip constraints file.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

START_TIMEOUT_S = 180


def _run(cmd, **kwargs) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _scripts(venv: Path) -> Path:
    return venv / ("Scripts" if os.name == "nt" else "bin")


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--extras", default="", help="e.g. ui")
    parser.add_argument("--freeze", type=Path, metavar="PATH")
    args = parser.parse_args()

    wheel = args.wheel.resolve()
    repo = Path(__file__).resolve().parents[2]
    root = Path(tempfile.mkdtemp(prefix="fibsem-wheel-"))
    venv = root / "venv"
    home = root / "home"
    home.mkdir()
    # A first launch: no ~/.fibsem from an earlier run, and no PYTHONPATH.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(HOME=str(home), USERPROFILE=str(home), PYTHONNOUSERSITE="1")

    _run([sys.executable, "-m", "venv", str(venv)])
    bin_dir = _scripts(venv)
    python = bin_dir / ("python.exe" if os.name == "nt" else "python")
    spec = f"fibsem[{args.extras}] @ {wheel.as_uri()}" if args.extras else str(wheel)
    _run([python, "-m", "pip", "install", "--upgrade", "pip"], cwd=root, env=env)
    _run([python, "-m", "pip", "install", spec], cwd=root, env=env)

    where = _run(
        [python, "-c", "import fibsem; print(fibsem.__file__)"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print("fibsem imported from", where)
    if Path(where).resolve().is_relative_to(repo):
        print("error: fibsem was imported from the checkout, not the wheel")
        return 1

    _run([bin_dir / "fibsem-cli", "--help"], cwd=root, env=env, stdout=subprocess.DEVNULL)

    port = _free_port()
    log = root / "server.log"
    cmd = [
        bin_dir / "fibsem-autolamella-server",
        "--port",
        str(port),
        "--discovery",
        str(root / "app-server.json"),
    ]
    print("+", " ".join(str(c) for c in cmd), flush=True)
    with open(log, "w") as out:
        server = subprocess.Popen(
            cmd, cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + START_TIMEOUT_S
            while not _health(port):
                if server.poll() is not None:
                    print(f"error: the server exited ({server.returncode})")
                    return 1
                if time.monotonic() > deadline:
                    print(f"error: no /health answer in {START_TIMEOUT_S} s")
                    return 1
                time.sleep(1)
            print(f"/health answered on port {port}")
        finally:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
            print("---- server log ----")
            print(log.read_text(errors="replace"))

    if args.freeze is not None:
        frozen = _run(
            [python, "-m", "pip", "freeze", "--exclude", "fibsem"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        ).stdout
        header = (
            f"# Versions fibsemOS was installed and started with on {sys.platform}, "
            f"Python {sys.version_info.major}.{sys.version_info.minor}.\n"
            "#   pip install \"fibsem[ui]==VERSION\" -c THIS_FILE\n"
        )
        args.freeze.write_text(header + frozen)
        print(f"wrote {args.freeze}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

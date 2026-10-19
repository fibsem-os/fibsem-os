import os
from pathlib import Path

REPO = Path(__file__).parents[3]


def recording_env() -> dict:
    """The environment a recording script runs in: no simulated delays, and this
    checkout's ``fibsem`` first, not whichever one the interpreter has installed."""
    path = os.pathsep.join(filter(None, [str(REPO), os.environ.get("PYTHONPATH")]))
    return dict(os.environ, FIBSEM_SIM_NO_DELAY="1", PYTHONPATH=path)

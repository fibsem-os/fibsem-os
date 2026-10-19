"""The agent level (FIB-1035), importable without the [server] extra.

The App Server's auth (``fibsem.server.auth``) gates each route on a level and
needs FastAPI; the GUI only needs the names, and must run where FastAPI is not
installed.
"""

from enum import Enum


class AgentLevel(str, Enum):
    """How far the agent may act in the App Server (FIB-1035): the one knob the
    operator sees, and the server's only gate: each route needs a level.

    - ``off``: every agent-token request is refused, reads included.
    - ``watch``: reads everything and adds notes; acts on nothing, stops nothing.
    - ``assist``: as Drive, but each action is a request the operator approves
      (FIB-1283); stopping needs no approval.
    - ``drive``: acts on the session: starts, stops, schedules, answers, edits;
      and on the microscope, when the devices setting is on (FIB-1267).
    """

    OFF = "off"
    WATCH = "watch"
    ASSIST = "assist"
    DRIVE = "drive"


# How recently the agent must have been heard to count as listening: someone
# who can answer a run's questions at Drive. A supervising agent long-polls the
# event stream, so a connected one is heard every few seconds; past this it has
# gone, and a held question waits for it to come back.
AGENT_LISTENING_S = 120.0

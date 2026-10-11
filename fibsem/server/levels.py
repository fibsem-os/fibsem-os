"""The agent level (FIB-1035), importable without the [server] extra.

The App Server's auth (``fibsem.server.auth``) maps each level to scopes and
needs FastAPI; the GUI only needs the names, and must run where FastAPI is not
installed.
"""

from enum import Enum


class AgentLevel(str, Enum):
    """How far the agent may act in the App Server (FIB-1035): the one knob the
    operator sees. Scopes stay the server's mechanism underneath; each level
    arms a set of them (``LEVEL_SCOPES``).

    - ``off``: every agent-token request is refused, reads included.
    - ``watch``: reads everything and adds notes; acts on nothing, stops nothing.
    - ``assist``: as Drive, but each action is a request the operator approves
      (FIB-1283). Defined, not offered yet.
    - ``drive``: acts on the session: starts, stops, schedules, answers, edits.

    Hardware joins Assist and Drive in FIB-1267.
    """

    OFF = "off"
    WATCH = "watch"
    ASSIST = "assist"
    DRIVE = "drive"

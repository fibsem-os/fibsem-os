"""The agent level, importable without the [server] extra.

The App Server's auth (``fibsem.server.auth``) gates each route on a level and
needs FastAPI; the GUI only needs the names, and must run where FastAPI is not
installed.
"""

from enum import Enum
from typing import Any, Dict, Optional


class AgentLevel(str, Enum):
    """How far the agent may act in the App Server: the one knob the
    operator sees, and the server's only gate: each route needs a level.

    - ``off``: every agent-token request is refused, reads included.
    - ``watch``: reads everything and adds notes; acts on nothing, stops nothing.
    - ``assist``: as Drive, but each action is a request the operator approves;
      stopping needs no approval.
    - ``drive``: acts on the session: starts, stops, schedules, answers, edits;
      and on the microscope, when the devices setting is on.
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


# Each level as the operator reads it: its name, the chip's word, and what it
# allows. The window's chip and menu and the web app's say the same.
LEVEL_NAMES = {
    AgentLevel.OFF: "Off",
    AgentLevel.WATCH: "Watch",
    AgentLevel.ASSIST: "Assist",
    AgentLevel.DRIVE: "Drive",
}
LEVEL_WORDS = {
    AgentLevel.OFF: "Agent: Off",
    AgentLevel.WATCH: "Agent: Watching",
    AgentLevel.ASSIST: "Agent: Assisting",
    AgentLevel.DRIVE: "Agent: Driving",
}
# What an agent asking to connect could do once allowed, at the level now.
_ALLOWED_AT = {
    AgentLevel.OFF: "now Off, so it can do nothing",
    AgentLevel.WATCH: "now Watch, so it can look but not act",
    AgentLevel.ASSIST: "now Assist, so it asks you before it acts",
    AgentLevel.DRIVE: "now Drive, so it can act on the microscope",
}


def if_allowed(level: AgentLevel, replaces: Optional[str] = None) -> str:
    """What allowing an agent to connect means, at *level*: the card's words.
    One agent per microscope: allowing it unpairs *replaces*, if one is."""
    words = (
        "It can read the session, and acts only at the level you set: "
        f"{_ALLOWED_AT[AgentLevel(level)]}. It stays allowed until you unpair "
        "it in Connect an agent."
    )
    if replaces:
        words += f" It replaces {replaces}, which is unpaired."
    return words


LEVEL_TIPS = {
    AgentLevel.OFF: "No access",
    AgentLevel.WATCH: "Read only",
    AgentLevel.ASSIST: "Asks first",
    AgentLevel.DRIVE: "Full control",
}

# The skill an agent works by at each level on AutoLamella work, after the
# fibsem-agent and autolamella-agent skills. Off has none: the agent is refused
# everything.
LEVEL_SKILLS = {
    AgentLevel.WATCH: "observe-autolamella",
    AgentLevel.ASSIST: "assist-autolamella",
    AgentLevel.DRIVE: "supervise-autolamella",
}


def _doing(tool: Any) -> str:
    """What a request's step is doing, in a word: "applying" read as writing a
    config whatever the step was."""
    tool = str(tool or "")
    if tool.startswith(("milling_", "spot_burn")):
        return "milling"
    if tool.startswith(("stage_", "item_move_", "grid_load", "grid_unload")):
        return "moving"
    if "acquire" in tool or tool.startswith(("fm_auto_focus", "beam_auto_")):
        return "imaging"
    return "running"


def chip_request_text(request: Dict[str, Any]) -> str:
    """The chip's word while a request waits or is being applied."""
    if request.get("status") == "applying":
        actions = request.get("actions", [])
        total = len(actions)
        step = min(len(request.get("results", [])) + 1, total)
        doing = _doing(actions[step - 1].get("tool")) if actions else "running"
        return f"Agent: {doing} {step}/{total}"
    return "Agent asks"


def chip_text(
    level: Optional[AgentLevel], request: Optional[Dict[str, Any]], suggests: bool
) -> str:
    """The chip's word: what the agent asks for or is applying, that it suggests a
    decision, or else its level."""
    if request is not None:
        return chip_request_text(request)
    if suggests:
        return "Agent suggests"
    return LEVEL_WORDS[level] if level is not None else ""

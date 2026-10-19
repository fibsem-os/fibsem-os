"""The agent's look, shared by the status bar chip and the agent card: one style
for each level, and Stop agent drawn the same way, so the controls read as one
set wherever they show."""

from __future__ import annotations

from fibsem.ai.agent.levels import AgentLevel
from fibsem.ui.tokens import (
    BORDER_COLOR,
    PURPLE_COLOR,
    RED_COLOR,
    REVIEW_COLOR,
    TEXT_MUTED_COLOR,
    TEXT_STRONG_COLOR,
)

CHIP = """
    QPushButton {{
        color: {color};
        background-color: {background};
        border: {border};
        padding: 4px 10px;
        border-radius: 3px;
        font-weight: bold;
    }}
    QPushButton::menu-indicator {{ width: 0px; }}
"""


def _tint(colour: str, alpha: float) -> str:
    return "rgba({}, {}, {}, {})".format(
        *(int(colour[i : i + 2], 16) for i in (1, 3, 5)), alpha
    )


# Assisting's fill: the purple at a third strength, between Watching's outline and
# Driving's solid fill.
PURPLE_TINT = _tint(PURPLE_COLOR, 0.35)


def pulse_chip(strength: float) -> str:
    """The chip while the agent acts on the microscope: its purple fill at
    *strength* (0 to 1), from Assisting's tint up to Driving's solid."""
    return CHIP.format(
        color=TEXT_STRONG_COLOR,
        background=_tint(PURPLE_COLOR, 0.35 + 0.65 * strength),
        border=f"1px solid {PURPLE_COLOR}",
    )


# Something waiting for the operator, or being applied: amber, as the Review
# tab's waiting decisions.
REQUEST_CHIP = CHIP.format(
    color=TEXT_STRONG_COLOR,
    background=_tint(REVIEW_COLOR, 0.30),
    border=f"1px solid {REVIEW_COLOR}",
)

# Stop agent: the chip's shape, outlined red as the queue's Stop Task.
STOP_CHIP = (
    CHIP.format(
        color=TEXT_STRONG_COLOR,
        background="transparent",
        border=f"1px solid {RED_COLOR}",
    )
    + f"QPushButton:hover {{ background-color: {_tint(RED_COLOR, 0.25)}; }}"
)

# Each level: its name, the chip's word, its look, and what it allows.
LEVELS = {
    AgentLevel.OFF: (
        "Off",
        "Agent: Off",
        CHIP.format(color=TEXT_MUTED_COLOR, background="transparent", border="none"),
        "No access",
    ),
    AgentLevel.WATCH: (
        "Watch",
        "Agent: Watching",
        CHIP.format(
            color=TEXT_STRONG_COLOR,
            background="transparent",
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Read only",
    ),
    AgentLevel.ASSIST: (
        "Assist",
        "Agent: Assisting",
        CHIP.format(
            color=TEXT_STRONG_COLOR,
            background=PURPLE_TINT,
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Asks first",
    ),
    AgentLevel.DRIVE: (
        "Drive",
        "Agent: Driving",
        CHIP.format(
            color=TEXT_STRONG_COLOR,
            background=PURPLE_COLOR,
            border=f"1px solid {PURPLE_COLOR}",
        ),
        "Full control",
    ),
}

LEVEL_NAMES = {level: name for level, (name, _, _, _) in LEVELS.items()}
# The chip's word for each level, for anywhere else that names it.
LEVEL_WORDS = {level: word for level, (_, word, _, _) in LEVELS.items()}
LEVEL_STYLES = {level: style for level, (_, _, style, _) in LEVELS.items()}
LEVEL_TIPS = {level: tip for level, (_, _, _, tip) in LEVELS.items()}

# The level menu: roomier than the app's default; its rows paint their own hover
# (a QWidgetAction's row gets none from QMenu::item:selected).
LEVEL_MENU = f"""
    QMenu {{ padding: 4px 0px; }}
    #levelRow {{ background: transparent; }}
    #levelRow:hover {{ background: {BORDER_COLOR}; }}
    #levelRow QLabel {{ background: transparent; }}
"""

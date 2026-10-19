"""Attention Required, or Agent deciding when the agent at Drive is listening
(round 10, FIB-1498)."""

from unittest.mock import MagicMock

from fibsem.applications.autolamella.ui.attention import AttentionRouter
from fibsem.workflows.status import Hold, HoldKind


def _router(qapp):
    controls = MagicMock()
    router = AttentionRouter(
        prompts=MagicMock(),
        tab_widget=MagicMock(),
        controls=controls,
        status_bar=MagicMock(),
        timeline=MagicMock(),
        sound_enabled=lambda: True,
    )
    return router, controls


def test_a_hold_the_agent_is_deciding_says_so_in_the_agent_s_colour(qapp):
    router, controls = _router(qapp)
    hold = Hold(kind=HoldKind.proposal, releases="decide Setup", agent_deciding=True)
    router.show_hold(hold)
    label, tooltip = controls.set_attention.call_args.args
    assert label == "Agent deciding" and "decide it yourself" in tooltip
    assert controls.set_attention.call_args.kwargs == {"agent": True}

    router.show_hold(Hold(kind=HoldKind.proposal, releases="decide Setup"))
    assert controls.set_attention.call_args.args[0] == "Attention Required"

"""AutoLamella's workflow buttons as one component (FIB-1188): what each state shows,
and that each button says it was pressed -- what that does is the window's."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from fibsem.applications.autolamella.ui.workflow_controls import (  # noqa: E402
    AGENT,
    AUTOMATED,
    SUPERVISED,
    WorkflowControls,
)


@pytest.fixture
def controls(qapp):
    controls = WorkflowControls()
    yield controls
    controls.deleteLater()


def _shown(controls):
    buttons = {
        "attention": controls.attention_btn,
        "supervision": controls.supervision_btn,
        "run": controls.run_btn,
        "stop": controls.stop_btn,
    }
    return {name for name, button in buttons.items() if not button.isHidden()}


def test_at_rest_it_offers_run_disabled(controls):
    assert _shown(controls) == {"run"}
    assert not controls.run_btn.isEnabled(), "nothing is selected yet"


def test_running_swaps_run_for_stop_and_back(controls):
    controls.set_running(True)
    assert _shown(controls) == {"stop"}
    controls.set_supervision(SUPERVISED, "Polishing")
    assert _shown(controls) == {"stop", "supervision"}
    controls.set_running(False)
    assert _shown(controls) == {"run"}, "the chip goes with the run"


@pytest.mark.parametrize(
    "mode, text",
    [(SUPERVISED, "Supervised"), (AUTOMATED, "Automated"), (AGENT, "Agent")],
)
def test_the_chip_names_the_mode_and_the_task(controls, mode, text):
    controls.set_supervision(mode, "Polishing")
    assert controls.supervision_btn.text() == text
    assert controls.supervision_btn.toolTip().startswith("Polishing ")


def test_attention_shows_what_it_is_told_and_hides_on_none(controls):
    controls.set_attention("Review Required (2)", "decide 01-a and 02-b")
    assert controls.attention_btn.text() == "Review Required (2)"
    assert controls.attention_btn.toolTip() == "decide 01-a and 02-b"
    assert "attention" in _shown(controls)
    controls.set_attention(None)
    assert "attention" not in _shown(controls)


def test_run_says_why_it_cannot(controls):
    controls.set_run_enabled(False, "Select a lamella to run the workflow")
    assert controls.run_btn.toolTip() == "Select a lamella to run the workflow"
    controls.set_run_enabled(True, "Run workflow: 1 lamella, 2 tasks")
    assert controls.run_btn.isEnabled()


def test_each_button_says_it_was_pressed(controls):
    pressed = []
    for name in ("run", "stop", "supervision", "attention"):
        getattr(controls, f"{name}_clicked").connect(
            lambda name=name: pressed.append(name)
        )
    controls.set_run_enabled(True, "")
    controls.set_running(False)
    controls.run_btn.click()
    controls.set_running(True)
    controls.stop_btn.click()
    controls.set_supervision(SUPERVISED, "Polishing")
    controls.supervision_btn.click()
    controls.set_attention("Attention Required")
    controls.attention_btn.click()
    assert pressed == ["run", "stop", "supervision", "attention"]

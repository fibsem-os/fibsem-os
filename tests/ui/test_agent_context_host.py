"""AgentContext against the real window.

The unit tests (tests/autolamella/test_agent_context.py) drive the facade over
real domain objects in a plain holder; this file proves the production host —
an actual AutoLamellaUI — satisfies the same contract, in the states the facade
will actually meet it: freshly constructed (nothing connected, nothing loaded)
and with an experiment adopted."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict

from fibsem.applications.autolamella.server import AgentContext
from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.AutoLamellaUI import AutoLamellaUI
from fibsem.structures import MicroscopeState


@pytest.fixture
def ui(qapp):
    window = AutoLamellaUI(parent_ui=None)
    try:
        yield window
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


def test_a_fresh_window_satisfies_the_host_contract(ui):
    ctx = AgentContext(ui)
    status = ctx.status()
    json.dumps(status)
    assert status["microscope_connected"] is False
    assert status["experiment"] is None
    assert status["workflow"]["running"] is False
    assert ctx.queue()["available"] is False
    assert ctx.run_summary()["available"] is False
    # No image widget until a microscope connects: unavailable, not an error.
    assert ctx.display_images() == {"available": False, "sem": None, "fib": None}


def test_display_images_mirror_the_widgets_current_images(ui):
    import numpy as np

    from fibsem.structures import FibsemImage

    class _Widget:
        eb_image = FibsemImage.generate_blank_image(resolution=[128, 128], hfw=1e-6)
        ib_image = None

    _Widget.eb_image.data[:] = np.random.randint(
        0, 255, _Widget.eb_image.data.shape, dtype=np.uint8
    )
    ui.image_widget = _Widget()
    ctx = AgentContext(ui)

    payload = ctx.display_images()
    json.dumps(payload)
    assert payload["available"] is True
    assert payload["fib"] is None  # nothing displayed on that side yet
    sem = payload["sem"]
    assert sem["width"] > 0 and sem["height"] > 0
    assert sem["image_b64_jpeg"]
    # Blank placeholders carry no acquisition date; the key degrades to None
    # rather than the whole payload failing.
    assert "acquired_at" in sem


def test_an_adopted_experiment_is_visible_through_the_facade(ui, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="host-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())

    ctx = AgentContext(ui)  # built BEFORE the experiment exists — call-time wins
    ui.experiment = exp

    status = ctx.status()
    assert status["experiment"]["name"] == "host-exp"
    assert status["experiment"]["num_items"] == 1
    assert ctx.task_outputs(exp.positions[0].name)["available"] is True

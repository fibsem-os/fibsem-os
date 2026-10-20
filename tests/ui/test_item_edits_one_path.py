"""A lamella's own fields are edited on one path: the lamella editor, the
verdict menu and the App Server's item routes all reach
``config_edits.edit_item_fields`` (a correlation through
``config_edits.apply_correlation`` first), with its version check and its
alignment-area revert.

The real main window on Demo. Each GUI edit is driven through the handler its
widget calls, each route through the context method the route calls.
"""

import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")

from psygnal.containers import EventedDict  # noqa: E402
from PyQt5.QtWidgets import QMenu  # noqa: E402

from fibsem.applications.autolamella import config_edits  # noqa: E402
from fibsem.applications.autolamella.server import AutoLamellaAppContext  # noqa: E402
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
)
from fibsem.applications.autolamella.ui.common import (  # noqa: E402
    lamella_display,
)
from fibsem.applications.autolamella.ui.protocol import (  # noqa: E402
    lamella_editor as lamella_editor_module,
)
from fibsem.correlation.history import new_run_dir, save_run  # noqa: E402
from fibsem.correlation.structures import (  # noqa: E402
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
)
from fibsem.structures import FibsemRectangle, MicroscopeState, Point  # noqa: E402
from fibsem.ui.widgets.canvas.canvas_state import AlignmentSpec  # noqa: E402
from fibsem.workflows.quality import Verdict  # noqa: E402

# Every module that holds the shared edit by name.
_HOLDERS = (config_edits, lamella_editor_module, lamella_display)


@pytest.fixture(scope="module")
def window(qapp):
    from fibsem.applications.autolamella.ui import window as module

    win = module.AutoLamellaWindow()
    win.system_widget.connect_to_microscope()
    yield win
    if win.microscope is not None:
        win.microscope.disconnect()
    original_quit = qapp.quit
    qapp.quit = lambda: None
    try:
        win.close()
    finally:
        qapp.quit = original_quit


@pytest.fixture
def experiment(window, tmp_path):
    exp = Experiment(path=tmp_path / "exp", name="one-path")
    exp.task_protocol = AutoLamellaTaskProtocol()
    (tmp_path / "exp").mkdir(parents=True, exist_ok=True)
    exp.add_new_lamella(MicroscopeState(), EventedDict())
    window.experiment = exp
    window.lamella_tab.editor.set_experiment()
    window.lamella_tab.editor.select_lamella(exp.positions[0].name)
    yield exp
    window.lamella_tab.editor.flush_pending_save()


@pytest.fixture
def editor(window, experiment):
    return window.lamella_tab.editor


@pytest.fixture
def context(window):
    return AutoLamellaAppContext(window.session)


@pytest.fixture
def calls(monkeypatch):
    """The patches ``edit_item_fields`` is called with, whoever calls it."""
    edit = config_edits.edit_item_fields
    assert all(module.edit_item_fields is edit for module in _HOLDERS)
    seen = []

    def spy(lamella, patch, version):
        seen.append(dict(patch))
        return edit(lamella, patch, version)

    for module in _HOLDERS:
        monkeypatch.setattr(module, "edit_item_fields", spy)
    return seen


def _route(context, lamella, patch):
    result = context.apply_item_patch(
        lamella.name, patch, config_edits.item_fields_version(lamella)
    )
    assert result["applied"], result
    return result


def test_the_point_of_interest(editor, context, experiment, calls):
    lamella = experiment.positions[0]

    editor._on_point_of_interest_updated(Point(x=2e-6, y=-1e-6))
    assert (lamella.poi.x, lamella.poi.y) == (2e-6, -1e-6)
    _route(context, lamella, config_edits.poi_patch(Point(x=3e-6, y=1e-6)))
    assert (lamella.poi.x, lamella.poi.y) == (3e-6, 1e-6)

    assert [set(patch) for patch in calls] == [{"poi.x", "poi.y"}] * 2


def test_the_alignment_area(editor, context, experiment, calls, monkeypatch):
    monkeypatch.setattr(editor, "alignment_area_editable", True, raising=False)
    lamella = experiment.positions[0]
    sides = {f"alignment_area.{s}" for s in ("left", "top", "width", "height")}

    editor._on_alignment_area_updated(FibsemRectangle(0.1, 0.2, 0.3, 0.4))
    assert lamella.alignment_area == FibsemRectangle(0.1, 0.2, 0.3, 0.4)
    _route(context, lamella, {"alignment_area.left": 0.2})
    assert lamella.alignment_area.left == 0.2

    assert set(calls[0]) == sides
    assert calls[1] == {"alignment_area.left": 0.2}


def test_the_description(editor, context, experiment, calls):
    lamella = experiment.positions[0]

    editor.line_edit_description.setText("thin at the top")
    editor._on_description_edited()
    assert lamella.description == "thin at the top"
    _route(context, lamella, {"description": "cracked"})
    assert lamella.description == "cracked"

    assert calls == [{"description": "thin at the top"}, {"description": "cracked"}]


def test_the_verdict(context, experiment, calls):
    lamella = experiment.positions[0]
    changed = []
    menu = QMenu()
    sub = lamella_display.add_defect_menu(menu, lamella, lambda: changed.append(1))
    redrawn = []
    lamella.events.defect.connect(lambda *_: redrawn.append(1))

    good = next(a for a in sub.actions() if a.text() == "Good")
    good.trigger()
    assert lamella.defect.verdict is Verdict.GOOD
    assert lamella.defect.updated_at is not None
    _route(context, lamella, {"defect.verdict": "UNASSESSED"})
    assert lamella.defect.verdict is Verdict.UNASSESSED
    assert lamella.defect.updated_at is None  # taken back, as the menu takes it back

    assert calls == [{"defect.verdict": "GOOD"}, {"defect.verdict": "UNASSESSED"}]
    assert changed == [1] and len(redrawn) == 2  # each a new record the displays see


def test_the_correlation_result(editor, context, experiment, calls, monkeypatch):
    lamella = experiment.positions[0]
    correlated = []
    apply = config_edits.apply_correlation
    assert lamella_editor_module.apply_correlation is apply

    def spy(result, *args):
        correlated.append(result.poi[0].px_m)
        return apply(result, *args)

    from fibsem.applications.autolamella.server import context as context_module

    for module in (lamella_editor_module, context_module):
        monkeypatch.setattr(module, "apply_correlation", spy)

    def result(point):
        return CorrelationResult(poi=[CorrelationPointOfInterest(px_m=point)])

    editor._handle_correlation_dialog_result(result(Point(2e-6, -3e-6)))
    assert (lamella.poi.x, lamella.poi.y) == (2e-6, -3e-6)

    run_path = new_run_dir(str(lamella.path))
    save_run(run_path, CorrelationState(result=result(Point(1e-6, 1e-6))))
    applied = context.apply_item_correlation(lamella.name, run_path)
    assert applied["applied"], applied
    assert (lamella.poi.x, lamella.poi.y) == (1e-6, 1e-6)

    assert correlated == [Point(2e-6, -3e-6), Point(1e-6, 1e-6)]
    assert [set(patch) for patch in calls] == [{"poi.x", "poi.y"}] * 2


def test_an_invalid_alignment_area_is_refused_and_put_back(
    editor, experiment, monkeypatch
):
    """The route's refusal, in the editor: nothing set, nothing saved, and the
    canvas shows the area the lamella still has."""
    monkeypatch.setattr(editor, "alignment_area_editable", True, raising=False)
    monkeypatch.setattr(editor, "show_alignment_area", True, raising=False)
    lamella = experiment.positions[0]
    had = FibsemRectangle(0.6, 0.3, 0.3, 0.4)
    lamella.alignment_area = deepcopy(had)
    drawn, toasts, saves = [], [], []
    monkeypatch.setattr(editor, "_save_experiment", lambda: saves.append(1))
    set_overlay = editor.view_controller.set_overlay

    def overlay(beam, spec):
        if isinstance(spec, AlignmentSpec):
            drawn.append(deepcopy(spec.rect))
        set_overlay(beam, spec)

    monkeypatch.setattr(editor.view_controller, "set_overlay", overlay)
    monkeypatch.setattr(
        lamella_editor_module.notification_service,
        "show_toast",
        lambda message, kind="info": toasts.append(kind),
    )

    editor._on_alignment_area_updated(FibsemRectangle(0.9, 0.3, 0.3, 0.4))

    assert lamella.alignment_area == had
    assert drawn[-1] == had
    assert toasts == ["warning"]
    assert saves == []

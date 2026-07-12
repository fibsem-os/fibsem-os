"""Offscreen tests for LamellaSelectionDialog (place positions on the overview).

Exercises the working-set + commit logic with a real demo-acquired image (so
the stage<->pixel reproject has valid metadata) and a fake host that stands in
for AutoLamellaUI's creation plumbing.
"""

import os
from copy import deepcopy

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("matplotlib")

from types import SimpleNamespace  # noqa: E402

from PyQt5.QtWidgets import QApplication  # noqa: E402

from fibsem import utils  # noqa: E402
from fibsem.structures import (  # noqa: E402
    BeamType,
    FibsemStagePosition,
    ImageSettings,
)
from fibsem.applications.autolamella.structures import (  # noqa: E402
    AutoLamellaTaskProtocol,
    Experiment,
    GridRecord,
    Lamella,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def scene(tmp_path):
    microscope, _ = utils.setup_session(manufacturer="Demo")
    image = microscope.acquire_image(
        ImageSettings(resolution=(1536, 1024), hfw=400e-6, beam_type=BeamType.ELECTRON)
    )
    assert image.metadata is not None  # the overview must carry acquisition state
    exp = Experiment.create(path=str(tmp_path), name="exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    rec = GridRecord(name="grid-aspen")
    exp.add_grid(rec)
    return microscope, image, exp, rec


class _FakeHost:
    """Stand-in for AutoLamellaUI: records add_new_lamella + actually adds it."""

    def __init__(self, experiment):
        self.experiment = experiment
        self.added = []
        self.refreshed = 0
        self.lamella_added_signal = SimpleNamespace(emit=lambda: None)

    def add_new_lamella(self, stage_position, name, microscope_state, grid_id, notify):
        self.added.append({"name": name, "grid_id": grid_id, "notify": notify,
                           "state": microscope_state})
        state = deepcopy(microscope_state)
        state.stage_position = deepcopy(stage_position)
        self.experiment.add_new_lamella(
            state, self.experiment.task_protocol.task_config, name=name, grid_id=grid_id)

    def update_lamella_combobox(self):
        self.refreshed += 1

    def update_ui(self):
        self.refreshed += 1


def _dialog(scene, host):
    from fibsem.ui.widgets.lamella_selection_dialog import LamellaSelectionDialog

    microscope, image, exp, rec = scene
    return LamellaSelectionDialog(exp, rec, image, microscope, host=host)


def _existing_lamella(microscope, exp, rec, name="L1"):
    lam = Lamella(petname=name, path=str(exp.path) + f"/{name}", number=1, grid_id=rec._id)
    lam.milling_pose = microscope.get_microscope_state()  # real lamellae always have one
    exp.add_lamella(lam)


def test_seeds_from_existing_lamellae(qapp, scene):
    microscope, image, exp, rec = scene
    _existing_lamella(microscope, exp, rec)
    host = _FakeHost(exp)
    dlg = _dialog(scene, host)
    assert [e.name for e in dlg._entries] == ["L1"]
    assert dlg._entries[0].is_new is False  # it's an existing lamella


def test_add_then_accept_creates_lamella(qapp, scene):
    microscope, image, exp, rec = scene
    host = _FakeHost(exp)
    dlg = _dialog(scene, host)
    assert dlg._entries == []  # grid has no lamellae yet

    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    assert len(dlg._entries) == 1 and dlg._entries[0].is_new

    dlg._on_accept()
    # committed through the host with the overview state + the grid id, deferred refresh
    assert len(host.added) == 1
    assert host.added[0]["grid_id"] == rec._id
    assert host.added[0]["notify"] is False
    assert host.added[0]["state"] is image.metadata.microscope_state
    assert exp.get_lamellae_for_grid(rec)[0].grid_id == rec._id


def test_draft_names_use_petname_scheme(qapp, scene):
    import re

    microscope, image, exp, rec = scene
    host = _FakeHost(exp)
    dlg = _dialog(scene, host)
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    dlg._on_add(FibsemStagePosition(x=1e-5, y=0, z=0, r=0, t=0))

    n1, n2 = dlg._entries[0].name, dlg._entries[1].name
    # same scheme as Experiment.generate_lamella_name (number-prefixed), sequential
    assert n1 != n2
    lead = lambda s: int(re.search(r"\d+", s).group())
    assert lead(n2) == lead(n1) + 1


def test_move_existing_updates_position_on_accept(qapp, scene):
    microscope, image, exp, rec = scene
    _existing_lamella(microscope, exp, rec)
    stored = exp.get_lamellae_for_grid(rec)[0]
    host = _FakeHost(exp)
    dlg = _dialog(scene, host)

    dlg._set_selected(0)
    new_pos = FibsemStagePosition(x=5e-5, y=-5e-5, z=0, r=0, t=0)
    dlg._on_move(new_pos)
    dlg._on_accept()

    # no new lamella created; the existing one's milling_pose moved
    assert host.added == []
    assert stored.milling_pose.stage_position.x == pytest.approx(5e-5)


def test_unmoved_existing_lamella_untouched_on_accept(qapp, scene):
    microscope, image, exp, rec = scene
    _existing_lamella(microscope, exp, rec)
    stored = exp.get_lamellae_for_grid(rec)[0]
    original_pose = stored.milling_pose  # identity check
    host = _FakeHost(exp)
    dlg = _dialog(scene, host)

    dlg._on_accept()  # accept without moving anything

    assert stored.milling_pose is original_pose  # not rewritten with the overview state
    assert host.added == []


def test_markers_drawn_for_positions(qapp, scene):
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    dlg._on_add(FibsemStagePosition(x=2e-5, y=0, z=0, r=0, t=0))

    specs = dlg.canvas._markers._specs
    assert len([s for s in specs if s.kind == "rect"]) == 2       # a FOV box per position
    assert len([s for s in specs if s.kind == "crosshair"]) == 2  # + a crosshair marker each


def test_fov_toggle_removes_rectangles(qapp, scene):
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))

    dlg.canvas._set_show_fov(False)
    specs = dlg.canvas._markers._specs
    assert [s for s in specs if s.kind == "rect"] == []
    assert len([s for s in specs if s.kind == "crosshair"]) == 1  # crosshair marker stays


def test_names_toggle_clears_labels(qapp, scene):
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    assert any(s.label for s in dlg.canvas._markers._specs)  # named by default

    dlg.canvas._set_show_names(False)
    assert all(s.label == "" for s in dlg.canvas._markers._specs)


def test_marker_update_preserves_view(qapp, scene):
    # adding/selecting a marker only re-draws the overlay, never re-fits the image
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    ax = dlg.canvas.canvas._ax
    ax.set_xlim(100, 300)
    ax.set_ylim(300, 100)

    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    assert ax.get_xlim() == pytest.approx((100, 300))
    assert ax.get_ylim() == pytest.approx((300, 100))


def test_uses_shared_image_canvas_with_downsample_cap(qapp, scene):
    from fibsem.ui.widgets.canvas.image_canvas import FibsemImageCanvas
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    assert isinstance(dlg.canvas.canvas, FibsemImageCanvas)
    # the display buffer is the (capped) downsample, never larger than the cap
    base = dlg.canvas.canvas._display_base
    assert base is not None and max(base.shape[:2]) <= 2048


def test_click_selects_nearest_then_clears(qapp, scene):
    from fibsem.imaging.tiled import reproject_stage_positions_onto_image2
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    pt = reproject_stage_positions_onto_image2(image, [dlg._entries[0].stage_position])[0]

    got = []
    dlg.canvas.position_selected.connect(lambda i: got.append(i))
    dlg.canvas._on_click(pt.x, pt.y, None)          # on the marker -> selects index 0
    assert got[-1] == 0
    dlg.canvas._on_click(pt.x + 1e6, pt.y, None)    # far away -> clears
    assert got[-1] is None


def test_delete_draft_removes_without_committing(qapp, scene):
    microscope, image, exp, rec = scene
    dlg = _dialog(scene, _FakeHost(exp))
    dlg._on_add(FibsemStagePosition(x=0, y=0, z=0, r=0, t=0))
    assert len(dlg._entries) == 1

    dlg._on_list_remove(SimpleNamespace(name=dlg._entries[0].name))  # post-confirm signal
    assert dlg._entries == []
    assert dlg._to_delete == []  # a draft was never committed

    dlg._on_accept()
    assert exp.get_lamellae_for_grid(rec) == []  # nothing added


def test_delete_existing_applied_on_accept(qapp, scene):
    microscope, image, exp, rec = scene
    _existing_lamella(microscope, exp, rec)
    stored = exp.get_lamellae_for_grid(rec)[0]
    dlg = _dialog(scene, _FakeHost(exp))

    dlg._on_list_remove(SimpleNamespace(name=stored.name))
    assert stored in exp.positions      # deferred — still there until Accept
    assert stored in dlg._to_delete

    dlg._on_accept()
    assert stored not in exp.positions  # removed on commit


def test_delete_existing_discarded_on_cancel(qapp, scene):
    microscope, image, exp, rec = scene
    _existing_lamella(microscope, exp, rec)
    stored = exp.get_lamellae_for_grid(rec)[0]
    dlg = _dialog(scene, _FakeHost(exp))

    dlg._on_list_remove(SimpleNamespace(name=stored.name))
    dlg.reject()  # Cancel discards the queued deletion
    assert stored in exp.positions


def test_no_microscope_is_read_only(qapp, scene):
    microscope, image, exp, rec = scene
    host = _FakeHost(exp)
    from fibsem.ui.widgets.lamella_selection_dialog import LamellaSelectionDialog
    dlg = LamellaSelectionDialog(exp, rec, image, microscope=None, host=host)
    # right-click add is a no-op without a microscope to project the click
    dlg.canvas._on_right_click(10, 10, None)
    assert dlg._entries == []

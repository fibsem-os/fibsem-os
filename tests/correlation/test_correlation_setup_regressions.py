"""Regressions introduced by the Setup-tab move (FIB-319).

Setup moved from implicit seeding in the protocol editor, through a pre-dialog,
into a section inside the correlation window. Things lost or invented on the way:

* ``load_spot_burns`` stopped gating anything — an experiment that turned burn
  seeding off got it anyway;
* the manual-edit guard couldn't see surface points, so a re-seed wiped one
  without asking, silently disarming the RI pre-correction with it;
* the same guard announced "edited" after interpolation, which only resampled z;
* a declined re-seed left the radio claiming a source that was never applied;
* spot burns read as applied with no FIB image to place them in — and the editor
  handed over a *blank generated placeholder* when the lamella had no reference
  image, which would scale the burns and the POI by placeholder dimensions.

Headless PyQt5, offscreen.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import types

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.config import CorrelationConfig
from fibsem.correlation.history import CorrelationRun, LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage, Point
from fibsem.ui.correlation.widgets.correlation_setup_section import (
    SEED_NONE,
    SEED_PREVIOUS,
    SEED_SPOT_BURNS,
    CorrelationSetupSection,
)


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


OLD = "2026-07-20_09-00-00"
NEW = "2026-07-24_14-32-00"


def _widget():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def _fib_image():
    return FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)


def _run(name, inp=None):
    return CorrelationRun(
        path="/tmp/" + name,
        name=name,
        state=CorrelationState(input_data=inp or CorrelationInputData()),
    )


def _asked(monkeypatch, answer):
    """Record every confirm/inform the widget raises, answering with `answer`."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    seen = []
    for name in ("question", "information"):
        monkeypatch.setattr(
            ctw.QMessageBox,
            name,
            staticmethod(lambda *a, **k: (seen.append(a[1]), answer)[1]),
        )
    return seen


# ---------------------------------------------------------------------------
# 1 — load_spot_burns gates the default again
# ---------------------------------------------------------------------------


def test_load_spot_burns_false_does_not_preselect_burns(qapp):
    """The flag meant "don't seed burns automatically". It kept round-tripping
    through the protocol while nothing read it."""
    s = CorrelationSetupSection(
        spot_burns=[Point(0.1, 0.2)],
        history=LamellaCorrelation(runs=[]),
        config=CorrelationConfig(load_spot_burns=False),
    )
    assert s.seed_source() == SEED_NONE
    # still offered, though: an explicit click is not automatic seeding
    assert s.rb_burns.isEnabled()


# ---------------------------------------------------------------------------
# 2 — the manual-edit guard sees surface points, and isn't fooled by a rescale
# ---------------------------------------------------------------------------


def _seed_burns(w, **kwargs):
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig(), **kwargs
    )
    section.emit_current_seed()
    return section


def test_a_surface_point_is_something_to_lose(qapp, monkeypatch):
    """Surface points live in their own single-value fields, not the three
    coordinate lists — so the guard walked straight past them."""
    from PyQt5.QtWidgets import QMessageBox

    w = _widget()
    w.set_fib_image(_fib_image())
    section = _seed_burns(w)

    w._coords_tab.fm_surface_list.add_coordinate(
        Coordinate(point=PointXYZ(x=10.0, y=20.0, z=3.0), point_type=PointType.SURFACE_FM)
    )
    assert w._has_manual_edits()

    asked = _asked(monkeypatch, QMessageBox.Cancel)
    section.rb_none.setChecked(True)  # would wipe the surface point

    assert asked, "no confirmation before discarding a surface point"
    assert w.data.fm_surface_coordinate is not None


def test_interpolating_does_not_look_like_an_edit(qapp, monkeypatch):
    """Interpolation rescales every FM-side z. The baseline has to move with it,
    or the next re-seed reports edits the user never made."""
    from PyQt5.QtWidgets import QMessageBox

    from tests.correlation.test_correlation_setup_section import _fm_image

    w = _widget()
    w.set_fib_image(_fib_image())
    w.set_fm_image(_fm_image(nz=11))
    seed = CorrelationInputData(
        fm_coordinates=[
            Coordinate(point=PointXYZ(x=5.0, y=6.0, z=4.0), point_type=PointType.FM)
        ]
    )
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)],
        history=LamellaCorrelation(runs=[_run(NEW, seed)]),
        config=CorrelationConfig(),
    )
    section.emit_current_seed()
    assert not w._has_manual_edits()

    w._adopt_interpolated_volume(_fm_image(nz=22, pixel_size_z=100e-9), old_nz=11)
    assert w.data.fm_coordinates[0].point.z == pytest.approx(8.0)  # z did move
    assert not w._has_manual_edits()  # ...but the user didn't move it

    asked = _asked(monkeypatch, QMessageBox.Ok)
    section.rb_burns.setChecked(True)
    assert not asked, "warned about edits after nothing was edited"


# ---------------------------------------------------------------------------
# 3 — a declined re-seed puts the controls back
# ---------------------------------------------------------------------------


def test_declining_restores_the_radio(qapp, monkeypatch):
    """Qt flips the radio before emitting `toggled`, so an unapplied source stays
    on screen — and re-clicking the same radio emits nothing, so the obvious
    recovery does nothing either."""
    from PyQt5.QtWidgets import QMessageBox

    w = _widget()
    w.set_fib_image(_fib_image())
    section = _seed_burns(w)
    w.data.fib_coordinates[0].point.x = 111.0  # hand-moved

    _asked(monkeypatch, QMessageBox.Cancel)
    section.rb_none.setChecked(True)

    assert section.seed_source() == SEED_SPOT_BURNS
    assert section.rb_burns.isChecked()
    assert w.data.fib_coordinates[0].point.x == 111.0


def test_declining_restores_the_run_selection(qapp, monkeypatch):
    """Same for the run dropdown, which re-seeds on every index change."""
    from PyQt5.QtWidgets import QMessageBox

    w = _widget()
    w.set_fib_image(_fib_image())
    inp = CorrelationInputData(
        fib_coordinates=[Coordinate(point=PointXYZ(x=1.0), point_type=PointType.FIB)]
    )
    section = w.add_lamella_setup(
        history=LamellaCorrelation(runs=[_run(OLD, inp), _run(NEW, inp)]),
        config=CorrelationConfig(),
    )
    section.emit_current_seed()
    assert section.seed_source() == SEED_PREVIOUS
    w.data.fib_coordinates[0].point.x = 222.0

    _asked(monkeypatch, QMessageBox.Cancel)
    section.run_combo.setCurrentIndex(1)

    assert section.run_combo.currentIndex() == 0
    assert w.data.fib_coordinates[0].point.x == 222.0


# ---------------------------------------------------------------------------
# 5 — spot burns need a FIB image, and a real one
# ---------------------------------------------------------------------------


def test_burns_are_not_offered_without_a_fib_image(qapp):
    """Nothing to multiply the normalised burns by — so don't present the choice,
    and above all don't make it the default the window opens on."""
    w = _widget()  # no FIB image
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig()
    )
    assert not section.rb_burns.isEnabled()
    assert "needs the FIB image" in section.rb_burns.text()
    assert section.seed_source() == SEED_NONE


def test_burns_become_available_when_the_image_arrives(qapp):
    """Opening with no image and browsing to one is a supported route."""
    w = _widget()
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)], config=CorrelationConfig()
    )
    w.set_fib_image(_fib_image())

    assert section.rb_burns.isEnabled()
    assert "needs the FIB image" not in section.rb_burns.text()


def test_spot_burns_without_a_fib_image_are_refused(qapp, monkeypatch):
    """Belt and braces behind the disabled radio: the seeding call returns early
    with no image, and used to leave the radio reading as applied — no points, no
    message, no way to tell."""
    w = _widget()  # no FIB image
    section = w.add_lamella_setup(
        spot_burns=[Point(0.25, 0.5)],
        history=LamellaCorrelation(runs=[_run(NEW)]),
        config=CorrelationConfig(),
    )
    section.emit_current_seed()  # defaults to the previous run

    asked = _asked(monkeypatch, None)
    section.rb_burns.setChecked(True)

    assert asked == ["No FIB image"]
    assert section.seed_source() == SEED_PREVIOUS  # not left claiming burns


def _editor_stub(lamella_path, fib_filename, image):
    """Enough of the protocol editor to run _open_correlation_dialog unbound —
    constructing the real widget needs napari.

    A real QWidget, not a namespace: it is passed as the dialog's parent, so the
    end-to-end variant of this harness needs it to survive PyQt's type check.
    """
    from PyQt5.QtWidgets import QComboBox, QWidget

    from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget as Editor,
    )

    fib_combo = QComboBox()
    if fib_filename:
        fib_combo.addItem(fib_filename)
    stub = QWidget()
    stub._selected_lamella = types.SimpleNamespace(path=lamella_path, task_config={})
    stub.parent_widget = None
    stub.image = image
    stub.fm_image = None
    stub.combobox_fib_filenames = fib_combo
    stub.combobox_fm_filenames = QComboBox()
    stub._image_path = Editor._image_path
    stub._image_paths = Editor._image_paths
    stub._spot_burn_coordinates = Editor._spot_burn_coordinates
    return stub


def _fake_dialog_class(record):
    class _FakeDialog:
        def __init__(self, parent=None):
            record.append(self)
            self.fib_image = None

        def set_project_dir(self, path):
            pass

        def set_fib_image(self, image):
            self.fib_image = image

        def set_fm_image(self, image):
            pass

        def add_lamella_setup(self, **kwargs):
            return types.SimpleNamespace(emit_current_seed=lambda: None)

        def exec_(self):
            return 0  # rejected — stop before the result handling

    return _FakeDialog


def _open_correlation(stub, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw
    from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget as Editor,
    )

    opened = []
    monkeypatch.setattr(ctw, "CorrelationTabDialog", _fake_dialog_class(opened))
    Editor._open_correlation_dialog(stub)
    return opened[0]


def test_editor_withholds_the_placeholder_image(qapp, tmp_path, monkeypatch):
    """With no *_ib.tif the editor's self.image is a blank generated placeholder
    for its own canvas. Correlating against it would scale the burns — and the
    POI written back to the experiment — by placeholder dimensions."""
    stub = _editor_stub(str(tmp_path), fib_filename="", image=_fib_image())
    assert _open_correlation(stub, monkeypatch).fib_image is None


def test_editor_passes_a_real_reference_image(qapp, tmp_path, monkeypatch):
    real = tmp_path / "01_ref_ib.tif"
    _fib_image().save(str(real))
    image = _fib_image()
    stub = _editor_stub(str(tmp_path), fib_filename=real.name, image=image)
    assert _open_correlation(stub, monkeypatch).fib_image is image

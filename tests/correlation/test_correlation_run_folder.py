"""A run folder is created when a correlation is recorded, not when one is opened
(FIB-320).

``Correlation/<timestamp>/`` used to be minted before the window appeared, and
the initial seed auto-saved into it immediately — so cancelling still left a
complete run on disk. That was harmless while nothing read those folders;
FIB-299/301/302 made them *functional* (they seed the next correlation), which
turned it into two real defects:

* an abandoned session became the next open's starting coordinates — Cancel read
  as "discard", but wasn't;
* history counted opens rather than correlations, so a first correlation stopped
  being treated as one.

Headless PyQt5, offscreen.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import json

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.history import LamellaCorrelation
from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage

CORRELATION_JSON = "correlation.json"


def _widget():
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    return CorrelationTabWidget()


def _inputs(x=1.0):
    return CorrelationInputData(
        fib_coordinates=[
            Coordinate(point=PointXYZ(x=x, y=2.0, z=0.0), point_type=PointType.FIB)
        ]
    )


def _result(inp=None):
    return CorrelationResult(
        poi=[CorrelationPointOfInterest()], rms_error=1.0, input_data=inp or _inputs()
    )


def _saved(run_dir):
    return json.loads((run_dir / CORRELATION_JSON).read_text())


def _saved_fib_x(run_dir):
    return _saved(run_dir)["input_data"]["fib_coordinates"][0]["point"]["x"]


# ---------------------------------------------------------------------------
# the widget: nothing on disk until there is something worth writing
# ---------------------------------------------------------------------------


def test_picking_points_alone_writes_nothing(qapp, tmp_path):
    """Seeding and editing coordinates is not a correlation. Writing here is what
    made every open a run."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))

    w.set_data(_inputs(x=42.0))
    w.data_changed.emit(w.data)

    assert not run.exists()


def test_an_existing_project_dir_saves_from_the_first_edit(qapp, tmp_path):
    """The standalone widget points at a directory the user picked, which already
    exists — deferring *creation* must not become deferring *saving* there, or
    quickstart silently stops persisting point-picking."""
    run = tmp_path / "chosen_by_the_user"
    run.mkdir()
    w = _widget()
    w.set_project_dir(str(run))

    w.set_data(_inputs(x=42.0))
    w.data_changed.emit(w.data)

    assert _saved_fib_x(run) == 42.0


def test_the_first_result_creates_the_run_folder(qapp, tmp_path):
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w.set_data(_inputs(x=42.0))

    w._on_result_ready(_result(w.data))

    assert (run / CORRELATION_JSON).is_file()
    assert _saved(run)["result"] is not None


def test_edits_after_a_result_keep_the_file_current(qapp, tmp_path):
    """Once the folder exists it tracks the session — including the FIB-295 case
    where an edit leaves the stored result stale but still recorded."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w.set_data(_inputs(x=42.0))
    w._on_result_ready(_result(w.data))

    w.data.fib_coordinates[0].point.x = 999.0
    w.data_changed.emit(w.data)

    assert _saved_fib_x(run) == 999.0
    assert _saved(run)["result"] is not None  # stale, not dropped


def test_discarding_the_result_does_not_stop_saving(qapp, tmp_path):
    """The folder is already a run; it must not go quietly out of date."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w._on_result_ready(_result())
    w._discard_result()

    w.set_data(_inputs(x=7.0))
    w.data_changed.emit(w.data)

    assert _saved(run)["result"] is None
    assert _saved_fib_x(run) == 7.0


def test_explicit_save_creates_the_folder_it_writes_into(qapp, tmp_path):
    """File > Save Correlation seeds its dialog with <run folder>/correlation.json.
    Deferring creation made that default a path whose directory doesn't exist, so
    accepting it raised FileNotFoundError out of open()."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w.set_data(_inputs(x=3.0))

    w.save_correlation(str(run / CORRELATION_JSON))  # the offered default

    assert _saved_fib_x(run) == 3.0


def test_save_plot_creates_the_folder_it_writes_into(qapp, tmp_path):
    """save_plot targets the run folder directly, so it can be the first writer."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(64, 64), hfw=100e-6))

    w.save_plot()

    assert run.is_dir()
    assert list(run.glob("correlation_plot_*.png"))


def test_save_plot_creates_the_folder_for_an_explicit_path_too(qapp, tmp_path):
    """The Save Plot dialog offers a default *inside* the run folder and then
    passes it back as an explicit path — the same absent directory, arriving by
    the branch that doesn't derive it."""
    run = tmp_path / "2026-07-27_09-00-00"
    w = _widget()
    w.set_project_dir(str(run))
    w.set_fib_image(FibsemImage.generate_blank_image(resolution=(64, 64), hfw=100e-6))

    w.save_plot(str(run / "correlation_plot_2026-07-27_09-05-00.png"))  # as offered

    assert (run / "correlation_plot_2026-07-27_09-05-00.png").is_file()


# ---------------------------------------------------------------------------
# end to end: the editor opens the real window and the user cancels
# ---------------------------------------------------------------------------


def test_a_run_without_a_result_says_so_in_the_dropdown(qapp):
    """A run folder is not proof a correlation ran: File > Save Correlation
    writes one deliberately (creating the folder is the point of an explicit
    save), and a run whose result was discarded keeps its folder. Picking a
    starting point means picking between those and real correlations."""
    from fibsem.correlation.history import CorrelationRun
    from fibsem.correlation.structures import CorrelationState
    from fibsem.ui.correlation.widgets.correlation_setup_section import (
        format_run_label,
    )

    def _run(result):
        return CorrelationRun(
            path="/tmp/r",
            name="2026-07-24_14-32-00",
            state=CorrelationState(input_data=CorrelationInputData(), result=result),
        )

    assert format_run_label(_run(_result())) == "2026-07-24 14:32:00"
    assert format_run_label(_run(None)) == "2026-07-24 14:32:00  ·  no result"


def test_a_failed_save_is_reported(qapp, tmp_path, monkeypatch):
    """The load handler warns on failure; the save handlers didn't, so the
    exception went to the event loop and the user was told nothing."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    warned = []
    monkeypatch.setattr(
        ctw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "x.json"), "")),
    )
    monkeypatch.setattr(
        ctw.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[1]))
    )

    w = _widget()
    monkeypatch.setattr(
        type(w), "save_correlation", lambda self, path: (_ for _ in ()).throw(OSError("disk full"))
    )
    w._on_save()  # must not raise

    assert warned == ["Save Error"]


def test_a_failed_plot_save_is_reported(qapp, tmp_path, monkeypatch):
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw

    warned = []
    monkeypatch.setattr(
        ctw.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(tmp_path / "x.png"), "")),
    )
    monkeypatch.setattr(
        ctw.QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[1]))
    )

    w = _widget()
    monkeypatch.setattr(
        type(w), "save_plot", lambda self, path=None: (_ for _ in ()).throw(OSError("disk full"))
    )
    w._on_save_plot_clicked()  # must not raise

    assert warned == ["Save Error"]


def test_a_cancelled_correlation_leaves_no_run_behind(qapp, tmp_path, monkeypatch):
    """The reproduction from the issue: open, edit a point, cancel — and the next
    open must not offer that abandoned session as a previous correlation."""
    import fibsem.ui.correlation.widgets.correlation_tab_widget as ctw
    from fibsem.applications.autolamella.ui.autolamella_lamella_protocol_editor import (
        AutoLamellaProtocolEditorWidget as Editor,
    )
    from tests.correlation.test_correlation_setup_regressions import _editor_stub

    opened = []

    def _cancel(self):
        opened.append(self)
        self.widget.set_data(_inputs(x=999.0))  # the user picks, then thinks better
        self.widget.data_changed.emit(self.widget.data)
        return 0  # QDialog.Rejected

    monkeypatch.setattr(ctw.CorrelationTabDialog, "exec_", _cancel)

    lamella = tmp_path / "lamella-01"
    lamella.mkdir()
    Editor._open_correlation_dialog(_editor_stub(str(lamella), "", image=None))

    assert opened, "the dialog never opened — the test proves nothing"
    assert LamellaCorrelation.discover(str(lamella / "Correlation")).runs == []

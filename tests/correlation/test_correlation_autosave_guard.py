"""Opening a run folder must not write to it (FIB-977).

The auto-save used to fire on the first ``data_changed`` -- which an image
load emits, before any coordinates exist -- and on a folder that already held
a ``correlation.json`` that wrote an empty state over it; the loader then read
the file it had just emptied. Auto-save is now armed only by a load or an
edit, never by an image.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("PyQt5")

from fibsem.correlation.structures import (  # noqa: E402
    Coordinate,
    CorrelationInputData,
    CorrelationState,
    PointType,
    PointXYZ,
)
from fibsem.structures import FibsemImage  # noqa: E402
from fibsem.ui.correlation.widgets.correlation_tab_widget import (  # noqa: E402
    CORRELATION_FILENAME,
    CorrelationTabWidget,
    load_project,
)


@pytest.fixture(autouse=True)
def _no_lut_download(monkeypatch):
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)


def _pairs(n: int) -> CorrelationInputData:
    return CorrelationInputData(
        fib_coordinates=[
            Coordinate(PointXYZ(10.0 * i, 20.0 * i, 0.0), PointType.FIB)
            for i in range(n)
        ],
        fm_coordinates=[
            Coordinate(PointXYZ(30.0 * i, 40.0 * i, 5.0), PointType.FM)
            for i in range(n)
        ],
    )


def _saved_run(folder, n_pairs: int) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6).save(
        str(folder / "ref_ib.tif")
    )
    path = folder / CORRELATION_FILENAME
    CorrelationState(input_data=_pairs(n_pairs)).save(str(path))
    return str(path)


@pytest.fixture
def widget(qapp):
    w = CorrelationTabWidget()
    yield w
    w.close()
    w.deleteLater()


def test_opening_a_saved_run_keeps_its_picks_on_disk_and_shows_them(widget, tmp_path):
    path = _saved_run(tmp_path / "run", 8)
    with open(path, "rb") as fh:
        before = fh.read()

    load_project(widget, str(tmp_path / "run"))

    assert len(widget.data.fib_coordinates) == 8
    assert len(widget.data.fm_coordinates) == 8
    with open(path) as fh:
        raw = json.load(fh)
    # the load may re-save the same state (adding the image's name and shape,
    # which the file did not have); the user's picks are what must survive
    for key in ("fib_coordinates", "fm_coordinates"):
        assert raw["input_data"][key] == json.loads(before)["input_data"][key]


def test_an_image_load_alone_never_writes(widget, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    widget.set_project_dir(str(run))
    widget.set_fib_image(
        FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)
    )
    assert not os.path.exists(run / CORRELATION_FILENAME)
    # an edit arms it
    widget._on_canvas_add_requested(5.0, 6.0, PointType.FIB)
    assert os.path.exists(run / CORRELATION_FILENAME)


def test_a_legacy_data_file_is_not_shadowed_by_an_empty_consolidated_file(
    widget, tmp_path
):
    run = tmp_path / "run"
    run.mkdir()
    FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6).save(
        str(run / "ref_ib.tif")
    )
    (run / "correlation_data.json").write_text(json.dumps(_pairs(5).to_dict()))

    load_project(widget, str(run))

    assert len(widget.data.fib_coordinates) == 5
    # the consolidated file now holds those five, or does not exist; never empty
    path = run / CORRELATION_FILENAME
    if path.exists():
        assert len(json.loads(path.read_text())["input_data"]["fib_coordinates"]) == 5


def test_changing_the_project_dir_disarms_until_the_next_load_or_edit(widget, tmp_path):
    first = tmp_path / "a"
    first.mkdir()
    widget.set_project_dir(str(first))
    widget.set_data(_pairs(4))  # a load arms it
    widget.data_changed.emit(widget.data)
    assert (first / CORRELATION_FILENAME).exists()

    second = tmp_path / "b"
    second.mkdir()
    widget.set_project_dir(str(second))
    widget.set_fib_image(
        FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6)
    )
    assert not (second / CORRELATION_FILENAME).exists()

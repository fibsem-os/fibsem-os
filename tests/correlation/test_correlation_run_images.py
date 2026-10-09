"""A correlation run is opened on the images it records, and only a result for
those images arms Continue (FIB-1237).

Opening a run folder used to guess its images -- the spot-burn reference and the
first stack in the lamella folder -- whatever the run recorded; the open then
rewrote the file with the guessed names; and a result counted as current on its
points alone, so Continue committed a target computed on one pair of images
against another.

Run directly (no display needed):
    QT_QPA_PLATFORM=offscreen python -m pytest tests/correlation/test_correlation_run_images.py
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from fibsem.correlation.structures import (
    Coordinate,
    CorrelationInputData,
    CorrelationPointOfInterest,
    CorrelationResult,
    CorrelationState,
    PointType,
    PointXYZ,
    image_stem,
    recorded_image_basename,
)
from fibsem.structures import FibsemImage, Point

_PX = 100e-6 / 300  # the blank images' pixel size: 100 um across 300 px


def _pairs(n: int = 4, **names) -> CorrelationInputData:
    return CorrelationInputData(
        fib_coordinates=[
            Coordinate(PointXYZ(10.0 * i, 20.0 * i, 0.0), PointType.FIB)
            for i in range(n)
        ],
        fm_coordinates=[
            Coordinate(PointXYZ(30.0 * i, 40.0 * i, 5.0), PointType.FM)
            for i in range(n)
        ],
        stored_fib_image_filename=names.get("fib"),
        stored_fm_image_filename=names.get("fm"),
        stored_fib_image_pixel_size=names.get("fib_px"),
    )


def _result(data: CorrelationInputData) -> CorrelationResult:
    return CorrelationResult(
        poi=[CorrelationPointOfInterest(image_px=Point(x=10.0, y=20.0))],
        rms_error=1.5,
        input_data=data,
    )


# ---------------------------------------------------------------------------
# What identifies an image across the forms a run has recorded it in
# ---------------------------------------------------------------------------


def test_a_windows_path_recorded_on_the_instrument_reduces_to_its_file_name():
    recorded = r"C:\Users\User\Desktop\2026\exp\01-fancy-mite\01-fancy-mite-zstack-18-56-36.ome.tiff"
    assert recorded_image_basename(recorded) == "01-fancy-mite-zstack-18-56-36.ome.tiff"


def test_an_old_stem_and_the_file_it_was_saved_as_are_the_same_image():
    stem = "ref_Rough Milling_final_res_02"
    assert image_stem(stem) == image_stem("ref_Rough Milling_final_res_02_ib.tif")
    assert image_stem("a-zstack.ome.tiff") == image_stem("a-zstack")


def test_the_electron_image_of_a_stage_is_not_its_ion_image():
    assert image_stem("ref_X_final_res_02_eb.tif") != image_stem(
        "ref_X_final_res_02_ib.tif"
    )


# ---------------------------------------------------------------------------
# A result is current only for the images it was computed on
# ---------------------------------------------------------------------------


def test_a_result_for_another_fib_image_is_not_current_and_says_which():
    result = _result(_pairs(fib="ref_Spot Burn Fiducial_final_res_02_ib.tif"))
    now = _pairs(fib="ref_Rough Milling_final_res_02_ib.tif")

    assert result.matches_inputs(now) is False
    reason = result.image_mismatch(now)
    assert "ref_Spot Burn Fiducial_final_res_02_ib.tif" in reason
    assert "ref_Rough Milling_final_res_02_ib.tif" in reason


def test_a_result_recorded_under_the_old_stem_is_current_for_its_file():
    result = _result(_pairs(fib="ref_Rough Milling_final_res_02"))
    assert result.matches_inputs(_pairs(fib="ref_Rough Milling_final_res_02_ib.tif"))


def test_a_side_that_records_no_name_is_not_evidence_of_a_mismatch():
    """Runs before FIB-1019 recorded no names; they must not all turn stale."""
    assert _result(_pairs()).matches_inputs(_pairs(fib="anything_ib.tif"))
    assert _result(_pairs(fib="anything_ib.tif")).matches_inputs(_pairs())


def test_a_result_at_another_fib_pixel_size_is_not_current():
    result = _result(_pairs(fib_px=_PX))
    assert result.matches_inputs(_pairs(fib_px=_PX)) is True
    assert result.matches_inputs(_pairs(fib_px=2 * _PX)) is False
    assert "FIB pixel size" in result.image_mismatch(_pairs(fib_px=2 * _PX))


# ---------------------------------------------------------------------------
# Opening a run folder
# ---------------------------------------------------------------------------


@pytest.fixture
def widget(qapp, monkeypatch):
    pytest.importorskip("PyQt5")
    import fibsem.ui.correlation.widgets.refractive_index_widget as riw
    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CorrelationTabWidget,
    )

    monkeypatch.setattr(riw, "_ensure_lut", lambda: None)
    w = CorrelationTabWidget()
    yield w
    w.close()
    w.deleteLater()


def _lamella(tmp_path):
    """An experiment with one lamella holding two FIB references, and a run
    folder below it -- the layout AutoLamella writes."""
    (tmp_path / "experiment.yaml").write_text("{}\n")
    lamella = tmp_path / "01-lamella"
    run = lamella / "Correlation" / "2026-10-09_12-00-00"
    run.mkdir(parents=True)
    for name in (
        "ref_Spot Burn Fiducial_final_res_02",
        "ref_Rough Milling_final_res_02",
    ):
        FibsemImage.generate_blank_image(resolution=(300, 200), hfw=100e-6).save(
            str(lamella / f"{name}_ib.tif")
        )
    return lamella, run


def _loaded_fib(w):
    return os.path.basename(w._fib_image.filepath) if w._fib_image else None


def test_opening_a_run_loads_the_fib_image_it_records_not_the_spot_burn_guess(
    widget, tmp_path
):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _, run = _lamella(tmp_path)
    # recorded as the old acquisition stem, as the 2026-07-29 runs are
    data = _pairs(fib="ref_Rough Milling_final_res_02")
    path = run / "correlation.json"
    CorrelationState(input_data=data, result=_result(data)).save(str(path))
    before = path.read_bytes()

    load_project(widget, str(run))

    assert _loaded_fib(widget) == "ref_Rough Milling_final_res_02_ib.tif"
    assert path.read_bytes() == before  # an open is not an edit
    assert widget._btn_continue.isEnabled() is True  # the right images: current


def test_a_recorded_image_that_is_missing_is_not_replaced_by_a_guess(widget, tmp_path):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _, run = _lamella(tmp_path)
    CorrelationState(input_data=_pairs(fib="ref_Gone_final_res_02")).save(
        str(run / "correlation.json")
    )

    load_project(widget, str(run))

    assert widget._fib_image is None
    assert len(widget.data.fib_coordinates) == 4  # the points still open


def test_a_run_that_records_no_image_still_opens_on_the_guess(widget, tmp_path):
    from fibsem.ui.correlation.widgets.correlation_tab_widget import load_project

    _, run = _lamella(tmp_path)
    CorrelationState(input_data=_pairs()).save(str(run / "correlation.json"))

    load_project(widget, str(run))

    assert _loaded_fib(widget) == "ref_Spot Burn Fiducial_final_res_02_ib.tif"


def test_loading_another_lamellas_file_does_not_arm_continue(widget, tmp_path):
    lamella, _ = _lamella(tmp_path)
    widget.set_fib_image(
        FibsemImage.load(str(lamella / "ref_Rough Milling_final_res_02_ib.tif"))
    )
    other = _pairs(fib="ref_Spot Burn Fiducial_final_res_02_ib.tif")
    path = tmp_path / "other.json"
    CorrelationState(input_data=other, result=_result(other)).save(str(path))

    widget.load_correlation(str(path))

    assert widget._btn_continue.isEnabled() is False
    assert "other images" in widget._lbl_status.text()
    assert "ref_Spot Burn Fiducial_final_res_02_ib.tif" in widget._lbl_status.text()


def test_continue_records_a_loaded_result_in_the_run_folder(
    widget, tmp_path, monkeypatch
):
    """Loading writes nothing, so a result committed straight after a load
    would otherwise leave this run's folder without a record of it."""
    from PyQt5.QtWidgets import QMessageBox

    from fibsem.ui.correlation.widgets.correlation_tab_widget import (
        CORRELATION_FILENAME,
    )

    source = tmp_path / "source.json"
    data = _pairs()
    CorrelationState(input_data=data, result=_result(data)).save(str(source))
    run = tmp_path / "run"
    run.mkdir()
    widget.set_project_dir(str(run))
    widget.load_correlation(str(source))
    assert not (run / CORRELATION_FILENAME).exists()

    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    monkeypatch.setattr(widget, "save_plot", lambda *a, **k: None)
    widget._on_continue_pressed()

    saved = CorrelationState.load(str(run / CORRELATION_FILENAME))
    assert saved.result is not None
    assert len(saved.input_data.fib_coordinates) == 4

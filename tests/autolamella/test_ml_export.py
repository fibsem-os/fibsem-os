"""Tests for exporting AutoLamella experiments to the targeting ML format (FIB-305).

The exported sample is one lamella: the final FIB reference image from its Select
Milling Position task, labelled with the point of interest the operator picked.

Headless: no Qt, no microscope, no napari.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from psygnal.containers import EventedDict

from fibsem.applications.autolamella.structures import (
    AutoLamellaTaskProtocol,
    DefectType,
    Experiment,
)
from fibsem.applications.autolamella.tools import ml_export
from fibsem.applications.autolamella.workflows.tasks.fiducial import (
    MillFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.rough import MillRoughTaskConfig
from fibsem.applications.autolamella.workflows.tasks.select_position import (
    SelectMillingPositionTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.spot_burn import (
    SpotBurnFiducialTaskConfig,
)
from fibsem.applications.autolamella.workflows.tasks.trench import MillTrenchTaskConfig
from fibsem.structures import (
    BeamType,
    FibsemImage,
    FibsemImageMetadata,
    FibsemStagePosition,
    ImageSettings,
    MicroscopeState,
    Point,
)

PIXELSIZE = 1e-7  # 100 nm/px
SHAPE = (512, 512)  # (height, width) -> centre at (256, 256)
TASK_NAME = "Select Milling Position"


def _stage(x=0.0, y=0.0, z=0.0, r=0.0, t=0.0) -> FibsemStagePosition:
    return FibsemStagePosition(x=x, y=y, z=z, r=r, t=t)


def _make_fib_image(shape=SHAPE, pixelsize=PIXELSIZE, hfw=100e-6) -> FibsemImage:
    metadata = FibsemImageMetadata(
        image_settings=ImageSettings(beam_type=BeamType.ION, hfw=hfw),
        pixel_size=Point(pixelsize, pixelsize),
        microscope_state=MicroscopeState(stage_position=_stage()),
    )
    return FibsemImage(data=np.zeros(shape, dtype=np.uint8), metadata=metadata)


def _task_config(task_name: str = TASK_NAME) -> EventedDict:
    return EventedDict(
        {task_name: SelectMillingPositionTaskConfig(task_name=task_name)}
    )


def _make_experiment(
    tmp_path: Path, n_lamella: int = 1, poi=None, task_name: str = TASK_NAME
) -> Experiment:
    exp = Experiment(path=tmp_path, name="test-exp")
    exp.task_protocol = AutoLamellaTaskProtocol()
    os.makedirs(exp.path, exist_ok=True)

    for _ in range(n_lamella):
        exp.add_new_lamella(
            MicroscopeState(stage_position=_stage()), _task_config(task_name)
        )
    for lamella in exp.positions:
        lamella.poi = poi if poi is not None else Point(0.0, 0.0)
    return exp


def _save_reference_image(lamella, image=None, res: int = 2, task_name=TASK_NAME):
    """Write a final-set FIB reference image where the exporter expects it."""
    os.makedirs(lamella.path, exist_ok=True)
    name = f"ref_{task_name}_final_res_{res:02d}_ib"
    (image or _make_fib_image()).save(os.path.join(str(lamella.path), name))
    return os.path.join(str(lamella.path), f"{name}.tif")


# ── locating the source image ────────────────────────────────────────────────


def test_task_names_for_type_finds_the_task(tmp_path):
    exp = _make_experiment(tmp_path)
    names = ml_export.task_names_for_type(exp.positions[0], "SELECT_MILLING_POSITION")
    assert names == [TASK_NAME]


def test_task_names_for_type_ignores_other_types(tmp_path):
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_config["Mill Trench"] = MillTrenchTaskConfig(task_name="Mill Trench")

    names = ml_export.task_names_for_type(lamella, "SELECT_MILLING_POSITION")
    assert names == [TASK_NAME]


def test_find_final_fib_image_prefers_the_most_zoomed(tmp_path):
    """res_NN is sorted largest FOV first, so the highest suffix is most zoomed."""
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    _save_reference_image(lamella, res=1)
    expected = _save_reference_image(lamella, res=2)

    assert ml_export.find_final_fib_image(lamella) == (
        expected,
        "SELECT_MILLING_POSITION",
    )


def test_find_final_fib_image_none_without_images(tmp_path):
    exp = _make_experiment(tmp_path)
    assert ml_export.find_final_fib_image(exp.positions[0]) is None


def test_find_final_fib_image_none_without_the_task(tmp_path):
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    _save_reference_image(lamella)
    lamella.task_config.clear()  # task never configured

    assert ml_export.find_final_fib_image(lamella) is None


def test_find_final_fib_image_ignores_sem_and_non_final(tmp_path):
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    os.makedirs(lamella.path, exist_ok=True)
    for name in (
        f"ref_{TASK_NAME}_final_res_01_eb",  # SEM, not FIB
        f"ref_{TASK_NAME}_start_ib",  # start of task, not final
        f"ref_{TASK_NAME}_post_tilt_ib",
    ):
        _make_fib_image().save(os.path.join(str(lamella.path), name))

    assert ml_export.find_final_fib_image(lamella) is None


# ── falling back to a later task ─────────────────────────────────────────────

FALLBACK_TASKS = [
    ("SPOT_BURN_FIDUCIAL", SpotBurnFiducialTaskConfig, "Spot Burn Fiducial"),
    ("MILL_FIDUCIAL", MillFiducialTaskConfig, "Mill Fiducial"),
    ("MILL_ROUGH", MillRoughTaskConfig, "Rough Milling"),
]


@pytest.mark.parametrize("task_type,config_cls,task_name", FALLBACK_TASKS)
def test_find_final_fib_image_falls_back_to_later_task(
    tmp_path, task_type, config_cls, task_name
):
    """With no Select Milling Position images, a later task's set is used."""
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_config[task_name] = config_cls(task_name=task_name)
    expected = _save_reference_image(lamella, task_name=task_name)

    assert ml_export.find_final_fib_image(lamella) == (expected, task_type)


def test_find_final_fib_image_prefers_select_position_over_fallbacks(tmp_path):
    """Preference order is earliest task first -- the least milled sample."""
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    for task_type, config_cls, task_name in FALLBACK_TASKS:
        lamella.task_config[task_name] = config_cls(task_name=task_name)
        _save_reference_image(lamella, task_name=task_name)
    expected = _save_reference_image(lamella)  # Select Milling Position

    assert ml_export.find_final_fib_image(lamella) == (
        expected,
        "SELECT_MILLING_POSITION",
    )


def test_find_final_fib_image_fallback_order_is_earliest_first(tmp_path):
    """Given only fiducial and rough images, the fiducial one wins."""
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_config["Rough Milling"] = MillRoughTaskConfig(
        task_name="Rough Milling"
    )
    _save_reference_image(lamella, task_name="Rough Milling")
    lamella.task_config["Mill Fiducial"] = MillFiducialTaskConfig(
        task_name="Mill Fiducial"
    )
    expected = _save_reference_image(lamella, task_name="Mill Fiducial")

    assert ml_export.find_final_fib_image(lamella) == (expected, "MILL_FIDUCIAL")


def test_export_records_which_task_the_image_came_from(tmp_path):
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.task_config["Rough Milling"] = MillRoughTaskConfig(
        task_name="Rough Milling"
    )
    _save_reference_image(lamella, task_name="Rough Milling")
    output = str(tmp_path / "out")

    ml_export.export_experiment(exp, output)

    with open(os.path.join(output, "points", "0001.json")) as f:
        assert json.load(f)["source_task"] == "MILL_ROUGH"
    with open(os.path.join(output, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest["samples"][0]["source_task"] == "MILL_ROUGH"
    assert manifest["n_samples_by_source_task"] == {"MILL_ROUGH": 1}


# ── POI -> pixels ────────────────────────────────────────────────────────────
# The POI is a milling coordinate (metres, y-up); pixels are y-down.


def test_poi_at_origin_is_image_centre(tmp_path):
    exp = _make_experiment(tmp_path, poi=Point(0.0, 0.0))
    point = ml_export.poi_to_pixels(exp.positions[0], _make_fib_image())

    assert point.x == pytest.approx(256.0)
    assert point.y == pytest.approx(256.0)


def test_poi_x_is_not_flipped(tmp_path):
    exp = _make_experiment(tmp_path, poi=Point(1e-6, 0.0))  # +10 px
    assert ml_export.poi_to_pixels(
        exp.positions[0], _make_fib_image()
    ).x == pytest.approx(266.0)


def test_poi_y_is_flipped(tmp_path):
    """+y in milling coordinates is *up*, which is -y in pixels."""
    exp = _make_experiment(tmp_path, poi=Point(0.0, 1e-6))  # -10 px
    assert ml_export.poi_to_pixels(
        exp.positions[0], _make_fib_image()
    ).y == pytest.approx(246.0)


def test_poi_conversion_is_independent_of_field_of_view(tmp_path):
    """The POI is held in metres, so a different pixel size still lands correctly."""
    exp = _make_experiment(tmp_path, poi=Point(2e-6, 0.0))
    coarse = ml_export.poi_to_pixels(exp.positions[0], _make_fib_image(pixelsize=2e-7))

    assert coarse.x == pytest.approx(256.0 + 10.0)  # 2 um / 200 nm = 10 px


def test_poi_requires_pixel_size(tmp_path):
    exp = _make_experiment(tmp_path)
    image = _make_fib_image()
    image.metadata.pixel_size = None

    with pytest.raises(ValueError, match="pixel size"):
        ml_export.poi_to_pixels(exp.positions[0], image)


# ── collect_sample ───────────────────────────────────────────────────────────


def test_collect_sample_places_the_point(tmp_path):
    exp = _make_experiment(tmp_path, poi=Point(1e-6, 2e-6))
    _save_reference_image(exp.positions[0])

    sample = ml_export.collect_sample(exp.positions[0], "0001")
    assert sample.pixel_x == pytest.approx(266.0)
    assert sample.pixel_y == pytest.approx(236.0)


def test_collect_sample_records_shape_pixelsize_hfw(tmp_path):
    exp = _make_experiment(tmp_path)
    _save_reference_image(exp.positions[0], _make_fib_image(hfw=80e-6))

    sample = ml_export.collect_sample(exp.positions[0], "0001")
    assert sample.shape == SHAPE
    assert sample.pixelsize == pytest.approx(PIXELSIZE)
    assert sample.hfw == pytest.approx(80e-6)


def test_collect_sample_carries_provenance(tmp_path):
    exp = _make_experiment(tmp_path)
    lamella = exp.positions[0]
    lamella.defect.state = DefectType.FAILURE
    lamella.milling_angle = 0.3
    _save_reference_image(lamella)

    sample = ml_export.collect_sample(lamella, "0001")
    assert sample.defect == "FAILURE"
    assert sample.milling_angle == pytest.approx(0.3)
    assert sample.lamella_id == lamella.id
    assert sample.petname == lamella.name


def test_collect_sample_none_without_image(tmp_path):
    exp = _make_experiment(tmp_path)
    assert ml_export.collect_sample(exp.positions[0], "0001") is None


# ── rasterise_point ──────────────────────────────────────────────────────────


def test_rasterise_stamps_disc_at_point():
    label = ml_export.rasterise_point(100.0, 200.0, SHAPE, PIXELSIZE, 1e-6)
    assert label[200, 100] == ml_export.LABEL_VALUE


def test_rasterise_disc_radius_matches_metres():
    """A 1 um radius at 100 nm/px is 10 px."""
    label = ml_export.rasterise_point(100.0, 100.0, SHAPE, PIXELSIZE, 1e-6)
    assert label[100, 110] == ml_export.LABEL_VALUE  # on the edge
    assert label[100, 112] == 0  # beyond it


def test_rasterise_is_binary():
    label = ml_export.rasterise_point(256.0, 256.0, SHAPE, PIXELSIZE, 1e-6)
    assert set(np.unique(label)) == {0, ml_export.LABEL_VALUE}
    assert label.dtype == ml_export.LABEL_DTYPE


def test_rasterise_clips_disc_at_image_edge():
    label = ml_export.rasterise_point(2.0, 2.0, SHAPE, PIXELSIZE, 1e-6)
    assert label[0, 0] == ml_export.LABEL_VALUE
    assert label.shape == SHAPE


# ── export_experiment ────────────────────────────────────────────────────────


def test_default_output_path_is_inside_the_experiment(tmp_path):
    exp = _make_experiment(tmp_path)
    assert ml_export.default_output_path(exp) == os.path.join(
        str(exp.path), "targeting-export"
    )


def test_export_defaults_into_the_experiment_directory(tmp_path):
    """No output_path -> <experiment>/targeting-export."""
    exp = _make_experiment(tmp_path)
    _save_reference_image(exp.positions[0])

    summary = ml_export.export_experiment(exp)

    expected = os.path.join(str(exp.path), "targeting-export")
    assert summary.output_path == expected
    assert os.path.isfile(os.path.join(expected, "images", "0001.tif"))
    assert os.path.isfile(os.path.join(expected, "manifest.json"))


def test_export_explicit_output_path_still_wins(tmp_path):
    exp = _make_experiment(tmp_path)
    _save_reference_image(exp.positions[0])
    elsewhere = str(tmp_path / "elsewhere")

    summary = ml_export.export_experiment(exp, elsewhere)

    assert summary.output_path == elsewhere
    assert os.path.isfile(os.path.join(elsewhere, "images", "0001.tif"))
    assert not os.path.exists(os.path.join(str(exp.path), "targeting-export"))


def test_export_default_does_not_pick_up_its_own_output(tmp_path):
    """Exporting twice must not treat the first export's images as lamella data."""
    exp = _make_experiment(tmp_path)
    _save_reference_image(exp.positions[0])

    first = ml_export.export_experiment(exp)
    second = ml_export.export_experiment(exp)

    assert first.n_samples == 1
    assert second.n_samples == 1  # not 2


def test_export_writes_full_layout(tmp_path):
    exp = _make_experiment(tmp_path, n_lamella=1)
    _save_reference_image(exp.positions[0])
    output = str(tmp_path / "out")

    summary = ml_export.export_experiment(exp, output)

    assert summary.n_samples == 1
    assert os.path.isfile(os.path.join(output, "images", "0001.tif"))
    assert os.path.isfile(os.path.join(output, "points", "0001.json"))
    assert os.path.isfile(os.path.join(output, "labels", "0001.tif"))


def test_export_one_sample_per_lamella(tmp_path):
    exp = _make_experiment(tmp_path, n_lamella=3)
    for lamella in exp.positions:
        _save_reference_image(lamella)

    summary = ml_export.export_experiment(exp, str(tmp_path / "out"))
    assert summary.n_samples == 3
    assert [s.stem for s in summary.samples] == ["0001", "0002", "0003"]


def test_export_skips_lamella_without_image(tmp_path):
    """Numbering stays contiguous across a skip."""
    exp = _make_experiment(tmp_path, n_lamella=3)
    _save_reference_image(exp.positions[0])
    _save_reference_image(exp.positions[2])  # middle lamella never ran the task

    summary = ml_export.export_experiment(exp, str(tmp_path / "out"))
    assert summary.n_samples == 2
    assert [s.stem for s in summary.samples] == ["0001", "0002"]
    assert any(exp.positions[1].name in reason for reason in summary.skipped)


def test_export_sidecar_holds_subpixel_coordinates(tmp_path):
    exp = _make_experiment(tmp_path, poi=Point(1.5e-7, 0.0))  # 1.5 px
    _save_reference_image(exp.positions[0])
    output = str(tmp_path / "out")

    ml_export.export_experiment(exp, output)
    with open(os.path.join(output, "points", "0001.json")) as f:
        data = json.load(f)

    assert data["pixel_x"] == pytest.approx(257.5)
    assert isinstance(data["pixel_x"], float)


def test_export_label_disc_lands_on_sidecar_coordinates(tmp_path):
    """The round-trip that keeps points/ and labels/ from drifting apart."""
    import tifffile

    exp = _make_experiment(tmp_path, poi=Point(5e-6, 3e-6))
    _save_reference_image(exp.positions[0])
    output = str(tmp_path / "out")

    ml_export.export_experiment(exp, output)
    with open(os.path.join(output, "points", "0001.json")) as f:
        data = json.load(f)
    label = tifffile.imread(os.path.join(output, "labels", "0001.tif"))

    assert (
        label[round(data["pixel_y"]), round(data["pixel_x"])] == ml_export.LABEL_VALUE
    )


def test_export_writes_manifest_indexing_samples(tmp_path):
    exp = _make_experiment(tmp_path, n_lamella=2)
    for lamella in exp.positions:
        _save_reference_image(lamella)
    output = str(tmp_path / "out")

    ml_export.export_experiment(exp, output)
    with open(os.path.join(output, "manifest.json")) as f:
        manifest = json.load(f)

    assert manifest["format"] == "autolamella-targeting-points-v1"
    assert manifest["source_task_preference"][0] == "SELECT_MILLING_POSITION"
    assert manifest["n_samples"] == 2
    assert manifest["samples"][0]["image"] == "images/0001.tif"
    assert manifest["samples"][0]["points"] == "points/0001.json"


def test_export_manifest_records_skips(tmp_path):
    exp = _make_experiment(tmp_path, n_lamella=1)  # no reference image saved
    output = str(tmp_path / "out")

    ml_export.export_experiment(exp, output)
    with open(os.path.join(output, "manifest.json")) as f:
        manifest = json.load(f)

    assert manifest["n_samples"] == 0
    assert any("no final FIB reference image" in r for r in manifest["skipped"])


def test_export_start_index_continues_numbering(tmp_path):
    exp = _make_experiment(tmp_path, n_lamella=1)
    _save_reference_image(exp.positions[0])
    output = str(tmp_path / "out")

    summary = ml_export.export_experiment(exp, output, start_index=7)
    assert summary.samples[0].stem == "0007"
    assert os.path.isfile(os.path.join(output, "images", "0007.tif"))


def test_export_experiments_numbers_across_experiments(tmp_path):
    output = str(tmp_path / "out")
    paths = []
    for name in ("exp-a", "exp-b"):
        exp = Experiment(path=tmp_path / name, name=name)
        exp.task_protocol = AutoLamellaTaskProtocol()
        os.makedirs(exp.path, exist_ok=True)
        exp.add_new_lamella(MicroscopeState(stage_position=_stage()), _task_config())
        exp.positions[0].poi = Point(0.0, 0.0)
        _save_reference_image(exp.positions[0])
        exp.save()
        paths.append(exp.path)

    summary = ml_export.export_experiments(paths, output)

    assert summary.n_samples == 2
    assert [s.stem for s in summary.samples] == ["0001", "0002"]
    assert os.path.isfile(os.path.join(output, "images", "0002.tif"))


def test_export_experiments_without_output_uses_each_experiment_directory(tmp_path):
    """No output_path -> each experiment gets its own targeting-export, numbered from 1."""
    paths = []
    for name in ("exp-a", "exp-b"):
        exp = Experiment(path=tmp_path / name, name=name)
        exp.task_protocol = AutoLamellaTaskProtocol()
        os.makedirs(exp.path, exist_ok=True)
        exp.add_new_lamella(MicroscopeState(stage_position=_stage()), _task_config())
        exp.positions[0].poi = Point(0.0, 0.0)
        _save_reference_image(exp.positions[0])
        exp.save()
        paths.append(exp.path)

    summary = ml_export.export_experiments(paths)

    assert summary.n_samples == 2
    for path in paths:
        own = os.path.join(path, "targeting-export")
        assert os.path.isfile(os.path.join(own, "images", "0001.tif"))
        assert os.path.isfile(os.path.join(own, "manifest.json"))


# ── label radius ─────────────────────────────────────────────────────────────


def test_export_radius_controls_label_size(tmp_path):
    """Changing the label geometry means re-exporting at a different radius_m."""
    import tifffile

    exp = _make_experiment(tmp_path)
    _save_reference_image(exp.positions[0])

    ml_export.export_experiment(exp, str(tmp_path / "small"), radius_m=1e-6)
    ml_export.export_experiment(exp, str(tmp_path / "large"), radius_m=3e-6)

    small = tifffile.imread(str(tmp_path / "small" / "labels" / "0001.tif"))
    large = tifffile.imread(str(tmp_path / "large" / "labels" / "0001.tif"))

    assert (large > 0).sum() > (small > 0).sum()
